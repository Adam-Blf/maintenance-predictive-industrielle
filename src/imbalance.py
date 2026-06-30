# -*- coding: utf-8 -*-
"""Module de gestion du desequilibre de classes.

Contexte metier
---------------
La detection de pannes industrielles est un probleme de classification
binaire FORTEMENT desequilibre : les pannes sont rares (minorite) par
rapport aux periodes de fonctionnement normal (majorite). Ce desequilibre
cause trois problemes :
  - Un modele naive qui predit toujours "pas de panne" obtient une accuracy
    elevee mais un Recall=0, ce qui est catastrophique en contexte industriel.
  - Les algorithmes standard optimisent l'accuracy, pas le Recall.
  - La cross-validation standard ne preserve pas les proportions de classes.

Solution
--------
Ce module expose quatre fonctions :
  - analyze_imbalance   : quantifie le desequilibre et ses effets.
  - apply_resampling    : reequilibre le train set (SMOTE / over / under).
  - optimize_threshold  : cherche le seuil de decision optimal.
  - compare_strategies  : genere un tableau recapitulatif multi-strategies.

Reference : Chawla et al. (2002) "SMOTE: Synthetic Minority Over-sampling
Technique" - Journal of Artificial Intelligence Research.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
)

try:
    from imblearn.over_sampling import RandomOverSampler, SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False


ResamplingMethod = Literal["random_over", "smote", "random_under", "none"]


# ---------------------------------------------------------------------------
# 1. Analyse prealable du desequilibre
# ---------------------------------------------------------------------------

def analyze_imbalance(
    y: pd.Series | np.ndarray,
    plot: bool = True,
    output_dir: Path | None = None,
) -> dict:
    """Analyse le desequilibre d'une variable cible binaire.

    Calcule le ratio majoritaire/minoritaire, montre pourquoi l'accuracy
    seule est trompeuse, et optionnellement trace un diagramme en barres.

    Parameters
    ----------
    y : pd.Series | np.ndarray
        Variable cible (0 = pas de panne, 1 = panne dans 24h).
    plot : bool
        Si True, affiche un diagramme de distribution des classes.
    output_dir : Path | None
        Si fourni, sauvegarde la figure dans ce dossier.

    Returns
    -------
    dict avec cles :
        count_majority, count_minority, ratio, accuracy_naive,
        classes, label_majority, label_minority
    """
    y_arr = np.array(y)
    values, counts = np.unique(y_arr, return_counts=True)

    majority_idx = int(np.argmax(counts))
    minority_idx = 1 - majority_idx

    count_majority = int(counts[majority_idx])
    count_minority = int(counts[minority_idx])
    label_majority = int(values[majority_idx])
    label_minority = int(values[minority_idx])

    ratio = count_majority / count_minority
    accuracy_naive = count_majority / len(y_arr)

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(
            "Analyse du desequilibre de classes", fontsize=14, fontweight="bold"
        )

        # Barplot distribution
        axes[0].bar(
            ["Sans panne (0)", "Panne dans 24h (1)"],
            [count_majority, count_minority],
            color=["#43A047", "#E53935"],
            edgecolor="white",
        )
        axes[0].set_title("Distribution des classes")
        axes[0].set_ylabel("Nombre d'observations")
        for bar, count in zip(axes[0].patches, [count_majority, count_minority]):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 50,
                f"{count:,}\n({count / len(y_arr) * 100:.1f}%)",
                ha="center", va="bottom", fontsize=11,
            )

        # Texte explicatif
        axes[1].axis("off")
        axes[1].text(
            0.1, 0.85,
            f"Ratio desequilibre : {ratio:.1f}:1",
            fontsize=13, fontweight="bold", transform=axes[1].transAxes,
        )
        axes[1].text(
            0.1, 0.70,
            f"Classe majoritaire (0) : {count_majority:,} obs ({accuracy_naive * 100:.1f}%)",
            fontsize=11, transform=axes[1].transAxes,
        )
        axes[1].text(
            0.1, 0.57,
            f"Classe minoritaire (1) : {count_minority:,} obs ({(1 - accuracy_naive) * 100:.1f}%)",
            fontsize=11, transform=axes[1].transAxes,
        )
        axes[1].text(
            0.1, 0.42,
            f"Accuracy d'un modele naif : {accuracy_naive * 100:.1f}%",
            fontsize=11, color="#E53935", transform=axes[1].transAxes,
        )
        axes[1].text(
            0.1, 0.30,
            "=> Attention : accuracy elevee mais Recall=0\n"
            "   Toutes les pannes sont manquees !",
            fontsize=10, color="#E53935", style="italic",
            transform=axes[1].transAxes,
        )
        axes[1].text(
            0.1, 0.15,
            "Metriques adaptees : F1, Recall, PR-AUC",
            fontsize=11, fontweight="bold", color="#163767",
            transform=axes[1].transAxes,
        )

        plt.tight_layout()
        if output_dir:
            out_path = Path(output_dir) / "imbalance_analysis.png"
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.show()

    return {
        "count_majority": count_majority,
        "count_minority": count_minority,
        "ratio": round(ratio, 2),
        "accuracy_naive": round(accuracy_naive, 4),
        "label_majority": label_majority,
        "label_minority": label_minority,
        "total": len(y_arr),
    }


# ---------------------------------------------------------------------------
# 2. Reequilibrage data-level
# ---------------------------------------------------------------------------

def apply_resampling(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    method: ResamplingMethod = "smote",
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Reequilibre le train set via sur- ou sous-echantillonnage.

    IMPORTANT : n'appeler que sur le TRAIN SET. Appliquer sur le test set
    introduirait une fuite d'information et gonflerait artificiellement les
    metriques.

    Parameters
    ----------
    X_train : array-like
        Features du train set uniquement.
    y_train : array-like
        Cible du train set uniquement.
    method : {"random_over", "smote", "random_under", "none"}
        Technique de reequilibrage :
        - "random_over"   : duplication aleatoire des exemples minoritaires.
        - "smote"         : generation d'exemples synthetiques (interpolation).
        - "random_under"  : suppression aleatoire des exemples majoritaires.
        - "none"          : aucune modification (retourne les donnees d'origine).
    random_state : int
        Graine pour la reproductibilite.

    Returns
    -------
    X_resampled, y_resampled : np.ndarray
        Train set reequilibre.
    """
    if not IMBLEARN_AVAILABLE and method != "none":
        raise ImportError(
            "imbalanced-learn requis : pip install imbalanced-learn>=0.11"
        )

    X = np.array(X_train)
    y = np.array(y_train)

    if method == "none":
        return X, y

    samplers = {
        "random_over": RandomOverSampler(random_state=random_state),
        "smote": SMOTE(random_state=random_state, k_neighbors=5),
        "random_under": RandomUnderSampler(random_state=random_state),
    }

    sampler = samplers[method]
    X_res, y_res = sampler.fit_resample(X, y)

    before = dict(zip(*np.unique(y, return_counts=True)))
    after = dict(zip(*np.unique(y_res, return_counts=True)))
    print(
        f"[{method}] avant={before} -> apres={after} "
        f"(+{len(y_res) - len(y):+,} observations)"
    )
    return X_res, y_res


# ---------------------------------------------------------------------------
# 3. Ajustement du seuil de decision
# ---------------------------------------------------------------------------

def optimize_threshold(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    metric: Literal["f1", "recall"] = "f1",
    output_dir: Path | None = None,
) -> dict:
    """Cherche le seuil de decision optimal pour maximiser f1 ou recall.

    Par defaut les classifieurs utilisent 0.5 comme seuil de decision.
    Dans un contexte desequilibre, ce seuil sous-estime les pannes.
    On explore [0.05, 0.95] par pas de 0.05 pour trouver le seuil optimal.

    Parameters
    ----------
    model : sklearn estimator
        Modele entraine avec methode predict_proba.
    X_test : np.ndarray
        Features du test set.
    y_test : np.ndarray
        Labels du test set.
    metric : {"f1", "recall"}
        Metrique a maximiser. "recall" favorise la detection de toutes les
        pannes (moins de faux negatifs). "f1" equilibre precision et recall.
    output_dir : Path | None
        Dossier ou sauvegarder la courbe de seuil.

    Returns
    -------
    dict avec cles :
        optimal_threshold, optimal_score, metric, f1_at_0_5,
        recall_at_0_5, precision_at_0_5, fn_at_0_5, fn_at_optimal
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    thresholds = np.arange(0.05, 0.96, 0.05)
    scores = []

    for thr in thresholds:
        y_pred_thr = (y_proba >= thr).astype(int)
        if metric == "f1":
            s = f1_score(y_test, y_pred_thr, zero_division=0)
        else:
            s = recall_score(y_test, y_pred_thr, zero_division=0)
        scores.append(s)

    best_idx = int(np.argmax(scores))
    optimal_thr = float(thresholds[best_idx])
    optimal_score = float(scores[best_idx])

    # Stats au seuil par defaut (0.5) pour comparaison
    y_default = (y_proba >= 0.5).astype(int)
    y_optimal = (y_proba >= optimal_thr).astype(int)

    cm_default = confusion_matrix(y_test, y_default)
    cm_optimal = confusion_matrix(y_test, y_optimal)

    fn_default = int(cm_default[1, 0]) if cm_default.shape == (2, 2) else 0
    fn_optimal = int(cm_optimal[1, 0]) if cm_optimal.shape == (2, 2) else 0

    if output_dir:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Optimisation du seuil de decision", fontsize=13, fontweight="bold")

        # Courbe metrique vs seuil
        axes[0].plot(thresholds, scores, color="#163767", linewidth=2.5, marker="o", markersize=4)
        axes[0].axvline(optimal_thr, color="#E53935", linestyle="--", label=f"Seuil optimal={optimal_thr:.2f}")
        axes[0].axvline(0.5, color="#FB8C00", linestyle=":", label="Seuil defaut=0.50")
        axes[0].set_xlabel("Seuil de decision")
        axes[0].set_ylabel(metric.upper())
        axes[0].set_title(f"Courbe {metric.upper()} vs seuil")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Comparaison FN seuil defaut vs optimal
        axes[1].bar(
            ["Seuil defaut (0.50)", f"Seuil optimal ({optimal_thr:.2f})"],
            [fn_default, fn_optimal],
            color=["#FB8C00", "#43A047"],
        )
        axes[1].set_title("Faux negatifs (pannes non detectees)")
        axes[1].set_ylabel("Nombre de FN")
        for bar, val in zip(axes[1].patches, [fn_default, fn_optimal]):
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                str(val), ha="center", fontsize=12, fontweight="bold",
            )

        plt.tight_layout()
        plt.savefig(Path(output_dir) / "threshold_optimization.png", dpi=150, bbox_inches="tight")
        plt.show()

    result = {
        "optimal_threshold": optimal_thr,
        "optimal_score": round(optimal_score, 4),
        "metric": metric,
        "f1_at_0_5": round(f1_score(y_test, y_default, zero_division=0), 4),
        "recall_at_0_5": round(recall_score(y_test, y_default, zero_division=0), 4),
        "fn_at_0_5": fn_default,
        "fn_at_optimal": fn_optimal,
    }

    if output_dir:
        with open(Path(output_dir) / "optimal_threshold.json", "w") as f:
            json.dump(result, f, indent=2)

    return result


# ---------------------------------------------------------------------------
# 4. Tableau recapitulatif multi-strategies
# ---------------------------------------------------------------------------

def compare_strategies(results_dict: dict) -> pd.DataFrame:
    """Genere un tableau recapitulatif des strategies de reequilibrage.

    Parameters
    ----------
    results_dict : dict
        Dictionnaire de la forme :
        {
          "Strategy Name": {
            "recall": float, "precision": float, "f1": float,
            "roc_auc": float, "pr_auc": float, "fit_time": float
          }, ...
        }

    Returns
    -------
    pd.DataFrame
        Tableau trie par Recall decroissant, avec colonnes formatees.
    """
    rows = []
    for strategy, metrics in results_dict.items():
        rows.append({
            "Strategie": strategy,
            "Recall": round(metrics.get("recall", 0), 4),
            "Precision": round(metrics.get("precision", 0), 4),
            "F1": round(metrics.get("f1", 0), 4),
            "ROC-AUC": round(metrics.get("roc_auc", 0), 4),
            "PR-AUC": round(metrics.get("pr_auc", 0), 4),
            "Temps (s)": round(metrics.get("fit_time", 0), 2),
        })

    df = pd.DataFrame(rows).sort_values("Recall", ascending=False).reset_index(drop=True)
    return df


def plot_pr_curves(
    pr_curves_dict: dict,
    output_dir: Path | None = None,
) -> None:
    """Trace les courbes Precision-Recall pour plusieurs strategies.

    Parameters
    ----------
    pr_curves_dict : dict
        {strategy_name: {"precision": array, "recall": array, "pr_auc": float}}
    output_dir : Path | None
        Dossier de sauvegarde.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#163767", "#FF43B8", "#E53935", "#43A047", "#0C78B4", "#FB8C00", "#9C27B0"]

    for i, (name, data) in enumerate(pr_curves_dict.items()):
        pr_auc_val = data.get("pr_auc", auc(data["recall"], data["precision"]))
        ax.plot(
            data["recall"], data["precision"],
            label=f"{name} (PR-AUC={pr_auc_val:.3f})",
            color=colors[i % len(colors)],
            linewidth=2,
        )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Courbes Precision-Recall par strategie de reequilibrage")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    if output_dir:
        plt.savefig(Path(output_dir) / "pr_curves_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
