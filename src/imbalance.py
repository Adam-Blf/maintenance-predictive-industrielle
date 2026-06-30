# -*- coding: utf-8 -*-
"""Module de gestion du desequilibre de classes.

Contexte metier
---------------
La detection de pannes industrielles est un probleme de classification
binaire FORTEMENT desequilibre : les pannes sont rares (minorite) par
rapport aux periodes de fonctionnement normal (majorite). Ce desequilibre
cause trois problemes concrets :

  1. Un modele naif qui predit toujours "pas de panne" obtient une accuracy
     elevee (~90%) mais un Recall=0 - toutes les pannes sont manquees.
  2. Les algorithmes standard optimisent l'accuracy globale, pas le Recall.
  3. La cross-validation standard ne preserve pas les proportions de classes
     sans StratifiedKFold.

Cinq strategies comparees
--------------------------
1. BASELINE (class_weight="balanced")
   Sklearn pondere la perte de chaque exemple inversement a sa frequence.
   Un exemple de panne "pese" autant que N exemples normaux (N = ratio).
   Simple, sans modification des donnees, parfois insuffisant sur IR > 15.

2. SMOTE (Synthetic Minority Oversampling TEchnique) - Chawla et al. 2002.
   Generation d'exemples SYNTHETIQUES de la classe minoritaire par
   interpolation entre voisins proches dans l'espace des features.
   Evite la duplication exacte (overfitting pur du Random OverSampler).
   Fonctionne apres preprocessing (espace numerique standardise).

3. ADASYN (ADaptive SYNthetic Sampling) - He et al. 2008.
   Variante de SMOTE concentree sur les zones de frontiere difficiles
   (exemples minoritaires entoures de majoritaires). Plus d'exemples
   synthetiques la ou le modele risque de se tromper.

4. RandomUnderSampler.
   Suppression aleatoire d'exemples majoritaires jusqu'a equilibre.
   Rapide, mais on perd de l'information (19k lignes sur 24k ici).

5. SMOTE + Tomek Links (SMOTETomek) - hybride.
   Phase 1 (SMOTE) : oversample la minorite.
   Phase 2 (Tomek) : supprime les paires ambigues a la frontiere.
   Resultat : dataset equilibre ET propre aux frontières de decision.

Architecture
------------
Les fonctions de ce module s'appuient sur imblearn.pipeline.Pipeline
(pas sklearn) qui autorise les resamplers entre transformateurs et
estimateur. Le resampler est IGNORE lors de predict/predict_proba -
garantie qu'on ne resamle jamais le test set.

    imblearn.pipeline.Pipeline([
        ("preprocessor", build_preprocessor()),   # ColumnTransformer sklearn
        ("resampler",    SMOTE()),                 # seulement sur train
        ("classifier",   RandomForestClassifier()),
    ])

API publique
------------
- analyze_imbalance         : stats du desequilibre (ratio, IR, counts).
- build_strategy_pipeline   : construit une imblearn.Pipeline par strategie.
- evaluate_strategy         : fit + predict + metriques sur test set.
- compare_all_strategies    : boucle sur les 5 strategies, retourne DataFrame.
- plot_class_distribution   : barplot de la distribution initiale.
- plot_pr_comparison        : courbes Precision/Rappel par strategie.
- plot_metrics_comparison   : barplot comparatif F1/Recall/Precision/ROC-AUC.
- plot_fit_time_comparison  : barplot horizontal du cout computationnel.
- optimize_threshold        : seuil de decision optimal (f1 ou recall).
- apply_resampling          : reequilibre raw arrays (compat. ancienne API).

Reference
---------
Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002).
SMOTE: Synthetic minority over-sampling technique.
Journal of Artificial Intelligence Research, 16, 321-357.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import (
    COLOR_ALERT_RED,
    COLOR_EFREI_BLUE,
    COLOR_EFREI_BRIGHT,
    COLOR_OK_GREEN,
    COLOR_WARNING,
    RANDOM_STATE,
)
from .preprocessing import build_preprocessor


# ---------------------------------------------------------------------------
# Palette des strategies (couleur + label FR lisible dans les graphiques)
# ---------------------------------------------------------------------------
STRATEGY_META: dict[str, dict[str, str]] = {
    "baseline":    {"label": "Baseline (class_weight)", "color": COLOR_EFREI_BLUE},
    "smote":       {"label": "SMOTE",                   "color": COLOR_EFREI_BRIGHT},
    "adasyn":      {"label": "ADASYN",                  "color": COLOR_OK_GREEN},
    "undersample": {"label": "Under-sampling",          "color": COLOR_WARNING},
    "smote_tomek": {"label": "SMOTE + Tomek Links",     "color": COLOR_ALERT_RED},
}

# Hyperparametres RF identiques pour toutes les strategies : on veut mesurer
# l'effet du resampling uniquement, pas un tuning cache.
_RF_PARAMS: dict = {
    "n_estimators": 200,
    "max_depth": 12,
    "min_samples_leaf": 4,
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}

ResamplingMethod = Literal["random_over", "smote", "random_under", "none"]


# ---------------------------------------------------------------------------
# 1. Analyse prealable du desequilibre
# ---------------------------------------------------------------------------

def analyze_imbalance(
    y: pd.Series | np.ndarray,
    output_dir: Path | None = None,
) -> dict:
    """Calcule les statistiques de desequilibre de la variable cible.

    Parameters
    ----------
    y : pd.Series | np.ndarray
        Vecteur cible binaire (0 = sain, 1 = panne dans 24h).
    output_dir : Path | None
        Si fourni, sauvegarde un barplot de distribution dans ce dossier.

    Returns
    -------
    dict avec les cles :
        count_majority, count_minority, ratio, accuracy_naive,
        label_majority, label_minority, total.
    """
    y_arr = np.asarray(y)
    values, counts = np.unique(y_arr, return_counts=True)

    majority_idx = int(np.argmax(counts))
    minority_idx = 1 - majority_idx

    count_majority = int(counts[majority_idx])
    count_minority = int(counts[minority_idx])
    label_majority = int(values[majority_idx])
    label_minority = int(values[minority_idx])

    total = len(y_arr)
    ratio = count_majority / count_minority if count_minority else float("inf")
    accuracy_naive = count_majority / total

    if output_dir is not None:
        _plot_distribution_internal(
            count_majority, count_minority, total, ratio, accuracy_naive,
            output_dir / "imbalance_analysis.png",
        )

    return {
        "count_majority": count_majority,
        "count_minority": count_minority,
        "ratio": round(ratio, 2),
        "accuracy_naive": round(accuracy_naive, 4),
        "label_majority": label_majority,
        "label_minority": label_minority,
        "total": total,
    }


def _plot_distribution_internal(
    count_maj: int,
    count_min: int,
    total: int,
    ratio: float,
    accuracy_naive: float,
    out_path: Path,
) -> None:
    """Sauvegarde le barplot de distribution en mode headless (sans plt.show)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "Analyse du desequilibre de classes", fontsize=14, fontweight="bold"
    )

    axes[0].bar(
        ["Sans panne (0)", "Panne dans 24h (1)"],
        [count_maj, count_min],
        color=[COLOR_OK_GREEN, COLOR_ALERT_RED],
        edgecolor="white",
    )
    axes[0].set_title("Distribution des classes")
    axes[0].set_ylabel("Nombre d'observations")
    for bar, count in zip(axes[0].patches, [count_maj, count_min]):
        pct = count / total * 100
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 50,
            f"{count:,}\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=11,
        )
    axes[0].spines[["top", "right"]].set_visible(False)
    axes[0].grid(axis="y", alpha=0.3, linestyle="--")

    axes[1].axis("off")
    txt = [
        (0.85, f"Imbalance Ratio : {ratio:.1f}:1", "bold", 13, "#222"),
        (0.70, f"Classe majoritaire (0) : {count_maj:,} ({100 * count_maj / total:.1f}%)", "normal", 11, "#222"),
        (0.57, f"Classe minoritaire (1) : {count_min:,} ({100 * count_min / total:.1f}%)", "normal", 11, "#222"),
        (0.42, f"Accuracy d'un modele naif : {accuracy_naive * 100:.1f}% (Recall=0 !)", "normal", 11, COLOR_ALERT_RED),
        (0.30, "Metriques adaptees : F1, Recall, PR-AUC", "bold", 11, COLOR_EFREI_BLUE),
    ]
    for y_pos, text, weight, size, color in txt:
        axes[1].text(0.05, y_pos, text, fontsize=size, fontweight=weight,
                     color=color, transform=axes[1].transAxes)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] Distribution sauvegardee -> {out_path}")


# ---------------------------------------------------------------------------
# 2. Construction des pipelines par strategie (imblearn)
# ---------------------------------------------------------------------------

def build_strategy_pipeline(strategy: str) -> ImbPipeline:
    """Construit une imblearn.Pipeline complete pour une strategie donnee.

    La pipeline enchaine :
      preprocessor (ColumnTransformer sklearn) -> [resampler] -> RF.

    Le resampler n'est applique QUE pendant le fit (imblearn le gere).
    Lors du predict/predict_proba, la pipeline ignore le resampler -
    garantie que le test set n'est jamais resamle.

    Parameters
    ----------
    strategy : str
        L'une de : "baseline", "smote", "adasyn", "undersample", "smote_tomek".

    Returns
    -------
    imblearn.pipeline.Pipeline

    Raises
    ------
    ValueError
        Si la strategie n'est pas reconnue.
    """
    preprocessor = build_preprocessor()

    if strategy == "baseline":
        # class_weight="balanced" : poids_classe_i = n_total / (n_classes * n_i).
        # Pas de resampling - strategie de reference deja utilisee dans script 03.
        return ImbPipeline([
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(**_RF_PARAMS, class_weight="balanced")),
        ])

    if strategy == "smote":
        return ImbPipeline([
            ("preprocessor", preprocessor),
            ("resampler", SMOTE(k_neighbors=5, sampling_strategy="auto",
                                random_state=RANDOM_STATE)),
            ("classifier", RandomForestClassifier(**_RF_PARAMS)),
        ])

    if strategy == "adasyn":
        return ImbPipeline([
            ("preprocessor", preprocessor),
            ("resampler", ADASYN(n_neighbors=5, sampling_strategy="auto",
                                 random_state=RANDOM_STATE)),
            ("classifier", RandomForestClassifier(**_RF_PARAMS)),
        ])

    if strategy == "undersample":
        return ImbPipeline([
            ("preprocessor", preprocessor),
            ("resampler", RandomUnderSampler(sampling_strategy="auto",
                                             random_state=RANDOM_STATE)),
            ("classifier", RandomForestClassifier(**_RF_PARAMS)),
        ])

    if strategy == "smote_tomek":
        return ImbPipeline([
            ("preprocessor", preprocessor),
            ("resampler", SMOTETomek(random_state=RANDOM_STATE)),
            ("classifier", RandomForestClassifier(**_RF_PARAMS)),
        ])

    raise ValueError(
        f"Strategie inconnue : '{strategy}'. "
        f"Choisir parmi {list(STRATEGY_META)}."
    )


# ---------------------------------------------------------------------------
# 3. Dataclass des metriques (1 ligne par strategie dans le tableau final)
# ---------------------------------------------------------------------------

@dataclass
class ImbalanceMetrics:
    """Metriques d'evaluation pour une strategie de reequilibrage.

    Attributs
    ---------
    strategy  : identifiant technique.
    label     : libelle FR pour les graphiques.
    precision : parmi les alertes, fraction de vraies pannes.
    recall    : parmi les pannes, fraction detectee.
    f1        : moyenne harmonique precision/recall.
    roc_auc   : aire sous la courbe ROC.
    pr_auc    : aire sous la courbe Precision/Rappel.
    fit_time_s: duree d'entrainement en secondes.
    """

    strategy: str
    label: str
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    fit_time_s: float

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# 4. Evaluation d'une strategie
# ---------------------------------------------------------------------------

def evaluate_strategy(
    strategy: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> ImbalanceMetrics:
    """Entraine et evalue une strategie sur le test set.

    Parameters
    ----------
    strategy : str
        Identifiant de la strategie (cle de STRATEGY_META).
    X_train, y_train : DataFrame/Series d'entrainement (non preprocesses).
    X_test, y_test   : DataFrame/Series de test (non preprocesses).

    Returns
    -------
    ImbalanceMetrics
        Metriques de la strategie sur le test set.
    """
    pipeline = build_strategy_pipeline(strategy)

    t0 = time.perf_counter()
    pipeline.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    return ImbalanceMetrics(
        strategy=strategy,
        label=STRATEGY_META[strategy]["label"],
        precision=precision_score(y_test, y_pred, zero_division=0),
        recall=recall_score(y_test, y_pred, zero_division=0),
        f1=f1_score(y_test, y_pred, zero_division=0),
        roc_auc=roc_auc_score(y_test, y_proba),
        pr_auc=average_precision_score(y_test, y_proba),
        fit_time_s=fit_time,
    )


# ---------------------------------------------------------------------------
# 5. Comparaison de toutes les strategies
# ---------------------------------------------------------------------------

def compare_all_strategies(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    verbose: bool = True,
) -> pd.DataFrame:
    """Entraine et evalue les 5 strategies, retourne un DataFrame de synthese.

    Parameters
    ----------
    X_train, y_train : donnees d'entrainement (non preprocessees).
    X_test, y_test   : donnees de test (jamais touchees par les resamplers).
    verbose          : affiche progression console.

    Returns
    -------
    pd.DataFrame trie par F1 decroissant, une ligne par strategie.
    """
    results: list[dict] = []
    for strategy in STRATEGY_META:
        if verbose:
            print(f"  [{strategy:12s}] Entrainement...", end=" ", flush=True)
        metrics = evaluate_strategy(strategy, X_train, y_train, X_test, y_test)
        results.append(metrics.to_dict())
        if verbose:
            print(
                f"F1={metrics.f1:.3f} | Recall={metrics.recall:.3f} "
                f"| PR-AUC={metrics.pr_auc:.3f} | {metrics.fit_time_s:.1f}s"
            )
    return pd.DataFrame(results).sort_values("f1", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 6. Visualisations (toutes headless : savefig uniquement, jamais plt.show)
# ---------------------------------------------------------------------------

def plot_class_distribution(y_train: pd.Series, out_path: Path) -> None:
    """Barplot de la distribution de classe avant reequilibrage.

    Delegation vers _plot_distribution_internal avec calcul des stats.
    """
    stats = analyze_imbalance(y_train)
    _plot_distribution_internal(
        stats["count_majority"], stats["count_minority"],
        stats["total"], stats["ratio"], stats["accuracy_naive"],
        out_path,
    )


def plot_pr_comparison(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    out_path: Path,
) -> None:
    """Courbes Precision/Rappel superposees pour les 5 strategies.

    La courbe PR est preferable a ROC sur des donnees desequilibrees : elle
    reflete directement le trade-off qui compte en maintenance predictive
    (bon Recall sans noyer le technicien sous les fausses alertes).

    Parameters
    ----------
    out_path : Path - chemin de sauvegarde (PNG).
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    for strategy, meta in STRATEGY_META.items():
        pipeline = build_strategy_pipeline(strategy)
        pipeline.fit(X_train, y_train)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        precision_arr, recall_arr, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)
        ax.plot(
            recall_arr, precision_arr,
            label=f"{meta['label']} (AP={pr_auc:.3f})",
            color=meta["color"],
            linewidth=2.0,
        )

    baseline_pr = float(y_test.mean())
    ax.axhline(
        baseline_pr, color="gray", linestyle="--", linewidth=1.2,
        label=f"Classifieur aleatoire (AP={baseline_pr:.3f})",
    )

    ax.set_xlabel("Rappel (Recall)", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(
        "Courbes Precision/Rappel par strategie de reequilibrage",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.grid(alpha=0.3, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] Courbes PR sauvegardees -> {out_path}")


def plot_metrics_comparison(results_df: pd.DataFrame, out_path: Path) -> None:
    """Barplot groupe comparant F1, Recall, Precision et PR-AUC par strategie.

    Parameters
    ----------
    results_df : DataFrame retourne par compare_all_strategies.
    out_path   : chemin de sauvegarde (PNG).
    """
    metrics_cols = ["f1", "recall", "precision", "pr_auc"]
    metric_labels = ["F1", "Recall", "Precision", "PR-AUC"]
    n_strategies = len(results_df)
    n_metrics = len(metrics_cols)

    x = np.arange(n_strategies)
    width = 0.18
    bar_colors = [COLOR_EFREI_BLUE, COLOR_ALERT_RED, COLOR_OK_GREEN, COLOR_WARNING]

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (col, label_m, color) in enumerate(zip(metrics_cols, metric_labels, bar_colors)):
        offset = (i - n_metrics / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, results_df[col],
            width=width * 0.92, label=label_m,
            color=color, alpha=0.88, edgecolor="white",
        )
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2, h + 0.008,
                f"{h:.2f}", ha="center", va="bottom", fontsize=7.5, color="#333",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(results_df["label"], rotation=14, ha="right", fontsize=10)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.set_title(
        "Comparaison des strategies de gestion du desequilibre\n"
        "(F1 / Recall / Precision / PR-AUC sur le test set)",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.legend(fontsize=10, ncol=4, loc="upper right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] Comparaison metriques sauvegardee -> {out_path}")


def plot_fit_time_comparison(results_df: pd.DataFrame, out_path: Path) -> None:
    """Barplot horizontal du temps d'entrainement par strategie.

    Le cout computationnel compte si le pipeline doit etre reentrainee
    frequemment (drift, nouvelles donnees, retraining hebdomadaire).

    Parameters
    ----------
    results_df : DataFrame retourne par compare_all_strategies.
    out_path   : chemin de sauvegarde (PNG).
    """
    colors = [STRATEGY_META[s]["color"] for s in results_df["strategy"]]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(
        results_df["label"], results_df["fit_time_s"],
        color=colors, edgecolor="white", height=0.55,
    )
    for i, (_, row) in enumerate(results_df.iterrows()):
        ax.text(
            row["fit_time_s"] + 0.3, i,
            f"{row['fit_time_s']:.1f}s", va="center", fontsize=10,
        )

    ax.set_xlabel("Temps d'entrainement (secondes)", fontsize=11)
    ax.set_title(
        "Cout computationnel par strategie", fontsize=13, fontweight="bold", pad=10,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.35, linestyle="--")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] Temps d'entrainement sauvegarde -> {out_path}")


# ---------------------------------------------------------------------------
# 7. Optimisation du seuil de decision (compat. ancienne API)
# ---------------------------------------------------------------------------

def optimize_threshold(
    model,
    X_test: np.ndarray | pd.DataFrame,
    y_test: np.ndarray | pd.Series,
    metric: Literal["f1", "recall"] = "f1",
    output_dir: Path | None = None,
) -> dict:
    """Cherche le seuil de decision optimal pour maximiser f1 ou recall.

    Par defaut les classifieurs utilisent 0.5 comme seuil. Sur donnees
    desequilibrees, ce seuil sous-estime les pannes. On explore [0.05, 0.95]
    par pas de 0.05 pour trouver le seuil qui maximise la metrique choisie.

    Parameters
    ----------
    model : estimateur sklearn/imblearn avec predict_proba.
    X_test : features du test set.
    y_test : labels du test set.
    metric : "f1" (defaut) ou "recall".
    output_dir : dossier de sauvegarde de la courbe (PNG) + JSON.

    Returns
    -------
    dict avec cles :
        optimal_threshold, optimal_score, metric,
        f1_at_0_5, recall_at_0_5, fn_at_0_5, fn_at_optimal.
    """
    from sklearn.metrics import confusion_matrix as _cm

    y_proba = model.predict_proba(X_test)[:, 1]
    thresholds = np.arange(0.05, 0.96, 0.05)
    scores: list[float] = []

    for thr in thresholds:
        y_pred_thr = (y_proba >= thr).astype(int)
        s = (
            f1_score(y_test, y_pred_thr, zero_division=0)
            if metric == "f1"
            else recall_score(y_test, y_pred_thr, zero_division=0)
        )
        scores.append(s)

    best_idx = int(np.argmax(scores))
    optimal_thr = float(thresholds[best_idx])
    optimal_score = float(scores[best_idx])

    y_default = (y_proba >= 0.5).astype(int)
    y_optimal = (y_proba >= optimal_thr).astype(int)

    cm_def = _cm(y_test, y_default)
    cm_opt = _cm(y_test, y_optimal)
    fn_default = int(cm_def[1, 0]) if cm_def.shape == (2, 2) else 0
    fn_optimal = int(cm_opt[1, 0]) if cm_opt.shape == (2, 2) else 0

    result = {
        "optimal_threshold": optimal_thr,
        "optimal_score": round(optimal_score, 4),
        "metric": metric,
        "f1_at_0_5": round(f1_score(y_test, y_default, zero_division=0), 4),
        "recall_at_0_5": round(recall_score(y_test, y_default, zero_division=0), 4),
        "fn_at_0_5": fn_default,
        "fn_at_optimal": fn_optimal,
    }

    if output_dir is not None:
        output_dir = Path(output_dir)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Optimisation du seuil de decision", fontsize=13, fontweight="bold")

        axes[0].plot(thresholds, scores, color=COLOR_EFREI_BLUE,
                     linewidth=2.5, marker="o", markersize=4)
        axes[0].axvline(optimal_thr, color=COLOR_ALERT_RED, linestyle="--",
                        label=f"Seuil optimal = {optimal_thr:.2f}")
        axes[0].axvline(0.5, color=COLOR_WARNING, linestyle=":",
                        label="Seuil defaut = 0.50")
        axes[0].set_xlabel("Seuil de decision")
        axes[0].set_ylabel(metric.upper())
        axes[0].set_title(f"Courbe {metric.upper()} vs seuil")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].spines[["top", "right"]].set_visible(False)

        axes[1].bar(
            ["Seuil defaut (0.50)", f"Seuil optimal ({optimal_thr:.2f})"],
            [fn_default, fn_optimal],
            color=[COLOR_WARNING, COLOR_OK_GREEN],
        )
        axes[1].set_title("Faux negatifs (pannes non detectees)")
        axes[1].set_ylabel("Nombre de faux negatifs")
        for bar, val in zip(axes[1].patches, [fn_default, fn_optimal]):
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                str(val), ha="center", fontsize=12, fontweight="bold",
            )
        axes[1].spines[["top", "right"]].set_visible(False)

        fig.tight_layout()
        fig.savefig(output_dir / "threshold_optimization.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        with open(output_dir / "optimal_threshold_16.json", "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)

    return result


# ---------------------------------------------------------------------------
# 8. Compatibilite ancienne API (apply_resampling / compare_strategies)
# ---------------------------------------------------------------------------

def apply_resampling(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    method: ResamplingMethod = "smote",
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    """Reequilibre le train set via sur- ou sous-echantillonnage.

    IMPORTANT : n'appeler que sur le TRAIN SET apres preprocessing.

    Parameters
    ----------
    method : {"random_over", "smote", "random_under", "none"}
        Technique de reequilibrage.

    Returns
    -------
    X_resampled, y_resampled (np.ndarray).
    """
    X = np.asarray(X_train)
    y = np.asarray(y_train)

    if method == "none":
        return X, y

    samplers: dict = {
        "random_over":  RandomOverSampler(random_state=random_state),
        "smote":        SMOTE(random_state=random_state, k_neighbors=5),
        "random_under": RandomUnderSampler(random_state=random_state),
    }
    if method not in samplers:
        raise ValueError(f"Methode inconnue : '{method}'. Choisir parmi {list(samplers)} ou 'none'.")

    sampler = samplers[method]
    X_res, y_res = sampler.fit_resample(X, y)

    before = dict(zip(*np.unique(y, return_counts=True)))
    after  = dict(zip(*np.unique(y_res, return_counts=True)))
    print(f"[{method}] avant={before} -> apres={after} ({len(y_res) - len(y):+,} obs)")
    return X_res, y_res


def compare_strategies(results_dict: dict) -> pd.DataFrame:
    """Genere un DataFrame recapitulatif a partir de dicts de metriques.

    Interface de compatibilite conservee pour les scripts qui construisent
    leurs propres dicts de resultats.

    Parameters
    ----------
    results_dict : {strategy_name: {"recall": float, "f1": float, ...}}

    Returns
    -------
    pd.DataFrame trie par Recall decroissant.
    """
    rows = [
        {
            "Strategie": name,
            "Recall":    round(m.get("recall", 0), 4),
            "Precision": round(m.get("precision", 0), 4),
            "F1":        round(m.get("f1", 0), 4),
            "ROC-AUC":   round(m.get("roc_auc", 0), 4),
            "PR-AUC":    round(m.get("pr_auc", 0), 4),
            "Temps (s)": round(m.get("fit_time", 0), 2),
        }
        for name, m in results_dict.items()
    ]
    return pd.DataFrame(rows).sort_values("Recall", ascending=False).reset_index(drop=True)


# Alias pour compat. eventuelle avec import pr_curves
def plot_pr_curves(pr_curves_dict: dict, output_dir: Path | None = None) -> None:
    """Trace les courbes PR a partir de dicts pre-calcules (ancienne API).

    Parameters
    ----------
    pr_curves_dict : {strategy: {"precision": arr, "recall": arr, "pr_auc": float}}
    output_dir : dossier de sauvegarde.
    """
    from sklearn.metrics import auc

    fig, ax = plt.subplots(figsize=(9, 6))
    palette = list(STRATEGY_META.values())

    for i, (name, data) in enumerate(pr_curves_dict.items()):
        meta = palette[i % len(palette)]
        pr_auc_val = data.get("pr_auc", auc(data["recall"], data["precision"]))
        ax.plot(
            data["recall"], data["precision"],
            label=f"{name} (PR-AUC={pr_auc_val:.3f})",
            color=meta["color"], linewidth=2,
        )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Courbes Precision-Recall par strategie de reequilibrage")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    if output_dir is not None:
        fig.savefig(Path(output_dir) / "pr_curves_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
