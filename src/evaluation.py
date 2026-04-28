# -*- coding: utf-8 -*-
"""Évaluation comparative des modèles.

Métriques sélectionnées pour la classification binaire (failure_within_24h) ·
  - **Accuracy** · classique mais trompeuse en cas de déséquilibre (85%
    de non-pannes : un modèle qui prédit toujours 0 aurait 85% d'accuracy).
    Conservée pour comparaison avec la littérature.
  - **Precision / Recall / F1** · le couple Precision/Recall est essentiel
    en maintenance prédictive · un faux négatif (panne ratée) coûte des
    arrêts de production (~1000 EUR), un faux positif coûte une intervention
    inutile (~100 EUR). Le F1 est leur moyenne harmonique.
  - **ROC-AUC** · invariant par seuil, mesure la capacité de discrimination
    globale. Vaut 0.5 pour un classifieur aléatoire, 1.0 pour un modèle
    parfait. Robuste au déséquilibre mais insensible aux erreurs rares.
  - **PR-AUC (Average Precision)** · aire sous la courbe Precision-Recall,
    plus fiable que ROC-AUC quand les classes sont déséquilibrées car elle
    pénalise davantage les faux positifs sur la classe minoritaire (pannes).
    Métrique recommandée pour les datasets déséquilibrés.
  - **Matrice de confusion** · pour analyse détaillée des types d'erreurs
    (FN = pannes non détectées, FP = fausses alarmes).
  - **Temps d'entraînement / latence** · métriques d'écoresponsabilité
    requises par le critère C4.3 RNCP40875.

Toutes les métriques sont calculées sur le **test set isolé**, jamais sur
les données vues à l'entraînement (anti data-leakage strict).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .config import (
    COLOR_ALERT_RED,
    COLOR_EFREI_BLUE,
    COLOR_EFREI_DARK,
    COLOR_OK_GREEN,
    REPORTS_FIGURES_DIR,
)


@dataclass
class ClassificationMetrics:
    """Conteneur typé des métriques de classification binaire.

    L'usage d'une dataclass plutôt qu'un dict simple offre :
      - typage statique (mypy / Pylance peuvent valider),
      - sérialisation `asdict()` directe vers JSON / CSV,
      - lisibilité du code consommateur (`m.f1` vs `m['f1']`).
    """

    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    fit_time_s: float
    predict_time_ms: float

    def to_dict(self) -> dict:
        """Conversion dict pour pandas/JSON."""
        return asdict(self)


def compute_classification_metrics(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    fit_time_s: float = 0.0,
    predict_time_ms: float = 0.0,
) -> ClassificationMetrics:
    """Calcule l'ensemble des métriques de classification binaire.

    Parameters
    ----------
    model_name : str
        Identifiant lisible du modèle (utilisé dans les graphes).
    y_true : np.ndarray
        Étiquettes réelles (0/1).
    y_pred : np.ndarray
        Prédictions binaires du modèle.
    y_proba : np.ndarray
        Probabilités de la classe positive (utilisées pour ROC/PR-AUC).
    fit_time_s : float
        Temps d'entraînement en secondes (mesure d'écoresponsabilité).
    predict_time_ms : float
        Temps moyen de prédiction par échantillon (latence).

    Returns
    -------
    ClassificationMetrics
        Dataclass contenant les 6 métriques + temps.
    """
    return ClassificationMetrics(
        model_name=model_name,
        accuracy=float(accuracy_score(y_true, y_pred)),
        # `zero_division=0` évite l'avertissement quand un modèle ne
        # prédit jamais la classe positive (Precision indéfinie).
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        roc_auc=float(roc_auc_score(y_true, y_proba)),
        pr_auc=float(average_precision_score(y_true, y_proba)),
        fit_time_s=float(fit_time_s),
        predict_time_ms=float(predict_time_ms),
    )


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    output_dir: Path = REPORTS_FIGURES_DIR,
) -> Path:
    """Génère la matrice de confusion en PNG haute résolution.

    On normalise par ligne (`normalize="true"`) pour afficher des
    pourcentages de Recall par classe · plus parlant qu'un comptage brut
    quand les classes sont déséquilibrées.
    """
    cm = confusion_matrix(y_true, y_pred, normalize="true")

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2%",
        cmap="Blues",
        xticklabels=["Pas de panne", "Panne 24h"],
        yticklabels=["Pas de panne", "Panne 24h"],
        cbar=False,
        ax=ax,
        annot_kws={"size": 13, "weight": "bold"},
    )
    ax.set_xlabel("Prédiction", fontsize=11, fontweight="bold")
    ax.set_ylabel("Vérité terrain", fontsize=11, fontweight="bold")
    ax.set_title(
        f"Matrice de confusion · {model_name}",
        fontsize=12,
        fontweight="bold",
        color=COLOR_EFREI_DARK,
    )
    plt.tight_layout()

    output_path = output_dir / f"confusion_matrix_{model_name}.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_roc_curves(
    results: dict[str, dict],
    output_dir: Path = REPORTS_FIGURES_DIR,
) -> Path:
    """Trace les courbes ROC superposées pour comparaison visuelle.

    Parameters
    ----------
    results : dict[str, dict]
        Dictionnaire `{model_name: {"y_true": ..., "y_proba": ...}}`.
    output_dir : Path
        Répertoire de sortie des PNG.

    Returns
    -------
    Path
        Chemin du fichier généré.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    # Palette manuelle pour différencier visuellement chaque modèle.
    palette = [COLOR_EFREI_BLUE, COLOR_EFREI_DARK, COLOR_ALERT_RED, COLOR_OK_GREEN]

    for (name, payload), color in zip(results.items(), palette):
        fpr, tpr, _ = roc_curve(payload["y_true"], payload["y_proba"])
        ax.plot(
            fpr,
            tpr,
            label=f"{name} (AUC = {auc(fpr, tpr):.3f})",
            color=color,
            linewidth=2,
        )

    # Diagonale de référence · classifieur aléatoire (AUC = 0.5).
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Aléatoire (AUC = 0.5)")

    ax.set_xlabel("Taux de faux positifs (FPR)", fontsize=11)
    ax.set_ylabel("Taux de vrais positifs (TPR / Recall)", fontsize=11)
    ax.set_title(
        "Courbes ROC · Comparaison des 4 modèles",
        fontsize=13,
        fontweight="bold",
        color=COLOR_EFREI_DARK,
    )
    ax.legend(loc="lower right", framealpha=0.95)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = output_dir / "roc_curves_comparison.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_pr_curves(
    results: dict[str, dict],
    output_dir: Path = REPORTS_FIGURES_DIR,
) -> Path:
    """Trace les courbes Precision-Recall (plus pertinentes que ROC en
    contexte déséquilibré).
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    palette = [COLOR_EFREI_BLUE, COLOR_EFREI_DARK, COLOR_ALERT_RED, COLOR_OK_GREEN]

    for (name, payload), color in zip(results.items(), palette):
        precision, recall, _ = precision_recall_curve(payload["y_true"], payload["y_proba"])
        ap = average_precision_score(payload["y_true"], payload["y_proba"])
        ax.plot(
            recall,
            precision,
            label=f"{name} (AP = {ap:.3f})",
            color=color,
            linewidth=2,
        )

    ax.set_xlabel("Recall (Sensibilité)", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title(
        "Courbes Precision-Recall · Comparaison",
        fontsize=13,
        fontweight="bold",
        color=COLOR_EFREI_DARK,
    )
    ax.legend(loc="lower left", framealpha=0.95)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = output_dir / "pr_curves_comparison.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_metrics_barplot(
    metrics_df: pd.DataFrame,
    output_dir: Path = REPORTS_FIGURES_DIR,
) -> Path:
    """Histogramme groupé · une barre par modèle, cluster par métrique.

    Permet de visualiser d'un coup d'oeil les compromis · un modèle peut
    avoir une excellente accuracy mais un recall médiocre, et vice-versa.
    """
    metrics_to_plot = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    plot_df = metrics_df.set_index("model_name")[metrics_to_plot].T

    fig, ax = plt.subplots(figsize=(11, 6))
    plot_df.plot(
        kind="bar",
        ax=ax,
        edgecolor="black",
        width=0.78,
        color=[COLOR_EFREI_BLUE, COLOR_EFREI_DARK, COLOR_ALERT_RED, COLOR_OK_GREEN],
    )

    ax.set_xlabel("Métrique", fontsize=11)
    ax.set_ylabel("Score (0-1)", fontsize=11)
    ax.set_title(
        "Comparaison des 4 modèles · 6 métriques",
        fontsize=13,
        fontweight="bold",
        color=COLOR_EFREI_DARK,
    )
    ax.set_ylim(0, 1.05)
    ax.legend(title="Modèle", loc="lower right", framealpha=0.95)
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=0)

    # Annotations numériques au-dessus de chaque barre · facilite la
    # lecture du tableau comparatif dans le rapport.
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=8, padding=2)

    plt.tight_layout()
    output_path = output_dir / "metrics_comparison_barplot.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_training_time_barplot(
    metrics_df: pd.DataFrame,
    output_dir: Path = REPORTS_FIGURES_DIR,
) -> Path:
    """Compare temps d'entraînement et latence d'inférence (écoresponsabilité).

    Le sujet RNCP demande explicitement d'évaluer le degré
    d'écoresponsabilité · plus le temps de calcul est long, plus l'empreinte
    énergétique est importante.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Subplot 1 · temps d'entraînement.
    axes[0].bar(
        metrics_df["model_name"],
        metrics_df["fit_time_s"],
        color=COLOR_EFREI_BLUE,
        edgecolor="black",
    )
    axes[0].set_title("Temps d'entraînement (s)", fontweight="bold")
    axes[0].set_ylabel("Secondes")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(True, axis="y", alpha=0.3)
    for i, v in enumerate(metrics_df["fit_time_s"]):
        axes[0].text(i, v, f"{v:.2f}s", ha="center", va="bottom", fontsize=9)

    # Subplot 2 · latence prédiction.
    axes[1].bar(
        metrics_df["model_name"],
        metrics_df["predict_time_ms"],
        color=COLOR_ALERT_RED,
        edgecolor="black",
    )
    axes[1].set_title("Latence prédiction (ms / échantillon)", fontweight="bold")
    axes[1].set_ylabel("Millisecondes")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(True, axis="y", alpha=0.3)
    for i, v in enumerate(metrics_df["predict_time_ms"]):
        axes[1].text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    plt.suptitle(
        "Coût computationnel · Écoresponsabilité",
        fontsize=13,
        fontweight="bold",
        color=COLOR_EFREI_DARK,
    )
    plt.tight_layout()
    output_path = output_dir / "compute_cost_comparison.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
