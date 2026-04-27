# -*- coding: utf-8 -*-
"""Génération des schémas pédagogiques inclus dans le rapport PDF.

Ces diagrammes ne sont PAS des graphiques de données · ce sont des
illustrations conceptuelles construites avec matplotlib (rectangles,
flèches, annotations) pour expliquer ·
  - l'architecture du système intelligent (data → modèle → API → dashboard),
  - le pipeline ML (EDA → preprocessing → split → train → evaluate),
  - le compromis biais-variance,
  - le workflow d'interprétabilité.

Le choix de matplotlib pur (vs PlantUML/Graphviz) est volontaire · pas
de dépendance externe binaire (Java, dot), tout fonctionne sur la
machine d'Adam sans setup additionnel et reste reproductible en CI.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from .config import (
    COLOR_ALERT_RED,
    COLOR_EFREI_BLUE,
    COLOR_EFREI_DARK,
    COLOR_OK_GREEN,
    COLOR_WARNING,
    REPORTS_FIGURES_DIR,
)


def _draw_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    color: str,
    text_color: str = "white",
    font_size: int = 10,
    boxstyle: str = "round,pad=0.05",
) -> None:
    """Helper · dessine une boîte arrondie avec du texte centré.

    L'usage de `FancyBboxPatch` plutôt qu'un `Rectangle` permet d'avoir
    des coins arrondis qui rendent mieux dans le rapport.
    """
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=boxstyle,
        linewidth=1.5,
        edgecolor="black",
        facecolor=color,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        color=text_color,
        fontsize=font_size,
        fontweight="bold",
        wrap=True,
    )


def _draw_arrow(
    ax,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    label: str = "",
    color: str = "black",
) -> None:
    """Helper · flèche annotée entre deux points (pour relier des boîtes)."""
    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="->",
        mutation_scale=18,
        linewidth=1.6,
        color=color,
    )
    ax.add_patch(arrow)
    if label:
        ax.text(
            (x1 + x2) / 2,
            (y1 + y2) / 2 + 0.08,
            label,
            ha="center",
            fontsize=8,
            color=color,
            style="italic",
        )


def render_architecture_diagram(
    output_dir: Path = REPORTS_FIGURES_DIR,
) -> Path:
    """Schéma · architecture globale du système intelligent.

    Inspiré de l'architecture médaillon mais simplifié pour un MVP ·
    Source de données → ETL → Modèle ML → API → Dashboard.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Titre du schéma.
    ax.text(
        7,
        5.6,
        "Architecture du système intelligent · Maintenance Prédictive",
        ha="center",
        fontsize=14,
        fontweight="bold",
        color=COLOR_EFREI_DARK,
    )

    # 1. Capteurs industriels (couche acquisition).
    _draw_box(ax, 0.2, 2.5, 1.8, 1.2, "Capteurs\nindustriels\nIoT", COLOR_WARNING)

    # 2. CSV / Data Lake (raw layer).
    _draw_box(ax, 2.5, 2.5, 1.8, 1.2, "CSV / Data Lake\n(bronze)", COLOR_EFREI_BLUE)

    # 3. Preprocessing pipeline.
    _draw_box(
        ax,
        4.8,
        2.5,
        1.8,
        1.2,
        "Pipeline\nPreprocessing\n(silver)",
        COLOR_EFREI_BLUE,
    )

    # 4. Modèles ML/DL.
    _draw_box(ax, 7.1, 2.5, 1.8, 1.2, "4 modèles\nML + DL\n(gold)", COLOR_EFREI_DARK)

    # 5. API REST FastAPI.
    _draw_box(ax, 9.4, 3.7, 1.8, 1.0, "API REST\nFastAPI", COLOR_OK_GREEN)

    # 6. Dashboard Streamlit.
    _draw_box(ax, 9.4, 1.5, 1.8, 1.0, "Dashboard\nStreamlit", COLOR_OK_GREEN)

    # 7. Utilisateur final.
    _draw_box(
        ax,
        11.7,
        2.5,
        1.8,
        1.2,
        "Responsable\nMaintenance",
        COLOR_ALERT_RED,
    )

    # Flèches de liaison (gauche à droite).
    _draw_arrow(ax, 2.0, 3.1, 2.5, 3.1)
    _draw_arrow(ax, 4.3, 3.1, 4.8, 3.1)
    _draw_arrow(ax, 6.6, 3.1, 7.1, 3.1)
    _draw_arrow(ax, 8.9, 3.4, 9.4, 4.0)
    _draw_arrow(ax, 8.9, 2.8, 9.4, 2.0)
    _draw_arrow(ax, 11.2, 4.0, 11.7, 3.4)
    _draw_arrow(ax, 11.2, 2.0, 11.7, 2.8)

    # Légende des couches médaillon.
    legend_y = 0.6
    ax.text(0.2, legend_y, "Légende ·", fontsize=10, fontweight="bold")
    bronze_patch = mpatches.Patch(color=COLOR_WARNING, label="Bronze · données brutes")
    silver_patch = mpatches.Patch(color=COLOR_EFREI_BLUE, label="Silver · données nettoyées")
    gold_patch = mpatches.Patch(color=COLOR_EFREI_DARK, label="Gold · features modèle")
    ax.legend(
        handles=[bronze_patch, silver_patch, gold_patch],
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.05),
        fontsize=9,
        framealpha=0.95,
    )

    plt.tight_layout()
    output_path = output_dir / "diagram_architecture.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def render_ml_pipeline_diagram(
    output_dir: Path = REPORTS_FIGURES_DIR,
) -> Path:
    """Schéma · pipeline ML séquentiel (de l'EDA au modèle final).

    Représente la méthodologie scientifique imposée par le sujet ·
    EDA → preprocessing → split stratifié → train multi-modèles →
    cross-validation → évaluation → interprétabilité → sélection finale.
    """
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4.5)
    ax.axis("off")

    ax.text(
        7,
        4.0,
        "Pipeline Data Science · de la donnée brute au modèle déployable",
        ha="center",
        fontsize=14,
        fontweight="bold",
        color=COLOR_EFREI_DARK,
    )

    # 7 étapes en file horizontale.
    steps = [
        ("EDA\n+ profiling", COLOR_EFREI_BLUE),
        ("Nettoyage\n+ Imputation", COLOR_EFREI_BLUE),
        ("Encodage\n+ Scaling", COLOR_EFREI_BLUE),
        ("Split\nstratifié 80/20", COLOR_WARNING),
        ("Entraînement\n4 modèles", COLOR_EFREI_DARK),
        ("Cross-validation\n5-fold", COLOR_EFREI_DARK),
        ("Évaluation\n+ SHAP", COLOR_OK_GREEN),
        ("Modèle\nfinal", COLOR_ALERT_RED),
    ]

    box_w = 1.5
    box_h = 1.4
    gap = 0.15
    total_w = len(steps) * box_w + (len(steps) - 1) * gap
    start_x = (14 - total_w) / 2

    for idx, (label, color) in enumerate(steps):
        x = start_x + idx * (box_w + gap)
        _draw_box(ax, x, 1.5, box_w, box_h, label, color, font_size=8)
        # Flèche vers l'étape suivante.
        if idx < len(steps) - 1:
            _draw_arrow(
                ax,
                x + box_w,
                1.5 + box_h / 2,
                x + box_w + gap,
                1.5 + box_h / 2,
            )

    plt.tight_layout()
    output_path = output_dir / "diagram_ml_pipeline.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def render_bias_variance_diagram(
    output_dir: Path = REPORTS_FIGURES_DIR,
) -> Path:
    """Schéma · compromis biais/variance (placement des 4 modèles).

    Diagramme conceptuel · axe X = complexité du modèle, axe Y = erreur.
    Permet d'illustrer pourquoi on compare 4 modèles au lieu de prendre
    directement le plus complexe.
    """
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 6))

    # Génération de courbes idéalisées (forme typique du compromis).
    complexity = np.linspace(0, 10, 200)
    bias = 5.0 / (1.0 + complexity * 0.5)  # Décroît avec la complexité.
    variance = 0.05 * complexity**1.8  # Croît avec la complexité.
    total = bias + variance

    ax.plot(complexity, bias, label="Biais²", color=COLOR_EFREI_BLUE, linewidth=2)
    ax.plot(complexity, variance, label="Variance", color=COLOR_ALERT_RED, linewidth=2)
    ax.plot(
        complexity,
        total,
        label="Erreur totale",
        color=COLOR_EFREI_DARK,
        linewidth=2.5,
        linestyle="--",
    )

    # Position des 4 modèles sur l'axe de complexité (estimation pédago).
    model_positions = {
        "Logistic\nRegression": 1.0,
        "Random\nForest": 4.5,
        "XGBoost": 6.5,
        "MLP": 7.8,
    }
    for model, pos in model_positions.items():
        # Calcul de l'erreur totale au point de complexité du modèle.
        idx = (np.abs(complexity - pos)).argmin()
        ax.axvline(pos, color="gray", linestyle=":", alpha=0.4)
        ax.annotate(
            model,
            xy=(pos, total[idx]),
            xytext=(pos, total[idx] + 1.5),
            ha="center",
            fontsize=9,
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="gray", lw=1),
        )

    ax.set_xlabel("Complexité du modèle →", fontsize=11)
    ax.set_ylabel("Erreur de généralisation", fontsize=11)
    ax.set_title(
        "Compromis Biais-Variance · positionnement des 4 modèles",
        fontsize=13,
        fontweight="bold",
        color=COLOR_EFREI_DARK,
    )
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 10)

    plt.tight_layout()
    output_path = output_dir / "diagram_bias_variance.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def render_decision_workflow_diagram(
    output_dir: Path = REPORTS_FIGURES_DIR,
) -> Path:
    """Schéma · workflow décisionnel côté responsable maintenance.

    Représente comment le responsable maintenance utilise le système ·
    consulter le dashboard → identifier machine à risque → consulter
    SHAP → décider intervention → planifier ressources.
    """
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5)
    ax.axis("off")

    ax.text(
        6.5,
        4.5,
        "Workflow décisionnel · du signal capteur à l'action terrain",
        ha="center",
        fontsize=13,
        fontweight="bold",
        color=COLOR_EFREI_DARK,
    )

    steps = [
        ("Capteurs\n(temps réel)", COLOR_WARNING),
        ("Score panne\n24h", COLOR_EFREI_BLUE),
        ("Seuil ?", COLOR_EFREI_DARK),
        ("Alerte\ndashboard", COLOR_ALERT_RED),
        ("SHAP\n(pourquoi ?)", COLOR_EFREI_BLUE),
        ("Décision\nintervention", COLOR_OK_GREEN),
        ("Ordre de\ntravail GMAO", COLOR_EFREI_DARK),
    ]

    box_w = 1.55
    box_h = 1.3
    gap = 0.15
    total_w = len(steps) * box_w + (len(steps) - 1) * gap
    start_x = (13 - total_w) / 2

    for idx, (label, color) in enumerate(steps):
        x = start_x + idx * (box_w + gap)
        # Le 3e bloc est un losange de décision (forme différente).
        boxstyle = "round4,pad=0.06" if idx == 2 else "round,pad=0.05"
        _draw_box(ax, x, 1.6, box_w, box_h, label, color, font_size=8, boxstyle=boxstyle)
        if idx < len(steps) - 1:
            _draw_arrow(
                ax,
                x + box_w,
                1.6 + box_h / 2,
                x + box_w + gap,
                1.6 + box_h / 2,
            )

    plt.tight_layout()
    output_path = output_dir / "diagram_decision_workflow.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def render_all_diagrams(output_dir: Path = REPORTS_FIGURES_DIR) -> dict[str, Path]:
    """Génère tous les schémas en une seule passe.

    Returns
    -------
    dict[str, Path]
        Mapping nom_schéma → chemin PNG produit. Utilisé par le générateur
        de rapport pour insérer les images aux bons endroits.
    """
    return {
        "architecture": render_architecture_diagram(output_dir),
        "ml_pipeline": render_ml_pipeline_diagram(output_dir),
        "bias_variance": render_bias_variance_diagram(output_dir),
        "decision_workflow": render_decision_workflow_diagram(output_dir),
    }
