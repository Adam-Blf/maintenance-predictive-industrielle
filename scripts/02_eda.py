# -*- coding: utf-8 -*-
"""Script · Analyse Exploratoire des Données (EDA).

Produit l'ensemble des graphiques d'EDA dans `reports/figures/` ·
  - Distribution de chaque capteur (histogramme + KDE).
  - Boxplot des capteurs par classe (panne / pas panne).
  - Matrice de corrélation (heatmap).
  - Distribution des classes cibles.
  - Distribution des modes opératoires.
  - Scatterplot vibration × température coloré par panne.

Tous ces visuels servent à la fois à la qualité analytique du rapport
et à la justification des choix de modélisation (ex. justifier la
standardisation par les distributions hétérogènes des capteurs).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    COLOR_ALERT_RED,
    COLOR_EFREI_BLUE,
    COLOR_EFREI_DARK,
    COLOR_OK_GREEN,
    NUMERIC_FEATURES,
    REPORTS_FIGURES_DIR,
    TARGET_BINARY,
    ensure_directories,
)
from src.data_loader import load_dataset  # noqa: E402

# Configuration globale des graphiques · style cohérent dans tout le rapport.
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.titlecolor"] = COLOR_EFREI_DARK


def plot_target_distribution(df: pd.DataFrame) -> Path:
    """Distribution de la cible binaire (déséquilibre des classes)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    counts = df[TARGET_BINARY].value_counts().sort_index()
    bars = ax.bar(
        ["Pas de panne 24h", "Panne 24h"],
        counts.values,
        color=[COLOR_OK_GREEN, COLOR_ALERT_RED],
        edgecolor="black",
    )
    # Annotations en valeur absolue + pourcentage.
    total = counts.sum()
    for bar, value in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:,}\n({value/total:.1%})",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    ax.set_title("Distribution de la cible · panne dans les 24h")
    ax.set_ylabel("Nombre d'observations")
    plt.tight_layout()
    output = REPORTS_FIGURES_DIR / "eda_target_distribution.png"
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_failure_type_distribution(df: pd.DataFrame) -> Path:
    """Distribution des types de panne (parmi les machines en panne)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    failure_only = df[df[TARGET_BINARY] == 1]
    counts = failure_only["failure_type"].value_counts()
    bars = ax.bar(
        counts.index,
        counts.values,
        color=COLOR_EFREI_BLUE,
        edgecolor="black",
    )
    for bar, v in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v,
            str(v),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax.set_title("Répartition des types de panne (machines en panne)")
    ax.set_ylabel("Nombre")
    plt.tight_layout()
    output = REPORTS_FIGURES_DIR / "eda_failure_type_distribution.png"
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_sensor_distributions(df: pd.DataFrame) -> Path:
    """Histogrammes des 10 capteurs numériques · grille 2×5."""
    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    axes = axes.flatten()

    for ax, feature in zip(axes, NUMERIC_FEATURES):
        # KDE + hist · double information densité + comptage.
        sns.histplot(df[feature], kde=True, ax=ax, color=COLOR_EFREI_BLUE, bins=40)
        ax.set_title(feature, fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("")

    plt.suptitle(
        "Distribution des 10 capteurs numériques",
        fontsize=14,
        fontweight="bold",
        color=COLOR_EFREI_DARK,
    )
    plt.tight_layout()
    output = REPORTS_FIGURES_DIR / "eda_sensor_distributions.png"
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_sensor_boxplots_by_class(df: pd.DataFrame) -> Path:
    """Boxplots capteurs × classe (panne / pas panne) · révèle les
    capteurs discriminants."""
    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    axes = axes.flatten()

    for ax, feature in zip(axes, NUMERIC_FEATURES):
        sns.boxplot(
            data=df,
            x=TARGET_BINARY,
            y=feature,
            hue=TARGET_BINARY,
            palette={0: COLOR_OK_GREEN, 1: COLOR_ALERT_RED},
            ax=ax,
            showfliers=False,
            legend=False,
        )
        ax.set_title(feature, fontsize=10)
        ax.set_xlabel("")
        ax.set_xticklabels(["OK", "Panne"], fontsize=9)
        ax.set_ylabel("")

    plt.suptitle(
        "Boxplots des capteurs · OK vs Panne 24h",
        fontsize=14,
        fontweight="bold",
        color=COLOR_EFREI_DARK,
    )
    plt.tight_layout()
    output = REPORTS_FIGURES_DIR / "eda_boxplots_by_class.png"
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_correlation_heatmap(df: pd.DataFrame) -> Path:
    """Matrice de corrélation des features numériques + cible."""
    cols = NUMERIC_FEATURES + [TARGET_BINARY]
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(11, 9))
    # `mask` triangle supérieur pour éviter la redondance visuelle.
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax,
        annot_kws={"size": 9},
    )
    ax.set_title(
        "Matrice de corrélation · capteurs + cible",
        fontsize=13,
        color=COLOR_EFREI_DARK,
    )
    plt.tight_layout()
    output = REPORTS_FIGURES_DIR / "eda_correlation_heatmap.png"
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_scatter_vib_temp(df: pd.DataFrame) -> Path:
    """Scatter vibration × température · les 2 features les plus
    discriminantes selon l'EDA. Coloré par classe."""
    fig, ax = plt.subplots(figsize=(10, 7))
    # Échantillonnage pour rester lisible (24k points = nuage opaque).
    sample = df.sample(n=min(5000, len(df)), random_state=42)

    for cls, color, label in [
        (0, COLOR_OK_GREEN, "Pas de panne 24h"),
        (1, COLOR_ALERT_RED, "Panne 24h"),
    ]:
        subset = sample[sample[TARGET_BINARY] == cls]
        ax.scatter(
            subset["vibration_rms"],
            subset["temperature_motor"],
            alpha=0.4,
            s=18,
            color=color,
            label=label,
            edgecolors="none",
        )

    ax.set_xlabel("Vibration RMS (mm/s)", fontsize=11)
    ax.set_ylabel("Température moteur (°C)", fontsize=11)
    ax.set_title(
        "Vibration × Température · pannes en zone haute droite",
        fontsize=13,
        color=COLOR_EFREI_DARK,
    )
    ax.legend(framealpha=0.95)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    output = REPORTS_FIGURES_DIR / "eda_scatter_vib_temp.png"
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_operating_mode(df: pd.DataFrame) -> Path:
    """Distribution des modes opératoires + taux de panne par mode."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Subplot 1 · comptage par mode.
    counts = df["operating_mode"].value_counts()
    axes[0].bar(counts.index, counts.values, color=COLOR_EFREI_BLUE, edgecolor="black")
    axes[0].set_title("Nombre d'observations par mode opératoire")
    axes[0].set_ylabel("Comptage")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=10)

    # Subplot 2 · taux de panne par mode (insight clé pour le rapport).
    failure_rate = df.groupby("operating_mode")[TARGET_BINARY].mean().sort_values()
    axes[1].bar(
        failure_rate.index,
        failure_rate.values * 100,
        color=COLOR_ALERT_RED,
        edgecolor="black",
    )
    axes[1].set_title("Taux de panne 24h (%) par mode opératoire")
    axes[1].set_ylabel("Taux de panne (%)")
    for i, v in enumerate(failure_rate.values):
        axes[1].text(
            i,
            v * 100,
            f"{v*100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.suptitle(
        "Analyse des modes opératoires",
        fontsize=14,
        fontweight="bold",
        color=COLOR_EFREI_DARK,
    )
    plt.tight_layout()
    output = REPORTS_FIGURES_DIR / "eda_operating_mode.png"
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> None:
    """Point d'entrée · génère tous les graphiques d'EDA."""
    ensure_directories()
    df = load_dataset()
    print(f"[EDA] Dataset chargé · {df.shape}")

    print("[EDA] Génération des 7 graphiques d'analyse...")
    outputs = {
        "target_dist": plot_target_distribution(df),
        "failure_type": plot_failure_type_distribution(df),
        "sensor_dist": plot_sensor_distributions(df),
        "boxplots": plot_sensor_boxplots_by_class(df),
        "correlation": plot_correlation_heatmap(df),
        "scatter_vt": plot_scatter_vib_temp(df),
        "operating_mode": plot_operating_mode(df),
    }
    for name, path in outputs.items():
        print(f"  - {name:<14} · {path}")

    # Sauvegarde des stats descriptives en CSV pour intégration au rapport.
    stats_path = REPORTS_FIGURES_DIR.parent / "eda_descriptive_stats.csv"
    df[NUMERIC_FEATURES].describe().to_csv(stats_path)
    print(f"[EDA] Stats descriptives · {stats_path}")
    print("Done.")


if __name__ == "__main__":
    main()
