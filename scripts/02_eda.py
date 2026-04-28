# -*- coding: utf-8 -*-
"""Script 02 · Analyse Exploratoire des Données (EDA).

À QUOI ÇA SERT ?
----------------
EDA = "Exploratory Data Analysis". Avant de construire un modèle de Machine
Learning, on REGARDE les données pour comprendre ·
  - Quelle est la forme du dataset (combien de lignes, de colonnes) ?
  - Y a-t-il des valeurs manquantes (NaN) ? Combien ?
  - Comment sont distribuées les variables (gaussienne ? asymétrique ?) ?
  - Y a-t-il des corrélations entre variables capteurs ?
  - La cible (panne oui/non) est-elle équilibrée ou déséquilibrée ?

C'est l'étape n°1 obligatoire en data science · si on saute l'EDA, on risque
d'entraîner un modèle sur des données qu'on ne comprend pas (et donc on ne
saura pas si les résultats sont crédibles).

Important · l'EDA NE MODIFIE PAS les données. Pas de cleaning, pas de split,
pas de scaling · juste de l'observation. Les transformations viennent dans
le script 03 (preprocessing + modélisation).

CE QUE CE SCRIPT PRODUIT (dans reports/02/)
-------------------------------------------
  - eda_target_distribution.png · combien de pannes vs sain ?
  - eda_failure_type_distribution.png · types de panne (5 classes)
  - eda_sensor_distributions.png · histogrammes des 7 capteurs numériques
  - eda_boxplots_by_class.png · capteurs séparés par classe (panne/sain)
  - eda_correlation_heatmap.png · corrélations entre capteurs
  - eda_scatter_vib_temp.png · vibration vs température, coloré par panne
  - eda_operating_mode.png · taux de panne par mode opératoire
  - eda_missing_values.png · % de NaN par colonne
  - eda_descriptive_stats.csv · stats (min/max/mean/std) en CSV
  - eda_missing_values.csv · synthèse des NaN en CSV

LES INSIGHTS DE CETTE EDA SERVENT À ·
  - Justifier la stratégie de gestion des NaN (imputation médiane).
  - Justifier le choix des modèles (classes déséquilibrées → privilégier
    F1, PR-AUC plutôt qu'accuracy).
  - Identifier les capteurs les plus discriminants (vibration et
    température sortent du lot).

USAGE
-----
    python scripts/02_eda.py
"""

# Cette ligne `from __future__` permet d'utiliser les annotations de type
# Python 3.10+ (ex. `list[str]` au lieu de `List[str]`) même sur Python 3.8/3.9.
from __future__ import annotations

# `sys` · pour ajouter le dossier projet au PYTHONPATH (sinon `from src...` casse).
import sys
# `Path` · alternative moderne aux strings pour manipuler les chemins de fichiers.
# Avantages · indépendant de l'OS (Windows/Linux/Mac), méthodes utiles (.exists(), .parent).
from pathlib import Path

# Matplotlib · la bibliothèque de référence pour faire des graphes en Python.
# `pyplot` (alias `plt`) est l'API simple "comme MATLAB" qu'on utilise ici.
import matplotlib.pyplot as plt
# Numpy · calcul numérique vectorisé (tableaux N-dimensions ultra-rapides).
# Toutes les libs ML (sklearn, pandas, ...) reposent dessus.
import numpy as np
# Pandas · manipulation de tableaux de données type Excel/CSV (DataFrames).
# `pd.read_csv()`, `df.groupby()`, `df.merge()` · les outils du data scientist.
import pandas as pd
# Seaborn · surcouche matplotlib avec des graphiques statistiques plus jolis
# (boxplots, heatmaps, kdeplots) en une ligne au lieu de 10.
import seaborn as sns

# `__file__` · chemin absolu du présent script.
# `.resolve().parent.parent` remonte de 2 niveaux · scripts/ → projet racine.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# `sys.path.insert(0, ...)` ajoute la racine au début du PYTHONPATH ·
# permet `from src.config import ...` même quand on lance depuis un autre dossier.
sys.path.insert(0, str(PROJECT_ROOT))

# Bootstrap · auto-install des dépendances manquantes (rend le repo
# clonable et exécutable sur n'importe quelle machine sans setup manuel).
from src.bootstrap import ensure_dependencies  # noqa: E402
ensure_dependencies()

# On importe nos constantes depuis src/config.py (single source of truth) ·
#   - COLOR_* · palette de couleurs cohérente dans tout le projet
#   - NUMERIC_FEATURES · liste des 7 capteurs numériques (vibration, temp, ...)
#   - S02_DIR · raccourci vers reports/02/ où ce script écrit ses sorties
#   - TARGET_BINARY · nom de la colonne cible ("failure_within_24h")
#   - ensure_directories · crée les dossiers de sortie si absents
# Le `# noqa: E402` désactive le warning "import après code" du linter
# (attendu ici puisqu'on doit d'abord modifier sys.path).
from src.config import (  # noqa: E402
    COLOR_ALERT_RED,
    COLOR_EFREI_BLUE,
    COLOR_EFREI_DARK,
    COLOR_OK_GREEN,
    NUMERIC_FEATURES,
    S02_DIR,
    TARGET_BINARY,
    ensure_directories,
)
# `load_dataset()` · charge le CSV Kaggle depuis data/raw/ et retourne un DataFrame.
from src.data_loader import load_dataset  # noqa: E402

# Configuration globale matplotlib/seaborn · une fois ici, valable pour tous les graphes.
# `style="whitegrid"` · fond blanc + grille discrète (lisible en impression).
# `palette="deep"` · palette par défaut seaborn, cohérente entre graphes.
sns.set_theme(style="whitegrid", palette="deep")
# Tous les titres de graphes seront automatiquement en gras + couleur EFREI.
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
    output = S02_DIR / "eda_target_distribution.png"
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
    output = S02_DIR / "eda_failure_type_distribution.png"
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
    output = S02_DIR / "eda_sensor_distributions.png"
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
    output = S02_DIR / "eda_boxplots_by_class.png"
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
    output = S02_DIR / "eda_correlation_heatmap.png"
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
    output = S02_DIR / "eda_scatter_vib_temp.png"
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
    output = S02_DIR / "eda_operating_mode.png"
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_missing_values(df: pd.DataFrame) -> Path:
    """Analyse des valeurs manquantes · % NaN par colonne.

    Le dataset Kaggle officiel injecte volontairement ~3-4% de NaN sur
    5 capteurs (vibration_rms, temperature_motor, current_phase_avg,
    pressure_level, rpm) pour mimer la réalité des capteurs IoT (drops,
    pannes de transmission). Cette figure quantifie le phénomène et
    justifie le choix d'imputation médiane dans le préprocesseur
    (cf. `src/preprocessing.py`).

    Conclusion EDA · imputation médiane sur les capteurs numériques
    (robuste aux outliers), one-hot encoding sur les catégoriels avec
    `handle_unknown="ignore"` pour gérer les modalités inconnues en
    inférence.
    """
    missing_count = df.isna().sum()
    missing_pct = (missing_count / len(df) * 100).round(2)
    missing_df = (
        pd.DataFrame({"count": missing_count, "pct": missing_pct})
        .query("count > 0")
        .sort_values("pct", ascending=True)
    )

    if missing_df.empty:
        # Cas dégénéré · pas de NaN, on génère quand même un visuel
        # informatif (utile pour le rapport).
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(
            0.5,
            0.5,
            "Aucune valeur manquante détectée dans le dataset",
            ha="center",
            va="center",
            fontsize=13,
            color=COLOR_OK_GREEN,
            fontweight="bold",
        )
        ax.axis("off")
    else:
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.barh(
            missing_df.index,
            missing_df["pct"],
            color=COLOR_ALERT_RED,
            edgecolor="black",
            alpha=0.85,
        )
        for bar, pct, count in zip(
            bars, missing_df["pct"].values, missing_df["count"].values
        ):
            ax.text(
                bar.get_width() + 0.05,
                bar.get_y() + bar.get_height() / 2,
                f"{pct:.2f}% ({count:,})",
                va="center",
                fontsize=10,
                fontweight="bold",
            )
        ax.set_xlabel("Pourcentage de valeurs manquantes (%)", fontsize=11)
        ax.set_title(
            "Valeurs manquantes par colonne · stratégie · imputation médiane",
            fontsize=13,
            color=COLOR_EFREI_DARK,
            fontweight="bold",
        )
        ax.grid(axis="x", alpha=0.3)
        ax.set_xlim(0, max(missing_df["pct"].max() * 1.25, 1))

    plt.tight_layout()
    output = S02_DIR / "eda_missing_values.png"
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> None:
    """Point d'entrée · génère tous les graphiques d'EDA + stats."""
    ensure_directories()
    df = load_dataset()
    print(f"[EDA] Dataset chargé · {df.shape}")

    # Synthèse rapide en console des valeurs manquantes (utile dans la
    # conclusion de l'EDA · cf. note méthodo du dossier de référence).
    missing_total = int(df.isna().sum().sum())
    missing_cols = int((df.isna().sum() > 0).sum())
    print(
        f"[EDA] Valeurs manquantes · {missing_total:,} cellules sur "
        f"{missing_cols} colonnes (stratégie · imputation médiane numérique, "
        "OHE catégoriel)"
    )

    print("[EDA] Génération des 8 graphiques d'analyse...")
    outputs = {
        "target_dist": plot_target_distribution(df),
        "failure_type": plot_failure_type_distribution(df),
        "sensor_dist": plot_sensor_distributions(df),
        "boxplots": plot_sensor_boxplots_by_class(df),
        "correlation": plot_correlation_heatmap(df),
        "scatter_vt": plot_scatter_vib_temp(df),
        "operating_mode": plot_operating_mode(df),
        "missing_values": plot_missing_values(df),
    }
    for name, path in outputs.items():
        print(f"  - {name:<14} · {path}")

    # Sauvegarde des stats descriptives en CSV pour intégration au rapport.
    stats_path = S02_DIR / "eda_descriptive_stats.csv"
    df[NUMERIC_FEATURES].describe().to_csv(stats_path)
    print(f"[EDA] Stats descriptives · {stats_path}")

    # Sauvegarde du % NaN par colonne (consommé par le rapport PDF).
    nan_path = S02_DIR / "eda_missing_values.csv"
    nan_summary = pd.DataFrame(
        {
            "column": df.columns,
            "n_missing": df.isna().sum().values,
            "pct_missing": (df.isna().sum() / len(df) * 100).round(2).values,
        }
    ).sort_values("pct_missing", ascending=False)
    nan_summary.to_csv(nan_path, index=False)
    print(f"[EDA] Synthèse NaN · {nan_path}")
    print("Done.")


if __name__ == "__main__":
    main()
