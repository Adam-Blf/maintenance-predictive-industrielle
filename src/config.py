# -*- coding: utf-8 -*-
"""Configuration centrale du projet Maintenance Prédictive Industrielle.

Ce module isole tous les chemins, constantes et hyperparamètres globaux pour
éviter la duplication entre scripts et faciliter la maintenance. On adopte
une approche `pathlib` pour rester portable Windows/Linux/macOS.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Arborescence du projet · racine = parent du répertoire `src`.
# Tous les chemins sont calculés à partir de ce point pour ne pas dépendre
# du cwd lorsqu'on exécute un script depuis un autre dossier.
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# Données brutes (CSV Kaggle ou version synthétique reproduisant le schéma).
DATA_RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"

# Données prétraitées · exports intermédiaires (train/test/scaler features).
DATA_PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"

# Artefacts modèles sérialisés (joblib) · un fichier par modèle entraîné.
MODELS_DIR: Path = PROJECT_ROOT / "models"

# Sortie graphique · figures EDA, matrices de confusion, ROC, schémas
# pédagogiques utilisés par FPDF2 dans le rapport final.
REPORTS_FIGURES_DIR: Path = PROJECT_ROOT / "reports" / "figures"

# Rapport PDF généré + dossier reports/ pour les annexes.
REPORTS_DIR: Path = PROJECT_ROOT / "reports"

# Logo EFREI utilisé en page de garde du rapport et dans le dashboard.
ASSETS_DIR: Path = PROJECT_ROOT / "assets"
EFREI_LOGO: Path = ASSETS_DIR / "logo_efrei.png"
# Variante blanche du logo EFREI · pour les slides à fond sombre (navy).
EFREI_LOGO_WHITE: Path = ASSETS_DIR / "logo_efrei_white.png"
# Variante noire monochrome du logo EFREI · pour impression noir et blanc.
EFREI_LOGO_BLACK: Path = ASSETS_DIR / "logo_efrei_noir.png"

# Nom canonique du dataset (cohérent avec le sujet officiel Kaggle).
DATASET_FILENAME: str = "industrial_machine_maintenance.csv"
DATASET_PATH: Path = DATA_RAW_DIR / DATASET_FILENAME

# ---------------------------------------------------------------------------
# Hyperparamètres de la phase de modélisation.
# Centralisés ici pour permettre une expérimentation rapide sans toucher
# aux scripts.
# ---------------------------------------------------------------------------

# Graine aléatoire propagée à numpy / sklearn / xgboost · garantit la
# reproductibilité bit-à-bit des résultats sur la même machine.
RANDOM_STATE: int = 42

# Proportion de test dans le split stratifié · 20% est un compromis
# classique entre représentativité du test set et données d'entraînement.
TEST_SIZE: float = 0.20

# Nombre de folds pour la cross-validation stratifiée. 5 folds est le
# standard académique · bon compromis biais/variance/temps de calcul.
CV_FOLDS: int = 5

# Variable cible principale · classification binaire de la panne dans 24h
# (la "tâche prédictive la plus naturelle du dataset" selon le sujet).
TARGET_BINARY: str = "failure_within_24h"

# Variable cible secondaire · classification multi-classe du type de panne
# (utilisée en bonus pour valoriser la grille de notation).
TARGET_MULTICLASS: str = "failure_type"

# Variable cible tertiaire · régression sur la durée de vie restante.
TARGET_REGRESSION: str = "rul_hours"

# ---------------------------------------------------------------------------
# Variables d'entrée du dataset · capteurs physiques + contexte machine.
# L'ordre est important · il sert de contrat d'interface pour le dashboard
# Streamlit ET pour l'API FastAPI (validation Pydantic).
# ---------------------------------------------------------------------------
NUMERIC_FEATURES: list[str] = [
    "vibration_rms",  # Vibration efficace en mm/s (capteur principal)
    "temperature_motor",  # Température du moteur en °C
    "rpm",  # Vitesse de rotation en tours/minute
    "pressure_level",  # Pression du circuit hydraulique en bar
    "ambient_temperature",  # Température ambiante en °C (contexte)
    "humidity",  # Humidité relative en % (contexte)
    "voltage",  # Tension d'alimentation en V
    "current",  # Courant absorbé en A
    "power_consumption",  # Puissance instantanée en kW
    "maintenance_age_days",  # Jours depuis dernière maintenance
]

# Variables catégorielles · seul `operating_mode` est qualitatif.
CATEGORICAL_FEATURES: list[str] = ["operating_mode"]

# Modes opératoires possibles · valeurs canoniques du dataset.
OPERATING_MODES: list[str] = ["Normal", "HighLoad", "Idle", "Maintenance"]

# Types de panne possibles (classification multi-classe).
FAILURE_TYPES: list[str] = [
    "None",  # Pas de panne (machine saine)
    "Mechanical",  # Usure mécanique (roulement, axe)
    "Electrical",  # Défaut électrique (court-circuit, surtension)
    "Thermal",  # Surchauffe / refroidissement insuffisant
    "Hydraulic",  # Fuite ou perte de pression
]

# Liste complète des features (utilisée pour bâtir le ColumnTransformer).
ALL_FEATURES: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# ---------------------------------------------------------------------------
# Charte graphique du projet · cohérente avec l'identité EFREI (bleu) et
# avec la sémantique "alerte" pour la maintenance prédictive (rouge).
# ---------------------------------------------------------------------------
COLOR_EFREI_BLUE: str = "#1E88E5"  # Bleu institutionnel EFREI
COLOR_EFREI_DARK: str = "#0D47A1"  # Bleu nuit pour titres
COLOR_ALERT_RED: str = "#E53935"  # Rouge alerte panne
COLOR_OK_GREEN: str = "#43A047"  # Vert OK / machine saine
COLOR_WARNING: str = "#FB8C00"  # Orange avertissement


def ensure_directories() -> None:
    """Crée tous les dossiers de sortie s'ils n'existent pas déjà.

    Cette fonction est idempotente · on peut l'appeler plusieurs fois sans
    effet de bord. Elle est invoquée en début de chaque script du dossier
    `scripts/` pour garantir que l'environnement d'exécution est prêt.
    """
    for directory in (
        DATA_RAW_DIR,
        DATA_PROCESSED_DIR,
        MODELS_DIR,
        REPORTS_FIGURES_DIR,
        REPORTS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
