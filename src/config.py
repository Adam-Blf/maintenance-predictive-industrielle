# -*- coding: utf-8 -*-
"""Configuration centrale du projet Maintenance Prédictive Industrielle.

Principe d'architecture
-----------------------
Ce module est le "single source of truth" de toutes les constantes du
projet : chemins de fichiers, hyperparamètres de modélisation, noms de
colonnes et charte graphique. Aucun autre module ne doit hardcoder ces
valeurs ; ils importent depuis `src.config`.

Cette approche offre trois avantages ·
  1. **Maintenabilité** · un seul endroit à modifier pour changer le
     chemin du dataset, la graine aléatoire ou les features.
  2. **Portabilité** · tous les chemins sont calculés relativement à
     `PROJECT_ROOT` (résolu via `__file__`), indépendamment du cwd.
  3. **Reproductibilité** · la graine RANDOM_STATE est propagée à tous
     les algorithmes stochastiques pour garantir des résultats
     bit-à-bit identiques entre exécutions.

Conventions d'import
---------------------
Les scripts dans `scripts/` ajoutent `PROJECT_ROOT` au `sys.path` et
importent ainsi : `from src.config import DATASET_PATH, RANDOM_STATE`.
Le dashboard et l'API font de même depuis leur propre répertoire.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Arborescence du projet · racine = parent du répertoire `src`.
# Tous les chemins sont calculés à partir de ce point pour ne pas dépendre
# du cwd lorsqu'on exécute un script depuis un autre dossier.
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# Données brutes · CSV Kaggle officiel (CC0 public domain).
DATA_RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"

# Données prétraitées · exports intermédiaires (train/test/scaler features).
DATA_PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"

# Artefacts modèles sérialisés (joblib) · un fichier par modèle entraîné.
MODELS_DIR: Path = PROJECT_ROOT / "models"

# Rapport PDF généré + dossier reports/ pour les annexes.
REPORTS_DIR: Path = PROJECT_ROOT / "reports"

# Convention de sortie · chaque script `scripts/NN_*.py` écrit ses
# artefacts (figures, CSV, JSON) dans `reports/NN/` à plat. Cela garantit
# une traçabilité directe entre un fichier de sortie et le script qui l'a
# produit · `reports/02/eda_target_distribution.png` ⇔ `scripts/02_eda.py`.
def script_output_dir(script_number: int) -> Path:
    """Retourne le dossier de sortie d'un script numéroté.

    Crée le dossier si nécessaire (idempotent). `script_number=2` retourne
    `reports/02/`. Les figures, CSV, JSON et autres artefacts du script
    vivent à plat dans ce dossier (pas de sous-niveau `figures/`).
    """
    out = REPORTS_DIR / f"{script_number:02d}"
    out.mkdir(parents=True, exist_ok=True)
    return out


# Constantes par script · pré-calculées pour import direct depuis les
# scripts. `from src.config import S02_DIR` puis `S02_DIR / "fig.png"`.
S02_DIR: Path = REPORTS_DIR / "02"  # EDA descriptive
S03_DIR: Path = REPORTS_DIR / "03"  # Train binaire (3 modèles + baseline)
S04_DIR: Path = REPORTS_DIR / "04"  # Interprétabilité (SHAP, permutation)
S05_DIR: Path = REPORTS_DIR / "05"  # Schémas pédagogiques
S06_DIR: Path = REPORTS_DIR / "06"  # Rapport PDF final (livrable jury)
S07_DIR: Path = REPORTS_DIR / "07"  # Train multiclass
S08_DIR: Path = REPORTS_DIR / "08"  # Train régression RUL
S09_DIR: Path = REPORTS_DIR / "09"  # Tuning Optuna
S10_DIR: Path = REPORTS_DIR / "10"  # Calibration + cost-sensitive threshold
S11_DIR: Path = REPORTS_DIR / "11"  # Présentation PPTX (livrable jury)

# Logo EFREI utilisé en page de garde du rapport et dans le dashboard.
ASSETS_DIR: Path = PROJECT_ROOT / "assets"
EFREI_LOGO: Path = ASSETS_DIR / "logo_efrei.png"
# Variante blanche du logo EFREI · pour les slides à fond sombre (navy).
EFREI_LOGO_WHITE: Path = ASSETS_DIR / "logo_efrei_white.png"
# Variante noire monochrome du logo EFREI · pour impression noir et blanc.
EFREI_LOGO_BLACK: Path = ASSETS_DIR / "logo_efrei_noir.png"

# Nom canonique du dataset · fichier officiel Kaggle
# `tatheerabbas/industrial-machine-predictive-maintenance` v3.0 (CC0 public
# domain). 24 042 lignes · 15 colonnes · NaN volontaires (~4% capteurs).
DATASET_FILENAME: str = "predictive_maintenance_v3.csv"
DATASET_PATH: Path = DATA_RAW_DIR / DATASET_FILENAME
DATASET_KAGGLE_REF: str = "tatheerabbas/industrial-machine-predictive-maintenance"

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

# Variable cible quaternaire (bonus) · coût estimé de réparation.
TARGET_REGRESSION_COST: str = "estimated_repair_cost"

# ---------------------------------------------------------------------------
# Variables d'entrée du dataset · capteurs physiques + contexte machine.
# Schéma officiel Kaggle v3.0 · `predictive_maintenance_v3.csv`.
# L'ordre est important · il sert de contrat d'interface pour le dashboard
# Streamlit ET pour l'API FastAPI (validation Pydantic).
# ---------------------------------------------------------------------------
NUMERIC_FEATURES: list[str] = [
    "vibration_rms",  # Vibration efficace en mm/s (capteur principal)
    "temperature_motor",  # Température du moteur en °C
    "current_phase_avg",  # Courant absorbé moyen sur 3 phases en A
    "pressure_level",  # Pression du circuit hydraulique en bar
    "rpm",  # Vitesse de rotation en tours/minute
    "hours_since_maintenance",  # Heures depuis dernière maintenance
    "ambient_temp",  # Température ambiante en °C (contexte)
]

# Variables catégorielles · mode opératoire + type de machine.
CATEGORICAL_FEATURES: list[str] = ["operating_mode", "machine_type"]

# Modes opératoires possibles · valeurs canoniques du dataset Kaggle.
OPERATING_MODES: list[str] = ["normal", "idle", "peak"]

# Types de machine du parc industriel simulé.
MACHINE_TYPES: list[str] = ["CNC", "Pump", "Compressor", "Robotic Arm"]

# Types de panne possibles (classification multi-classe).
FAILURE_TYPES: list[str] = [
    "none",  # Pas de panne (machine saine)
    "bearing",  # Usure roulement (mécanique)
    "motor_overheat",  # Surchauffe moteur (thermique)
    "hydraulic",  # Fuite ou perte de pression hydraulique
    "electrical",  # Défaut électrique (surintensité, sous-tension)
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

    Idempotente · appeler plusieurs fois ne crée pas de doublons ni ne
    lève d'exception si le dossier existe déjà (`exist_ok=True`).
    Invoquée en début de chaque script de `scripts/` pour garantir que
    l'environnement d'exécution est prêt avant tout accès disque.

    Dossiers créés
    --------------
    - data/raw         · CSV Kaggle officiel.
    - data/processed   · exports train/test splits.
    - models/          · artefacts joblib des modèles entraînés.
    - reports/         · racine, rapport PDF agrégé.
    - reports/NN/      · sorties par script (NN = numéro du script).
    """
    base_dirs = (
        DATA_RAW_DIR,
        DATA_PROCESSED_DIR,
        MODELS_DIR,
        REPORTS_DIR,
    )
    script_dirs = (
        S02_DIR, S03_DIR, S04_DIR, S05_DIR, S06_DIR,
        S07_DIR, S08_DIR, S09_DIR, S10_DIR, S11_DIR,
    )
    for directory in base_dirs + script_dirs:
        directory.mkdir(parents=True, exist_ok=True)
