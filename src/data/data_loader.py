# -*- coding: utf-8 -*-
"""Module de chargement des données.

À QUOI ÇA SERT ?
----------------
Une seule fonction publique · `load_dataset()` lit le CSV Kaggle officiel
depuis `data/raw/predictive_maintenance_v3.csv` et retourne un DataFrame
pandas validé contre le schéma attendu.

POLITIQUE STRICTE · PAS DE GÉNÉRATION DE DONNÉES
-------------------------------------------------
Le projet utilise EXCLUSIVEMENT le dataset officiel Kaggle
`tatheerabbas/industrial-machine-predictive-maintenance` v3.0 (CC0).
Aucun fallback synthétique, aucune génération de données. Si le CSV est
absent, on plante avec un message clair indiquant comment le télécharger.

Pourquoi · les données synthétiques masquent les biais du dataset réel
et donnent une fausse impression de qualité. Le sujet RNCP exige de
travailler sur des données industrielles réalistes · on s'y tient.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import DATASET_KAGGLE_REF, DATASET_PATH


# Schéma exact attendu (15 colonnes officielles) · contrat d'interface
# pour la pipeline. Toute déviation de ce schéma est une erreur immédiate.
EXPECTED_COLUMNS: list[str] = [
    "timestamp",
    "machine_id",
    "machine_type",
    "vibration_rms",
    "temperature_motor",
    "current_phase_avg",
    "pressure_level",
    "rpm",
    "operating_mode",
    "hours_since_maintenance",
    "ambient_temp",
    "rul_hours",
    "failure_within_24h",
    "failure_type",
    "estimated_repair_cost",
]


def load_dataset(path: Path | None = None) -> pd.DataFrame:
    """Charge le dataset CSV Kaggle officiel depuis le disque.

    Parameters
    ----------
    path : Path | None, optional
        Chemin vers le CSV. Si `None`, on utilise `DATASET_PATH` qui pointe
        sur `data/raw/predictive_maintenance_v3.csv` (officiel Kaggle).

    Returns
    -------
    pd.DataFrame
        DataFrame chargé · 24042 lignes, 15 colonnes.

    Raises
    ------
    FileNotFoundError
        Si le fichier n'existe pas. Le message indique comment le
        télécharger depuis Kaggle. AUCUN FALLBACK SYNTHÉTIQUE.
    ValueError
        Si le CSV chargé ne respecte pas le schéma attendu (15 colonnes
        canoniques). Bug ou mauvais fichier téléchargé.
    """
    target_path = path if path is not None else DATASET_PATH
    if not target_path.exists():
        raise FileNotFoundError(
            f"Dataset introuvable · {target_path}\n"
            f"Telecharger le CSV officiel Kaggle ({DATASET_KAGGLE_REF}) "
            "depuis https://www.kaggle.com/datasets/tatheerabbas/"
            "industrial-machine-predictive-maintenance, extraire le "
            "fichier predictive_maintenance_v3.csv dans data/raw/, "
            "puis relancer le script."
        )
    df = pd.read_csv(target_path, low_memory=False)
    _validate_schema(df)
    return df


def _validate_schema(df: pd.DataFrame) -> None:
    """Vérifie que le DataFrame respecte le schéma officiel Kaggle v3.0."""
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"Colonnes manquantes par rapport au schéma officiel · {sorted(missing)}"
        )
