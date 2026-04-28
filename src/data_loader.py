# -*- coding: utf-8 -*-
"""Module de chargement des données.

Source officielle · dataset Kaggle CC0 public domain
`tatheerabbas/industrial-machine-predictive-maintenance` v3.0 · 24042
lignes, 15 colonnes, NaN volontaires (~4% sur 5 capteurs).

Le fichier `predictive_maintenance_v3.csv` est versionné dans
`data/raw/` (téléchargement manuel ou via MCP Kaggle). Un générateur
synthétique de secours respecte exactement le schéma officiel pour
permettre à un évaluateur de relancer la pipeline hors ligne.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    DATASET_KAGGLE_REF,
    DATASET_PATH,
    FAILURE_TYPES,
    MACHINE_TYPES,
    OPERATING_MODES,
    RANDOM_STATE,
)


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
    """Charge le dataset CSV depuis le disque.

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
        Si le fichier n'existe pas. Indique a l'utilisateur de
        telecharger le CSV Kaggle officiel et de l'extraire dans
        `data/raw/`. Aucun fallback synthetique en production.
    """
    target_path = path if path is not None else DATASET_PATH
    if not target_path.exists():
        raise FileNotFoundError(
            f"Dataset introuvable · {target_path}\n"
            f"Telecharger le CSV officiel Kaggle ({DATASET_KAGGLE_REF}) "
            "depuis https://www.kaggle.com/datasets/tatheerabbas/"
            "industrial-machine-predictive-maintenance, extraire le "
            "fichier predictive_maintenance_v3.csv dans data/raw/, "
            "puis relancer `python scripts/02_eda.py`. "
            "Le projet n'utilise JAMAIS de dataset synthetique en production · "
            "generate_synthetic_dataset() est reservee aux tests unitaires."
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


def generate_synthetic_dataset(
    n_samples: int = 24_042,
    seed: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Génère un dataset synthétique au schéma officiel Kaggle v3.0.

    Fallback pour les évaluateurs qui n'ont pas accès au CSV Kaggle (offline,
    pas de clé API). Reproduit · 4 types de machines, 3 modes opératoires,
    NaN aléatoires sur les capteurs (~4%), 5 types de panne avec proportions
    réalistes (15% pannes globales), corrélations physiques entre features.

    Parameters
    ----------
    n_samples : int, default=24042
        Nombre de lignes à générer (valeur officielle du dataset Kaggle).
    seed : int, default=42
        Graine aléatoire pour la reproductibilité.

    Returns
    -------
    pd.DataFrame
        Dataset au schéma officiel exact (15 colonnes, ordre canonique).
    """
    rng = np.random.default_rng(seed)

    # Timestamps horaires sur ~3 ans à partir de 2024-01-01.
    # L'espacement horaire reproduit le schéma observé dans le dataset
    # Kaggle (une mesure par heure et par machine).
    base_time = pd.Timestamp("2024-01-01 00:00:00")
    timestamps = pd.to_datetime(base_time + pd.to_timedelta(np.arange(n_samples), unit="h"))

    # 20 machines uniques (cf. métadonnées Kaggle v3.0), tirées uniformément.
    # high=21 car rng.integers est exclusif sur la borne supérieure.
    machine_ids = rng.integers(low=1, high=21, size=n_samples)

    # Types de machine · répartition équilibrée 4 catégories.
    # Uniforme intentionnel : chaque type doit être suffisamment représenté
    # pour que le ColumnTransformer OHE voit chaque modalité au fit.
    machine_type = rng.choice(MACHINE_TYPES, size=n_samples)

    # Modes opératoires · distribution observée dans Kaggle v3.0 ·
    # 48% normal / 45% idle / 7% peak. Le mode peak est rare car il
    # correspond à des surcharges ponctuelles, pas au fonctionnement courant.
    operating_mode = rng.choice(OPERATING_MODES, size=n_samples, p=[0.48, 0.45, 0.07])

    # Heures depuis maintenance · loi exponentielle (queue lourde) reproduit
    # la réalité industrielle : la plupart des machines ont été entretenues
    # récemment, mais quelques-unes accumulent de longues périodes sans
    # intervention (négligence, sous-effectif). Clip à 2000h = ~83 jours max.
    hours_since_maintenance = np.clip(
        rng.exponential(scale=200.0, size=n_samples), a_min=0, a_max=2000
    )

    # ------------------------------------------------------------------
    # Capteurs physiques · valeurs centrales selon le mode opératoire.
    # Plages calibrées pour reproduire la distribution observée Kaggle.
    # La dépendance aux modes opératoires est intentionnelle : elle crée
    # des corrélations réalistes entre features, ce qui rend le dataset
    # plus difficile pour les modèles et plus représentatif du réel.
    # ------------------------------------------------------------------
    rpm_base = np.where(
        operating_mode == "idle", 850,
        np.where(operating_mode == "normal", 1500, 2200),
    )
    # Bruit gaussien faible (sigma=80 rpm) pour simuler les fluctuations
    # de régulation PID autour du setpoint.
    rpm = np.clip(rpm_base + rng.normal(0, 80, n_samples), 0, 3500)

    ambient_temp = np.clip(rng.normal(20.0, 6.0, n_samples), -10, 45)

    vibration_rms = np.clip(
        0.5
        + 0.0008 * rpm
        + 0.0015 * hours_since_maintenance
        + np.where(operating_mode == "peak", 1.2, 0.0)
        + rng.normal(0, 0.4, n_samples),
        0.05, 12.0,
    )

    temperature_motor = np.clip(
        ambient_temp
        + 0.012 * rpm
        + 0.015 * hours_since_maintenance
        + np.where(operating_mode == "peak", 18.0, 0.0)
        + rng.normal(0, 4.0, n_samples),
        10, 145,
    )

    pressure_level = np.where(
        operating_mode == "idle",
        rng.normal(15.0, 3.0, n_samples),
        np.where(operating_mode == "normal",
                 rng.normal(35.0, 5.0, n_samples),
                 rng.normal(60.0, 7.0, n_samples)),
    )
    pressure_level = np.clip(pressure_level, 0.1, 100.0)

    current_phase_avg = np.clip(
        2.0
        + 0.005 * rpm
        + np.where(operating_mode == "peak", 8.0, 0.0)
        + rng.normal(0, 1.0, n_samples),
        0.5, 30.0,
    )

    # ------------------------------------------------------------------
    # Cibles · règles physiquement cohérentes alignées Kaggle v3.0.
    # ------------------------------------------------------------------
    # Le risque de panne est modélisé comme une régression logistique
    # "oracle" dont on connaît les coefficients vrais. Cette approche
    # garantit que les features contiennent effectivement l'information
    # nécessaire pour prédire la cible (les modèles ML doivent pouvoir
    # retrouver ces coefficients approximativement).
    # Intercept = -2.5 → probabilité de base ~8% (taux de panne réaliste).
    risk_logit = (
        -2.5
        + 0.45 * (vibration_rms - 3.0)       # vibration : signal dominant
        + 0.04 * (temperature_motor - 60.0)   # surchauffe : signal secondaire
        + 0.003 * (hours_since_maintenance - 200)  # vieillissement progressif
        + 0.06 * np.abs(pressure_level - 35.0) / 10.0  # anomalie hydraulique
        + np.where(operating_mode == "peak", 0.6, 0.0)  # surcharge : facteur aggravant
        + rng.normal(0, 0.5, n_samples)  # bruit aléatoire : irréductible
    )
    risk_proba = 1.0 / (1.0 + np.exp(-risk_logit))
    # Simulation d'un tirage de Bernoulli pour chaque observation ·
    # l'indicateur binaire est 1 si la probabilité de panne dépasse
    # un seuil aléatoire uniforme (équivalent à un tirage Bernoulli(p)).
    failure_within_24h = (rng.uniform(0, 1, n_samples) < risk_proba).astype(int)

    # Failure type · multinomial selon les capteurs dominants.
    # Logique physique : le type de panne est déterminé par le signal
    # le plus anormal. Un softmax avec température 1.2 (légère accentuation)
    # sur les scores bruts convertit les scores en probabilités de classe.
    failure_type = np.full(n_samples, "none", dtype=object)
    failure_mask = failure_within_24h == 1
    n_failed = int(failure_mask.sum())
    if n_failed > 0:
        # Score bearing : vibration élevée + machine ancienne → usure roulement.
        bearing_score = vibration_rms[failure_mask] / 6.0 + 0.002 * hours_since_maintenance[failure_mask]
        # Score thermique : température moteur très au-dessus de 50°C → surchauffe.
        thermal_score = np.maximum(temperature_motor[failure_mask] - 50.0, 0.0) / 30.0
        # Score électrique : courant anormalement éloigné du nominal (8A) → défaut élec.
        elec_score = np.abs(current_phase_avg[failure_mask] - 8.0) / 5.0 + 0.2
        # Score hydraulique : pression anormale (fuite ou blocage) → défaut hydraulique.
        hydro_score = np.abs(pressure_level[failure_mask] - 35.0) / 15.0 + 0.2

        # Softmax avec facteur d'accentuation 1.2 pour que le type "dominant"
        # soit choisi plus souvent (évite une distribution trop uniforme).
        scores = np.stack([bearing_score, thermal_score, elec_score, hydro_score], axis=1)
        exp_s = np.exp(1.2 * scores)
        probas = exp_s / exp_s.sum(axis=1, keepdims=True)
        cumprobas = probas.cumsum(axis=1)
        # Tirage par inversion CDF : on cherche le premier bin dont la CDF
        # dépasse un uniforme, équivalent à un tirage catégoriel pondéré.
        u = rng.uniform(0, 1, size=n_failed).reshape(-1, 1)
        chosen_idx = (u < cumprobas).argmax(axis=1)
        type_map = ["bearing", "motor_overheat", "electrical", "hydraulic"]
        failure_type[failure_mask] = np.array([type_map[i] for i in chosen_idx])

    # RUL en heures · 0.5h pour machines à risque, jusqu'à 99h pour saines.
    rul_hours = np.round(
        np.clip(99.0 * (1.0 - risk_proba) + rng.normal(0, 5, n_samples), 0.5, 99.0),
        2,
    )

    # Coût de réparation · 0 si pas de panne, sinon 500-8000€ selon le type.
    cost_by_type = {
        "none": 0,
        "bearing": 1500,
        "motor_overheat": 3500,
        "hydraulic": 2200,
        "electrical": 1800,
    }
    base_cost = np.array([cost_by_type[ft] for ft in failure_type])
    estimated_repair_cost = np.where(
        failure_within_24h == 1,
        np.clip(base_cost * rng.uniform(0.5, 2.5, n_samples), 0, 8000).astype(int),
        0,
    )

    # Injection de NaN sur 5 capteurs (~4% chacun), comme dans le dataset officiel.
    df = pd.DataFrame({
        "timestamp": timestamps,
        "machine_id": machine_ids,
        "machine_type": machine_type,
        "vibration_rms": np.round(vibration_rms, 2),
        "temperature_motor": np.round(temperature_motor, 2),
        "current_phase_avg": np.round(current_phase_avg, 2),
        "pressure_level": np.round(pressure_level, 1),
        "rpm": np.round(rpm, 1),
        "operating_mode": operating_mode,
        "hours_since_maintenance": np.round(hours_since_maintenance, 1),
        "ambient_temp": np.round(ambient_temp, 1),
        "rul_hours": rul_hours,
        "failure_within_24h": failure_within_24h,
        "failure_type": failure_type,
        "estimated_repair_cost": estimated_repair_cost,
    })

    # Injection NaN aléatoires (mimique les capteurs IoT réels qui dropent).
    for col, frac in [
        ("vibration_rms", 0.042),
        ("temperature_motor", 0.035),
        ("current_phase_avg", 0.030),
        ("pressure_level", 0.038),
        ("rpm", 0.022),
    ]:
        n_nan = int(n_samples * frac)
        idx = rng.choice(n_samples, size=n_nan, replace=False)
        df.loc[idx, col] = np.nan

    return df[EXPECTED_COLUMNS]
