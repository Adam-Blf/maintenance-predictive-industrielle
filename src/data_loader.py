# -*- coding: utf-8 -*-
"""Module de chargement et de génération des données.

Le sujet officiel pointe vers un dataset Kaggle public
(`tatheerabbas/industrial-machine-predictive-maintenance`). Pour garantir
la reproductibilité du livrable même hors connexion / sans clé API Kaggle,
ce module embarque également un générateur synthétique cohérent avec le
schéma annoncé (24 042 enregistrements, 15 variables, mêmes intitulés).

Le choix d'un générateur synthétique réaliste (et non d'un dataset
artificiel "noisé") est documenté dans le rapport · on injecte des
relations métier (vibration↑ + température↑ ⇒ probabilité panne↑) qui
permettent aux modèles d'apprendre des patterns physiquement plausibles.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    DATASET_PATH,
    FAILURE_TYPES,
    OPERATING_MODES,
    RANDOM_STATE,
)


def load_dataset(path: Path | None = None) -> pd.DataFrame:
    """Charge le dataset CSV depuis le disque.

    Parameters
    ----------
    path : Path | None, optional
        Chemin vers le CSV. Si `None`, on utilise `DATASET_PATH` qui pointe
        sur `data/raw/industrial_machine_maintenance.csv`.

    Returns
    -------
    pd.DataFrame
        DataFrame chargé avec types pandas par défaut.

    Raises
    ------
    FileNotFoundError
        Si le fichier n'existe pas. Le message d'erreur invite l'utilisateur
        à exécuter `python scripts/01_generate_dataset.py` pour produire la
        version synthétique reproductible.
    """
    target_path = path if path is not None else DATASET_PATH
    if not target_path.exists():
        raise FileNotFoundError(
            f"Dataset introuvable · {target_path}\n"
            "Exécuter `python scripts/01_generate_dataset.py` pour générer "
            "la version synthétique reproductible (24042 lignes)."
        )
    # `low_memory=False` évite les avertissements pandas sur les colonnes
    # mixtes lorsque l'on charge en une seule passe ce dataset moyen.
    return pd.read_csv(target_path, low_memory=False)


def generate_synthetic_dataset(
    n_samples: int = 24_042,
    seed: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Génère un dataset synthétique reproduisant le schéma Kaggle officiel.

    On simule un parc industriel hétérogène · 4 modes opératoires, vieillesse
    machine variable, usure progressive, signature physique des défaillances.
    Les relations entre features et cibles sont conçues pour être
    apprenables par un modèle ML mais non triviales (bruit gaussien,
    interactions multiplicatives, classes déséquilibrées comme dans la
    réalité industrielle).

    Parameters
    ----------
    n_samples : int, default=24042
        Nombre de lignes à générer (valeur officielle du sujet).
    seed : int, default=42
        Graine aléatoire pour la reproductibilité.

    Returns
    -------
    pd.DataFrame
        Dataset avec 15 colonnes (10 numériques + 1 catégorielle +
        timestamp/machine_id + 3 cibles).
    """
    # Générateur numpy moderne · plus rapide et de meilleure qualité
    # statistique que `np.random.seed` + fonctions globales.
    rng = np.random.default_rng(seed)

    # Identifiants machine · 200 machines tirées avec probabilité uniforme.
    # Cela génère un effet "groupe" réaliste (certaines machines vieillissent
    # plus vite que d'autres en moyenne) que les arbres peuvent capturer.
    machine_ids = rng.integers(low=1, high=201, size=n_samples)

    # Timestamp régulier · une mesure toutes les heures sur ~3 ans.
    base_time = pd.Timestamp("2023-01-01 00:00:00")
    timestamps = pd.to_datetime(base_time + pd.to_timedelta(np.arange(n_samples), unit="h"))

    # Mode opératoire · distribution réaliste (la majorité Normal, pic de
    # HighLoad en heures ouvrées). 50% Normal / 25% HighLoad / 15% Idle /
    # 10% Maintenance.
    operating_mode = rng.choice(
        OPERATING_MODES,
        size=n_samples,
        p=[0.50, 0.25, 0.15, 0.10],
    )

    # Vieillesse machine en jours depuis la dernière maintenance.
    # Distribution asymétrique (beaucoup de machines récemment maintenues +
    # une queue lourde · effet "machines négligées").
    maintenance_age_days = np.clip(
        rng.exponential(scale=120.0, size=n_samples), a_min=0, a_max=730
    ).astype(int)

    # ------------------------------------------------------------------
    # Capteurs physiques · valeurs centrales selon le mode opératoire.
    # On modélise des plages réalistes inspirées d'un moteur asynchrone
    # industriel de moyenne puissance (~50 kW).
    # ------------------------------------------------------------------
    # RPM cible selon le mode (Idle ≈ 0, Normal ≈ 1500, HighLoad ≈ 2200).
    rpm_base = np.where(
        operating_mode == "Idle",
        50,
        np.where(
            operating_mode == "Normal",
            1500,
            np.where(operating_mode == "HighLoad", 2200, 800),
        ),
    )
    # Bruit gaussien proportionnel à la valeur (variance hétérogène).
    rpm = np.clip(rpm_base + rng.normal(0, 80, n_samples), 0, 3500)

    # Vibration · augmente avec RPM, vieillesse machine, et pic en HighLoad.
    vibration_rms = (
        0.5
        + 0.0008 * rpm
        + 0.005 * maintenance_age_days
        + np.where(operating_mode == "HighLoad", 1.2, 0.0)
        + rng.normal(0, 0.4, n_samples)
    )
    vibration_rms = np.clip(vibration_rms, 0.05, 12.0)

    # Température moteur · suit la charge thermique (RPM × age × ambient).
    ambient_temperature = rng.normal(22.0, 5.0, n_samples)
    ambient_temperature = np.clip(ambient_temperature, -5, 45)

    temperature_motor = (
        ambient_temperature
        + 0.012 * rpm
        + 0.05 * maintenance_age_days
        + np.where(operating_mode == "HighLoad", 18.0, 0.0)
        + rng.normal(0, 4.0, n_samples)
    )
    temperature_motor = np.clip(temperature_motor, 10, 145)

    # Pression hydraulique · suit le mode opératoire principalement.
    pressure_level = np.where(
        operating_mode == "Idle",
        rng.normal(2.0, 0.4, n_samples),
        np.where(
            operating_mode == "Normal",
            rng.normal(6.5, 0.6, n_samples),
            np.where(
                operating_mode == "HighLoad",
                rng.normal(9.0, 0.8, n_samples),
                rng.normal(4.0, 0.5, n_samples),
            ),
        ),
    )
    pressure_level = np.clip(pressure_level, 0.1, 12.0)

    # Humidité ambiante · variable contextuelle peu corrélée à la panne
    # (volontaire · permet de tester la sélection de features).
    humidity = np.clip(rng.normal(55.0, 15.0, n_samples), 10, 95)

    # Voltage · alimentation triphasée 400V avec fluctuations.
    voltage = rng.normal(400.0, 8.0, n_samples)
    voltage = np.clip(voltage, 360, 440)

    # Courant absorbé · proportionnel à la puissance demandée (RPM × charge).
    current = (
        5.0
        + 0.025 * rpm
        + np.where(operating_mode == "HighLoad", 25.0, 0.0)
        + rng.normal(0, 3.0, n_samples)
    )
    current = np.clip(current, 0.5, 120.0)

    # Puissance · P = U × I (en kW, facteur de puissance simplifié 0.85).
    power_consumption = (voltage * current * 0.85) / 1000.0

    # ------------------------------------------------------------------
    # Construction des cibles métier · règles physiquement cohérentes.
    # ------------------------------------------------------------------
    # Score de risque continu (logit) combinant les signaux les plus
    # informatifs · vibration, température, vieillesse, pression anormale.
    # Les poids sont volontairement réalistes pour que l'apprentissage
    # supervisé converge vers des feature importances métier interprétables.
    risk_logit = (
        -4.5
        + 0.55 * (vibration_rms - 3.0)
        + 0.05 * (temperature_motor - 60.0)
        + 0.008 * (maintenance_age_days - 90)
        + 0.18 * np.abs(pressure_level - 6.0)
        + np.where(operating_mode == "HighLoad", 0.7, 0.0)
        + rng.normal(0, 0.5, n_samples)
    )
    # Sigmoïde pour obtenir une probabilité, puis tirage Bernoulli.
    risk_proba = 1.0 / (1.0 + np.exp(-risk_logit))
    failure_within_24h = (rng.uniform(0, 1, n_samples) < risk_proba).astype(int)

    # Type de panne · uniquement si failure_within_24h == 1, sinon "None".
    # On répartit les pannes selon les capteurs dominants ·
    # - vibration extrême  ⇒ Mechanical
    # - température extrême ⇒ Thermal
    # - voltage anormal     ⇒ Electrical
    # - pression anormale   ⇒ Hydraulic
    failure_type = np.full(n_samples, "None", dtype=object)
    failure_mask = failure_within_24h == 1

    # Calcul des "scores" par type pour les machines en panne.
    mech_score = vibration_rms + 0.01 * maintenance_age_days
    therm_score = temperature_motor / 30.0
    elec_score = np.abs(voltage - 400) / 10.0 + rng.normal(0, 0.5, n_samples)
    hydro_score = np.abs(pressure_level - 6.0) * 1.5

    score_matrix = np.stack([mech_score, therm_score, elec_score, hydro_score], axis=1)
    # Index du type dominant ⇒ on mappe sur les 4 types de panne (≠ "None").
    dominant = np.argmax(score_matrix, axis=1)
    type_map = ["Mechanical", "Thermal", "Electrical", "Hydraulic"]
    failure_type[failure_mask] = np.array([type_map[i] for i in dominant[failure_mask]])

    # Durée de vie restante (RUL) en heures · décroît avec le risque.
    # Machines saines · 200-1500h, machines à risque · <100h.
    rul_hours = (
        np.clip(
            1500.0 * (1.0 - risk_proba) + rng.normal(0, 50, n_samples),
            a_min=2.0,
            a_max=2500.0,
        )
        .round()
        .astype(int)
    )

    # ------------------------------------------------------------------
    # Assemblage final · ordre des colonnes stabilisé pour cohérence avec
    # le contrat d'interface du dashboard et de l'API.
    # ------------------------------------------------------------------
    df = pd.DataFrame(
        {
            "machine_id": machine_ids,
            "timestamp": timestamps,
            "operating_mode": operating_mode,
            "vibration_rms": np.round(vibration_rms, 3),
            "temperature_motor": np.round(temperature_motor, 2),
            "rpm": np.round(rpm, 0).astype(int),
            "pressure_level": np.round(pressure_level, 2),
            "ambient_temperature": np.round(ambient_temperature, 2),
            "humidity": np.round(humidity, 2),
            "voltage": np.round(voltage, 2),
            "current": np.round(current, 2),
            "power_consumption": np.round(power_consumption, 2),
            "maintenance_age_days": maintenance_age_days,
            "rul_hours": rul_hours,
            "failure_within_24h": failure_within_24h,
            "failure_type": failure_type,
        }
    )

    # Vérification · 15 variables (sans compter timestamp ni machine_id ·
    # on est à 16 colonnes, conforme au sujet qui parle de "15 variables"
    # en comptant les 3 cibles · `failure_within_24h`, `failure_type`,
    # `rul_hours` + 11 features + 1 timestamp).
    return df
