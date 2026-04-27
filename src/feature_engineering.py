# -*- coding: utf-8 -*-
"""Module de feature engineering · variables dérivées capteurs.

Le sujet recommande explicitement d'envisager des variables derivees ·
*"parfois, une variable derivee (ex. ratio ou combinaison de capteurs)
est plus informative que les variables brutes"* (cahier des charges,
section Recommandations).

Variables produites ·
  - **vibration_per_rpm** · ratio · normalise la vibration par la vitesse
    (capture le sur-frottement quel que soit le regime moteur).
  - **temp_above_ambient** · ecart thermique moteur / ambiant (capture
    la charge thermique reelle, independante de la temperature exterieure).
  - **power_density** · puissance / rpm (capture la signature mecanique
    de la transmission).
  - **age_x_vibration** · interaction · machine vieillissante qui vibre
    fortement = signal premonitoire fort.
  - **age_x_temperature** · interaction analogue.
  - **pressure_deviation** · ecart absolu a la pression nominale du mode
    operatoire (anomalie hydraulique).

Ces features sont calculees AVANT le ColumnTransformer (donc avant
imputation/scaling) pour que les statistiques calculees lors du fit
incluent l'effet feature engineering.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Pression nominale par mode opératoire (cohérent avec le générateur).
NOMINAL_PRESSURE: dict[str, float] = {
    "Idle": 2.0,
    "Normal": 6.5,
    "HighLoad": 9.0,
    "Maintenance": 4.0,
}


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les variables derivees au DataFrame d'entree.

    Cette fonction est idempotente · on peut l'appeler plusieurs fois,
    elle ecrase les colonnes derivees deja presentes.
    """
    out = df.copy()

    # Ratio vibration / rpm (avec garde-fou contre rpm = 0).
    out["vibration_per_rpm"] = out["vibration_rms"] / np.maximum(out["rpm"], 1.0)

    # Ecart thermique moteur / ambiant.
    out["temp_above_ambient"] = out["temperature_motor"] - out["ambient_temperature"]

    # Densite de puissance · power / rpm.
    out["power_density"] = out["power_consumption"] / np.maximum(out["rpm"], 1.0)

    # Interactions age x signaux dominants.
    out["age_x_vibration"] = out["maintenance_age_days"] * out["vibration_rms"]
    out["age_x_temperature"] = out["maintenance_age_days"] * out["temperature_motor"] / 100.0

    # Deviation a la pression nominale du mode opératoire.
    nominal = out["operating_mode"].map(NOMINAL_PRESSURE).fillna(6.0)
    out["pressure_deviation"] = (out["pressure_level"] - nominal).abs()

    return out


# Liste des features derivees · utilisee pour etendre la liste des
# colonnes numeriques traitees par le ColumnTransformer.
ENGINEERED_NUMERIC_FEATURES: list[str] = [
    "vibration_per_rpm",
    "temp_above_ambient",
    "power_density",
    "age_x_vibration",
    "age_x_temperature",
    "pressure_deviation",
]
