# -*- coding: utf-8 -*-
"""Module de feature engineering · variables dérivées capteurs.

Le sujet recommande explicitement d'envisager des variables derivees ·
*"parfois, une variable derivee (ex. ratio ou combinaison de capteurs)
est plus informative que les variables brutes"* (cahier des charges,
section Recommandations).

Variables produites · alignees sur le schema Kaggle v3.0.
  - **vibration_per_rpm** · ratio · normalise la vibration par la vitesse
    (capture le sur-frottement quel que soit le regime moteur).
  - **temp_above_ambient** · ecart thermique moteur / ambiant (capture
    la charge thermique reelle, independante de la temperature exterieure).
  - **current_per_rpm** · ratio · courant absorbe / rpm (signature
    de la charge electrique reelle, anomalie si plus eleve qu'attendu).
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

# Pression nominale par mode opératoire · valeurs observées dataset Kaggle v3.0.
NOMINAL_PRESSURE: dict[str, float] = {
    "idle": 15.0,
    "normal": 35.0,
    "peak": 60.0,
}


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les variables derivees au DataFrame d'entree.

    Cette fonction est idempotente · on peut l'appeler plusieurs fois,
    elle ecrase les colonnes derivees deja presentes sans accumuler de
    duplicatas.

    Toutes les features sont calculees AVANT le ColumnTransformer pour
    que les statistiques de fit (médiane d'imputation, moyenne/écart-type
    de scaling) intègrent déjà l'effet du feature engineering. Calculer
    les features APRES le fit reviendrait à opérer sur des valeurs scalées
    perdant leur interprétabilité physique.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame brut issu du CSV Kaggle v3.0 (ou du générateur synthétique).
        Doit contenir les colonnes : vibration_rms, rpm, temperature_motor,
        ambient_temp, current_phase_avg, hours_since_maintenance,
        operating_mode, pressure_level.

    Returns
    -------
    pd.DataFrame
        Copie du DataFrame enrichi de 6 colonnes dérivées
        (voir ENGINEERED_NUMERIC_FEATURES pour la liste complète).

    Notes
    -----
    Pièges à éviter ·
    - Ne jamais appeler cette fonction APRES le split train/test · les
      statistiques NaN-fill (fillna(35.0) pour la pression) seraient
      calculées sur le test set, violant l'anti-data-leakage.
    - Le facteur x1000 sur current_per_rpm est intentionnel pour ramener
      la valeur dans une plage comparable aux autres features numériques
      et faciliter la convergence du MLP.
    """
    out = df.copy()

    # Ratio vibration / rpm : capture le sur-frottement independamment du
    # regime moteur. Un roulement usé produit une vibration anormalement
    # elevée même à bas régime. np.maximum evite la division par zéro.
    out["vibration_per_rpm"] = out["vibration_rms"] / np.maximum(out["rpm"], 1.0)

    # Ecart thermique moteur / ambiant : isole la chaleur propre au
    # fonctionnement de la machine par rapport aux variations saisonnières.
    # Un temperature_motor elevé par temps froid est plus inquiétant
    # qu'un temperature_motor similaire par forte chaleur ambiante.
    out["temp_above_ambient"] = out["temperature_motor"] - out["ambient_temp"]

    # Ratio courant / rpm · signature de la charge electrique reelle.
    # Si le moteur consomme beaucoup de courant pour peu de tours, cela
    # indique un frein mécanique ou un enroulement dégradé. Facteur x1000
    # pour ramener dans la plage [0, 30] au lieu de [0, 0.03].
    out["current_per_rpm"] = out["current_phase_avg"] / np.maximum(out["rpm"], 1.0) * 1000.0

    # Interactions age x signaux : une machine vieillissante qui présente
    # en plus une anomalie de vibration ou de température est dans une
    # situation combinée plus critique que la somme des deux signaux seuls.
    # Division par 1000 pour rester dans une plage numérique raisonnable.
    out["age_x_vibration"] = out["hours_since_maintenance"] * out["vibration_rms"]
    out["age_x_temperature"] = out["hours_since_maintenance"] * out["temperature_motor"] / 1000.0

    # Deviation absolue a la pression nominale du mode opératoire :
    # une pression trop basse (fuite) ou trop haute (blocage) est
    # également dangereuse. On prend la valeur absolue pour traiter les
    # deux cas symétriquement. fillna(35.0) = pression nominale "normal"
    # en cas de mode opératoire inconnu (robustesse à l'inférence).
    nominal = out["operating_mode"].map(NOMINAL_PRESSURE).fillna(35.0)
    out["pressure_deviation"] = (out["pressure_level"] - nominal).abs()

    return out


# Liste des features derivees · utilisee pour etendre la liste des
# colonnes numeriques traitees par le ColumnTransformer.
ENGINEERED_NUMERIC_FEATURES: list[str] = [
    "vibration_per_rpm",
    "temp_above_ambient",
    "current_per_rpm",
    "age_x_vibration",
    "age_x_temperature",
    "pressure_deviation",
]
