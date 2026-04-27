# -*- coding: utf-8 -*-
"""Modeles de regression sur `rul_hours` (Remaining Useful Life).

Tâche bonus du sujet · estimer le nombre d'heures restantes avant
defaillance. Utilite metier · planifier la maintenance preventive
en optimisant l'utilisation des creneaux d'arret programme.
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from .config import RANDOM_STATE
from .preprocessing import build_preprocessor


def build_ridge() -> Pipeline:
    """Ridge regression · baseline lineaire regularisee L2."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("regressor", Ridge(alpha=1.0, random_state=RANDOM_STATE)),
        ]
    )


def build_rf_regressor() -> Pipeline:
    """Random Forest regressor · capture les non-linéarités."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=250,
                    max_depth=None,
                    min_samples_leaf=5,
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def build_xgb_regressor() -> Pipeline:
    """Gradient Boosting regression · etat de l'art tabulaire."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "regressor",
                XGBRegressor(
                    n_estimators=400,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    objective="reg:squarederror",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    tree_method="hist",
                ),
            ),
        ]
    )


def build_mlp_regressor() -> Pipeline:
    """MLP regressor · le DL exige par le sujet, version regression."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "regressor",
                MLPRegressor(
                    hidden_layer_sizes=(64, 32, 16),
                    activation="relu",
                    solver="adam",
                    alpha=1e-3,
                    batch_size=256,
                    max_iter=200,
                    early_stopping=True,
                    n_iter_no_change=12,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
