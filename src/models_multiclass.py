# -*- coding: utf-8 -*-
"""Modeles de classification multi-classe sur `failure_type`.

Tâche bonus du sujet · identifier le type precis de panne parmi
{Mechanical, Thermal, Electrical, Hydraulic} (4 classes positives,
on filtre les 'None'). Permet a l'equipe maintenance de mobiliser la
bonne pièce détachée et la bonne competence avant l'intervention.
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from .config import RANDOM_STATE
from .preprocessing import build_preprocessor


def build_logistic_multiclass() -> Pipeline:
    """Régression logistique multi-classe (One-vs-Rest implicite)."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1500,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def build_rf_multiclass() -> Pipeline:
    """Random Forest · adapté multi-classe nativement."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=250,
                    max_depth=None,
                    min_samples_leaf=4,
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def build_xgb_multiclass(num_class: int) -> Pipeline:
    """XGBoost multi-classe avec softmax."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                XGBClassifier(
                    n_estimators=300,
                    learning_rate=0.07,
                    max_depth=6,
                    objective="multi:softprob",
                    num_class=num_class,
                    eval_metric="mlogloss",
                    subsample=0.85,
                    colsample_bytree=0.85,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    tree_method="hist",
                ),
            ),
        ]
    )


def build_mlp_multiclass() -> Pipeline:
    """MLP multi-classe · architecture identique à la classification binaire."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                MLPClassifier(
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
