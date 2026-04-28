# -*- coding: utf-8 -*-
"""Tests unitaires · modèles (binaire, multi-classe, régression)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ALL_FEATURES, DATASET_PATH, TARGET_BINARY, TARGET_REGRESSION  # noqa: E402
from src.data_loader import load_dataset  # noqa: E402
from src.models import MODEL_CATALOG, get_model  # noqa: E402
from src.models_multiclass import (  # noqa: E402
    build_logistic_multiclass,
    build_rf_multiclass,
)
from src.models_regression import build_rf_regressor, build_ridge  # noqa: E402


@pytest.fixture(scope="module")
def small_df() -> pd.DataFrame:
    """Échantillon stratifié 600 lignes du dataset Kaggle officiel.

    On utilise UNIQUEMENT le dataset réel (politique projet · pas de
    génération synthétique). Si le CSV n'est pas téléchargé, les tests
    qui dépendent de cette fixture sont automatiquement skippés.
    """
    if not DATASET_PATH.exists():
        pytest.skip(f"Dataset Kaggle absent · {DATASET_PATH}")
    df = load_dataset()
    # Échantillonnage stratifié sur la cible binaire pour garder le ratio panne/sain.
    pos = df[df[TARGET_BINARY] == 1].sample(n=min(80, (df[TARGET_BINARY] == 1).sum()), random_state=11)
    neg = df[df[TARGET_BINARY] == 0].sample(n=min(520, (df[TARGET_BINARY] == 0).sum()), random_state=11)
    return pd.concat([pos, neg], ignore_index=True).sample(frac=1.0, random_state=11).reset_index(drop=True)


@pytest.mark.parametrize("name", ["logistic_regression", "random_forest", "xgboost", "mlp"])
def test_binary_model_pipeline_runs(name: str, small_df: pd.DataFrame) -> None:
    """Chaque modèle binaire fit + predict_proba sur un petit échantillon."""
    X = small_df[ALL_FEATURES]
    y = small_df[TARGET_BINARY]
    if name == "xgboost":
        # XGBoost a besoin de scale_pos_weight.
        ratio = float((y == 0).sum() / max((y == 1).sum(), 1))
        pipe = get_model(name, scale_pos_weight=ratio)
    else:
        pipe = get_model(name)
    pipe.fit(X, y)
    proba = pipe.predict_proba(X)[:, 1]
    assert proba.shape == (len(y),)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_multiclass_models(small_df: pd.DataFrame) -> None:
    """Les modèles multi-classe entraînent et prédisent correctement."""
    df = small_df[small_df["failure_type"].notna() & (small_df["failure_type"].astype(str) != "None")]
    if df["failure_type"].nunique() < 2:
        pytest.skip("Pas assez de classes dans l'échantillon")
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    y = le.fit_transform(df["failure_type"])
    X = df[ALL_FEATURES]

    for builder in (build_logistic_multiclass, build_rf_multiclass):
        pipe = builder()
        pipe.fit(X, pd.Series(y))
        pred = pipe.predict(X)
        assert pred.shape == (len(y),)
        assert set(pred.tolist()).issubset(set(np.unique(y).tolist()))


def test_regression_models(small_df: pd.DataFrame) -> None:
    """Les modèles de régression produisent des prédictions float."""
    X = small_df[ALL_FEATURES]
    y = small_df[TARGET_REGRESSION].astype(float)
    for builder in (build_ridge, build_rf_regressor):
        pipe = builder()
        pipe.fit(X, y)
        pred = pipe.predict(X)
        assert pred.shape == (len(y),)
        assert pred.dtype.kind == "f"


def test_get_model_invalid_name() -> None:
    """Une clé invalide lève KeyError clair."""
    with pytest.raises(KeyError):
        get_model("xxx-not-a-model")
