# -*- coding: utf-8 -*-
"""Tests unitaires · ColumnTransformer + feature engineering."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import generate_synthetic_dataset  # noqa: E402
from src.feature_engineering import (  # noqa: E402
    ENGINEERED_NUMERIC_FEATURES,
    add_engineered_features,
)
from src.preprocessing import build_preprocessor, get_feature_names  # noqa: E402


@pytest.fixture(scope="module")
def small_df() -> pd.DataFrame:
    """Petit dataset pour tests rapides."""
    return generate_synthetic_dataset(n_samples=500, seed=7)


def test_preprocessor_fit_transform_shape(small_df: pd.DataFrame) -> None:
    """Le ColumnTransformer doit produire une matrice 2D dense."""
    pre = build_preprocessor()
    from src.config import ALL_FEATURES

    X = small_df[ALL_FEATURES]
    out = pre.fit_transform(X)
    assert out.ndim == 2
    assert out.shape[0] == len(small_df)
    # 10 num + 4 modes OHE = 14 colonnes (handle_unknown=ignore).
    assert out.shape[1] >= 13


def test_preprocessor_no_nan_after_imputation(small_df: pd.DataFrame) -> None:
    """L'imputation median + most_frequent doit eliminer tous les NaN."""
    pre = build_preprocessor()
    from src.config import ALL_FEATURES

    X = small_df[ALL_FEATURES].copy()
    # Injection de NaN aléatoires.
    rng = np.random.default_rng(0)
    mask = rng.random(X.shape) < 0.05
    X = X.mask(mask)
    out = pre.fit_transform(X)
    assert not np.isnan(out).any()


def test_get_feature_names(small_df: pd.DataFrame) -> None:
    """get_feature_names retourne une liste non-vide post-fit."""
    pre = build_preprocessor()
    from src.config import ALL_FEATURES

    pre.fit(small_df[ALL_FEATURES])
    names = get_feature_names(pre)
    assert isinstance(names, list)
    assert len(names) >= 13
    assert "vibration_rms" in names


def test_engineered_features_no_inf(small_df: pd.DataFrame) -> None:
    """Les features dérivées ne contiennent ni inf ni nan."""
    out = add_engineered_features(small_df)
    for f in ENGINEERED_NUMERIC_FEATURES:
        assert f in out.columns
        assert not out[f].isna().any()
        assert np.isfinite(out[f]).all()


def test_engineered_features_idempotent(small_df: pd.DataFrame) -> None:
    """Appel multiple ne change pas le résultat."""
    a = add_engineered_features(small_df)
    b = add_engineered_features(a)
    for f in ENGINEERED_NUMERIC_FEATURES:
        pd.testing.assert_series_equal(a[f], b[f])
