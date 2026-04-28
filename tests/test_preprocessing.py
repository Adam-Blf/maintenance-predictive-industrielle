# -*- coding: utf-8 -*-
"""Tests unitaires · ColumnTransformer (imputation + scaling + OHE)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ALL_FEATURES, DATASET_PATH  # noqa: E402
from src.data_loader import load_dataset  # noqa: E402
from src.preprocessing import build_preprocessor, get_feature_names  # noqa: E402


@pytest.fixture(scope="module")
def small_df() -> pd.DataFrame:
    """Échantillon 500 lignes du dataset Kaggle officiel.

    Politique projet · pas de génération synthétique. Si le CSV est absent
    les tests dépendants sont skippés.
    """
    if not DATASET_PATH.exists():
        pytest.skip(f"Dataset Kaggle absent · {DATASET_PATH}")
    return load_dataset().sample(n=500, random_state=7).reset_index(drop=True)


def test_preprocessor_fit_transform_shape(small_df: pd.DataFrame) -> None:
    """Le ColumnTransformer doit produire une matrice 2D dense."""
    pre = build_preprocessor()
    X = small_df[ALL_FEATURES]
    out = pre.fit_transform(X)
    assert out.ndim == 2
    assert out.shape[0] == len(small_df)
    # 7 numériques + 3 modes OHE + 4 types OHE = 14 colonnes
    # (handle_unknown=ignore garantit qu'on ne plante pas si une modalité
    # absente du fit apparaît au transform).
    assert out.shape[1] >= 13


def test_preprocessor_no_nan_after_imputation(small_df: pd.DataFrame) -> None:
    """L'imputation median + most_frequent doit eliminer tous les NaN."""
    pre = build_preprocessor()
    X = small_df[ALL_FEATURES].copy()
    # Injection de NaN aléatoires pour simuler des capteurs en panne.
    rng = np.random.default_rng(0)
    mask = rng.random(X.shape) < 0.05
    X = X.mask(mask)
    out = pre.fit_transform(X)
    assert not np.isnan(out).any()


def test_get_feature_names(small_df: pd.DataFrame) -> None:
    """get_feature_names retourne une liste non-vide post-fit."""
    pre = build_preprocessor()
    pre.fit(small_df[ALL_FEATURES])
    names = get_feature_names(pre)
    assert isinstance(names, list)
    assert len(names) >= 13
    assert "vibration_rms" in names
