# -*- coding: utf-8 -*-
"""Tests unitaires · module d'évaluation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import (  # noqa: E402
    ClassificationMetrics,
    compute_classification_metrics,
)


def test_perfect_classifier() -> None:
    """Un classifieur parfait doit avoir toutes les métriques à 1."""
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = y_true.copy()
    y_proba = y_true.astype(float)
    m = compute_classification_metrics("perfect", y_true, y_pred, y_proba)
    assert m.accuracy == 1.0
    assert m.precision == 1.0
    assert m.recall == 1.0
    assert m.f1 == 1.0
    assert m.roc_auc == 1.0


def test_random_classifier_low_metrics() -> None:
    """Un classifieur aléatoire constant doit avoir un recall = 0 sur la classe 1."""
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_pred = np.zeros_like(y_true)  # toujours 0
    y_proba = np.full_like(y_true, 0.1, dtype=float)
    m = compute_classification_metrics("zero", y_true, y_pred, y_proba)
    assert m.recall == 0.0
    assert m.f1 == 0.0


def test_to_dict_keys() -> None:
    """to_dict retourne toutes les clés attendues."""
    m = ClassificationMetrics(
        model_name="x",
        accuracy=0.9,
        precision=0.8,
        recall=0.7,
        f1=0.75,
        roc_auc=0.92,
        pr_auc=0.6,
        fit_time_s=1.2,
        predict_time_ms=0.04,
    )
    d = m.to_dict()
    assert {"model_name", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"}.issubset(
        d.keys()
    )
