# -*- coding: utf-8 -*-
"""Tests de fumée · vérifient que l'ossature du projet est saine.

Ces tests sont volontairement minimaux · ils valident l'import des modules
et la cohérence du dataset généré, sans entrer dans le détail des
performances. L'évaluation des modèles se fait via la pipeline complète.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_imports() -> None:
    """Tous les modules src doivent s'importer sans erreur."""
    # Import groupe pour rester rapide.
    from src import (  # noqa: F401
        config,
        data_loader,
        diagrams,
        evaluation,
        interpretability,
        models,
        preprocessing,
        report,
    )


def test_config_paths() -> None:
    """Les chemins critiques sont des Path absolus."""
    from src.config import (
        DATA_RAW_DIR,
        EFREI_LOGO,
        MODELS_DIR,
    )
    from src.config import PROJECT_ROOT as ROOT

    assert ROOT.is_absolute()
    assert DATA_RAW_DIR.parent.parent == ROOT
    assert MODELS_DIR.parent == ROOT
    assert "logo_efrei" in EFREI_LOGO.name


def test_real_dataset_shape() -> None:
    """Le CSV Kaggle officiel respecte le schéma attendu (15 colonnes v3.0).

    Politique projet · pas de génération synthétique. Si le CSV n'a pas
    été téléchargé dans data/raw/, le test est skippé proprement.
    """
    import pytest

    from src.config import DATASET_PATH
    from src.data_loader import load_dataset

    if not DATASET_PATH.exists():
        pytest.skip(f"Dataset Kaggle absent · {DATASET_PATH}")

    df = load_dataset()
    # 15 colonnes Kaggle v3.0 (7 num + 2 cat + ts + id + 4 cibles).
    assert len(df.columns) == 15
    assert len(df) > 0
    assert df["failure_within_24h"].isin([0, 1]).all()
    assert (df["rul_hours"] >= 0).all()
    assert df["operating_mode"].isin(["normal", "idle", "peak"]).all()
    assert df["machine_type"].isin(["CNC", "Pump", "Compressor", "Robotic Arm"]).all()


def test_preprocessor_builds() -> None:
    """Le ColumnTransformer s'instancie sans erreur."""
    from src.preprocessing import build_preprocessor

    pre = build_preprocessor()
    assert pre is not None
    # Doit avoir 2 transformers (num + cat).
    assert len(pre.transformers) == 2


def test_models_factory() -> None:
    """Les 4 modèles sont disponibles dans le catalogue."""
    from src.models import MODEL_CATALOG, get_model

    assert set(MODEL_CATALOG.keys()) == {
        "logistic_regression",
        "random_forest",
        "xgboost",
        "mlp",
    }
    for name in MODEL_CATALOG:
        pipeline = get_model(name) if name != "xgboost" else get_model(name, scale_pos_weight=1.0)
        assert pipeline is not None
