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


def test_synthetic_dataset_shape() -> None:
    """Le générateur synthétique respecte les specs du sujet."""
    from src.data_loader import generate_synthetic_dataset

    df = generate_synthetic_dataset(n_samples=1000, seed=42)
    # 16 colonnes (10 num + 1 cat + 1 ts + 1 id + 3 cibles).
    assert len(df.columns) == 16
    assert len(df) == 1000
    assert df["failure_within_24h"].isin([0, 1]).all()
    assert (df["rul_hours"] >= 0).all()
    assert df["operating_mode"].isin(["Normal", "HighLoad", "Idle", "Maintenance"]).all()


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
