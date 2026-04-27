# -*- coding: utf-8 -*-
"""Tests d'intégration · API FastAPI via httpx TestClient.

Skip automatique si le modèle final n'est pas entraîné (CI sans
artefacts ML pré-générés).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODELS_DIR  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (MODELS_DIR / "final_model.joblib").exists(),
    reason="Modèle non entraîné (lancer scripts/03_train_models.py)",
)


@pytest.fixture(scope="module")
def client():
    """TestClient FastAPI réutilisé pour toutes les requêtes."""
    from fastapi.testclient import TestClient

    from api.main import app

    with TestClient(app) as c:
        yield c


def test_health(client) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    payload = r.json()
    assert payload["status"] in {"healthy", "degraded"}
    assert "api_version" in payload


def test_model_info(client) -> None:
    r = client.get("/model-info")
    assert r.status_code in (200, 503)
    if r.status_code == 200:
        payload = r.json()
        assert "model_name" in payload
        assert "features_required" in payload
        assert isinstance(payload["features_required"], list)


def test_predict_valid(client) -> None:
    payload = {
        "vibration_rms": 4.2,
        "temperature_motor": 78.5,
        "current_phase_avg": 12.3,
        "pressure_level": 58.7,
        "rpm": 2100,
        "hours_since_maintenance": 320.0,
        "ambient_temp": 24.1,
        "operating_mode": "peak",
        "machine_type": "CNC",
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    res = r.json()
    assert res["failure_within_24h"] in (0, 1)
    assert 0.0 <= res["probability"] <= 1.0
    assert res["risk_level"] in {"low", "moderate", "high"}


def test_predict_invalid_mode(client) -> None:
    """Mode opératoire inconnu → 422 Pydantic."""
    payload = {
        "vibration_rms": 4.2,
        "temperature_motor": 78.5,
        "current_phase_avg": 12.3,
        "pressure_level": 58.7,
        "rpm": 2100,
        "hours_since_maintenance": 320.0,
        "ambient_temp": 24.1,
        "operating_mode": "Turbo",  # invalide
        "machine_type": "CNC",
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_predict_out_of_range(client) -> None:
    """Valeur capteur hors plage → 422 Pydantic."""
    payload = {
        "vibration_rms": -50.0,  # négatif (hors ge=0)
        "temperature_motor": 78.5,
        "current_phase_avg": 12.3,
        "pressure_level": 58.7,
        "rpm": 2100,
        "hours_since_maintenance": 320.0,
        "ambient_temp": 24.1,
        "operating_mode": "normal",
        "machine_type": "CNC",
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 422
