# -*- coding: utf-8 -*-
"""API REST FastAPI · service d'inférence du modèle final.

Endpoints exposés · (cf. cahier des charges, EF5)

  - `POST /predict`    · reçoit les features capteurs, renvoie la prédiction
                          binaire + la probabilité.
  - `GET  /health`     · vérifie que le service tourne et que le modèle est
                          chargé.
  - `GET  /model-info` · métadonnées du modèle servi (nom, métriques, date).

Validation · Pydantic v2 · contrôle automatique des types et des plages.
Documentation · Swagger UI servi automatiquement sur `/docs` (gratuit
avec FastAPI).

Lancement ·
    uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Permet d'importer `src` en relatif · l'API est lancée depuis la racine
# projet via uvicorn `api.main:app`.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import __version__  # noqa: E402
from src.config import ALL_FEATURES, MODELS_DIR, OPERATING_MODES, REPORTS_DIR  # noqa: E402

# ---------------------------------------------------------------------------
# Initialisation de l'application + métadonnées exposées dans Swagger.
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Maintenance Prédictive · API d'inférence",
    description=(
        "Service REST exposant le modèle final candidat pour la prédiction "
        "de panne machine dans les 24 heures.\n\n"
        "**Auteurs** · Adam Beloucif & Emilien Morice\n"
        "**Cours** · M1 Mastère Data Engineering & IA · BC2 RNCP40875\n"
        "**École** · EFREI Paris Panthéon-Assas Université"
    ),
    version=__version__,
    contact={
        "name": "Adam Beloucif",
        "email": "adam.beloucif@efrei.net",
    },
)

# CORS permissif · dashboard Streamlit local + outils de test (Postman, curl).
# En production, restreindre `allow_origins` à l'URL exacte du dashboard.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schémas Pydantic · contrat d'entrée/sortie typé et validé.
# ---------------------------------------------------------------------------
class SensorReading(BaseModel):
    """Mesures capteurs envoyées en entrée par le client.

    Les bornes (`ge` / `le`) sont alignées sur les distributions observées
    dans le dataset · une valeur hors plage déclenche une 422 automatique
    sans atteindre le modèle.
    """

    vibration_rms: float = Field(..., ge=0.0, le=15.0, description="Vibration RMS (mm/s)")
    temperature_motor: float = Field(..., ge=-20.0, le=160.0, description="Température moteur (°C)")
    current_phase_avg: float = Field(..., ge=0.0, le=40.0, description="Courant phase moyen (A)")
    pressure_level: float = Field(..., ge=0.0, le=120.0, description="Pression (bar)")
    rpm: float = Field(..., ge=0.0, le=5000.0, description="Vitesse de rotation (tr/min)")
    hours_since_maintenance: float = Field(
        ..., ge=0.0, le=3000.0, description="Heures depuis dernière maintenance"
    )
    ambient_temp: float = Field(..., ge=-20.0, le=60.0, description="Température ambiante (°C)")
    operating_mode: Literal["normal", "idle", "peak"] = Field(
        ..., description="Mode opératoire"
    )
    machine_type: Literal["CNC", "Pump", "Compressor", "Robotic Arm"] = Field(
        ..., description="Type de machine industrielle"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


class PredictionResponse(BaseModel):
    """Réponse renvoyée par `/predict`."""

    failure_within_24h: int = Field(..., description="Prédiction binaire (0 = OK, 1 = panne)")
    probability: float = Field(..., ge=0.0, le=1.0, description="Probabilité de panne")
    risk_level: Literal["low", "moderate", "high"] = Field(
        ..., description="Niveau de risque (seuils 30% / 60%)"
    )
    recommendation: str = Field(..., description="Action recommandée pour l'opérateur")
    model_name: str = Field(..., description="Identifiant du modèle utilisé")
    timestamp_utc: str = Field(..., description="Horodatage UTC ISO 8601")


class HealthResponse(BaseModel):
    """Réponse renvoyée par `/health`."""

    status: Literal["healthy", "degraded"]
    model_loaded: bool
    api_version: str
    timestamp_utc: str


class ModelInfoResponse(BaseModel):
    """Réponse renvoyée par `/model-info`."""

    model_name: str
    api_version: str
    metrics: dict
    features_required: list[str]
    operating_modes: list[str]


# ---------------------------------------------------------------------------
# Chargement paresseux du modèle · au premier import du module FastAPI.
# Si le modèle n'existe pas encore, on lève une exception claire qui aide
# l'utilisateur à comprendre l'ordre d'exécution des scripts.
# ---------------------------------------------------------------------------
MODEL_PATH = MODELS_DIR / "final_model.joblib"
NAME_PATH = MODELS_DIR / "final_model_name.txt"
METRICS_PATH = REPORTS_DIR / "metrics_summary.json"

_model = None
_model_name = "unknown"
_model_metrics: dict = {}


def _load_model_lazy() -> None:
    """Charge le modèle si pas déjà fait. Idempotent."""
    global _model, _model_name, _model_metrics  # noqa: PLW0603
    if _model is not None:
        return
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Modèle introuvable · {MODEL_PATH}. " "Lancer `python scripts/03_train_models.py`."
        )
    _model = joblib.load(MODEL_PATH)
    if NAME_PATH.exists():
        _model_name = NAME_PATH.read_text(encoding="utf-8").strip()
    if METRICS_PATH.exists():
        all_metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        for m in all_metrics:
            if m["model_name"] == _model_name:
                _model_metrics = m
                break


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["monitoring"])
def health() -> HealthResponse:
    """Endpoint de santé · vérifie que le service répond et que le modèle
    est correctement chargé en mémoire."""
    try:
        _load_model_lazy()
        loaded = _model is not None
    except Exception:
        loaded = False
    return HealthResponse(
        status="healthy" if loaded else "degraded",
        model_loaded=loaded,
        api_version=__version__,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["monitoring"])
def model_info() -> ModelInfoResponse:
    """Métadonnées du modèle servi · utile pour les outils de monitoring
    et les vérifications d'intégrité côté dashboard."""
    try:
        _load_model_lazy()
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    return ModelInfoResponse(
        model_name=_model_name,
        api_version=__version__,
        metrics=_model_metrics or {},
        features_required=ALL_FEATURES,
        operating_modes=OPERATING_MODES,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(payload: SensorReading) -> PredictionResponse:
    """Prédiction de panne dans les 24 heures à partir des mesures capteurs.

    Le pipeline complet (preprocessing + modèle) est appliqué côté serveur
    pour garantir l'identité bit-à-bit avec le notebook d'entraînement
    et éliminer tout risque de divergence d'implémentation côté client.
    """
    try:
        _load_model_lazy()
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    # Construction du DataFrame · l'ordre des colonnes doit matcher
    # `ALL_FEATURES` (contrat du préprocesseur).
    record = payload.model_dump()
    X = pd.DataFrame([{f: record[f] for f in ALL_FEATURES}])

    try:
        proba = float(_model.predict_proba(X)[0, 1])
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur d'inférence · {exc}",
        ) from exc

    prediction = int(proba >= 0.5)

    # Catégorisation du risque · seuils alignés avec le dashboard.
    if proba < 0.30:
        risk = "low"
        reco = "Aucune action requise. Surveillance continue."
    elif proba < 0.60:
        risk = "moderate"
        reco = "Programmer un contrôle visuel sous 48h."
    else:
        risk = "high"
        reco = "Planifier une intervention préventive dans les 12-24h."

    return PredictionResponse(
        failure_within_24h=prediction,
        probability=round(proba, 4),
        risk_level=risk,
        recommendation=reco,
        model_name=_model_name,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/", tags=["root"])
def root() -> dict:
    """Page d'accueil · redirige vers la doc Swagger."""
    return {
        "message": "Maintenance Prédictive · API d'inférence",
        "version": __version__,
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
    }
