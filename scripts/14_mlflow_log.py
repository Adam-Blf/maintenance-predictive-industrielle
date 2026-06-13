# -*- coding: utf-8 -*-
"""Script 14 · Traçabilité MLflow des modèles (BONUS industrialisation).

À QUOI ÇA SERT ?
----------------
En production ML, on doit tracer QUEL modèle, avec QUELLES métriques, a
été promu. MLflow journalise chaque run (paramètres, métriques, artefact
modèle) dans un registre local consultable via `mlflow ui`.

Ce script NE RÉ-ENTRAÎNE RIEN · il charge les modèles déjà sauvegardés,
recalcule leurs métriques sur le jeu de test (lecture seule) et les
journalise dans MLflow. Le modèle final est tagué `promoted`.

USAGE
-----
    python scripts/14_mlflow_log.py
    mlflow ui   # puis http://localhost:5000 pour explorer
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import mlflow
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)

from src.config import DATA_PROCESSED_DIR, MODELS_DIR

MODELS = {
    "logistic_regression": "logistic_regression.joblib",
    "random_forest": "random_forest.joblib",
    "xgboost": "xgboost.joblib",
    "mlp": "mlp.joblib",
}


def main() -> None:
    final_name = (MODELS_DIR / "final_model_name.txt").read_text(encoding="utf-8").strip()
    X = pd.read_csv(DATA_PROCESSED_DIR / "X_test.csv")
    y = pd.read_csv(DATA_PROCESSED_DIR / "y_test.csv").iloc[:, 0]

    mlflow.set_tracking_uri((PROJECT_ROOT / "mlruns").as_uri())
    mlflow.set_experiment("maintenance-predictive-binaire")

    for name, fname in MODELS.items():
        path = MODELS_DIR / fname
        if not path.exists():
            print(f"  (ignoré · {fname} absent)")
            continue
        model = joblib.load(path)
        proba = model.predict_proba(X)[:, 1]
        pred = (proba >= 0.5).astype(int)
        metrics = {
            "f1": f1_score(y, pred),
            "roc_auc": roc_auc_score(y, proba),
            "pr_auc": average_precision_score(y, proba),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred),
        }
        with mlflow.start_run(run_name=name):
            mlflow.log_param("model_type", name)
            mlflow.log_param("is_final", name == final_name)
            mlflow.log_metrics(metrics)
            if name == final_name:
                mlflow.set_tag("stage", "promoted")
            try:
                mlflow.sklearn.log_model(model, name="model")
            except Exception:
                pass  # le log d'artefact ne doit pas bloquer la traçabilité des métriques
        print(f"  [MLflow] {name:20} F1={metrics['f1']:.3f} ROC-AUC={metrics['roc_auc']:.3f}"
              + ("  <- promu" if name == final_name else ""))

    print(f"[MLflow] runs écrits dans {PROJECT_ROOT / 'mlruns'} · `mlflow ui` pour explorer")


if __name__ == "__main__":
    main()
