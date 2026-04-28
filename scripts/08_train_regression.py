# -*- coding: utf-8 -*-
"""Script · entraînement de la régression sur rul_hours (Remaining Useful Life).

Rôle dans le pipeline
----------------------
Script n°8, indépendant du script 03 (binaire). Peut être exécuté
après le script 01 uniquement. Tâche bonus du sujet.

Entrées
-------
data/raw/predictive_maintenance_v3.csv
    Dataset complet (24 042 lignes, colonne rul_hours continue [0.5, 99]).

Sorties
-------
models/regression_{model}.joblib (x4)
    Pipelines Ridge, RF, XGBoost, MLP entraînés.
models/regression_final.joblib
    Copie du meilleur modèle (critère : R² le plus élevé sur test set).
models/regression_final_name.txt
    Nom textuel du meilleur modèle.
reports/metrics_regression.csv / .json
    Tableau comparatif MAE (h), RMSE (h), R², temps d'entraînement.
reports/figures/regression_pred_vs_true.png
    Scatter plot prédictions vs valeurs réelles du meilleur modèle.

Pré-requis
----------
Script 01 exécuté (dataset disponible dans data/raw/).

Lien cahier des charges
-----------------------
Régression RUL demandée comme tâche bonus dans le cahier des charges.
Le MAE en heures est la métrique la plus interprétable métier
("en moyenne, on se trompe de X heures sur la durée de vie restante").

Usage ·
    python scripts/08_train_regression.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt  # noqa: E402

from src.config import (  # noqa: E402
    ALL_FEATURES,
    COLOR_EFREI_BLUE,
    COLOR_EFREI_DARK,
    MODELS_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    REPORTS_FIGURES_DIR,
    TARGET_REGRESSION,
    TEST_SIZE,
    ensure_directories,
)
from src.data_loader import load_dataset  # noqa: E402
from src.models_regression import (  # noqa: E402
    build_mlp_regressor,
    build_rf_regressor,
    build_ridge,
    build_xgb_regressor,
)


def main() -> None:
    """Orchestre l'entraînement, l'évaluation et la persistance des 4 modèles de régression."""
    ensure_directories()
    print("[REGRESSION] Chargement dataset...")
    df = load_dataset()
    X = df[ALL_FEATURES].copy()
    y = df[TARGET_REGRESSION].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    factories = {
        "ridge": build_ridge,
        "random_forest": build_rf_regressor,
        "xgboost": build_xgb_regressor,
        "mlp": build_mlp_regressor,
    }

    results = []
    pred_data = {}
    for name, factory in factories.items():
        print(f"\n[TRAIN] {name}...")
        pipeline = factory()
        t0 = time.time()
        pipeline.fit(X_train, y_train)
        fit_time = time.time() - t0
        y_pred = pipeline.predict(X_test)

        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = float(r2_score(y_test, y_pred))
        print(f"  MAE={mae:.2f}h  RMSE={rmse:.2f}h  R²={r2:.4f}  fit={fit_time:.2f}s")

        joblib.dump(pipeline, MODELS_DIR / f"regression_{name}.joblib", compress=3)
        results.append(
            {
                "model_name": name,
                "mae_hours": mae,
                "rmse_hours": rmse,
                "r2": r2,
                "fit_time_s": fit_time,
            }
        )
        pred_data[name] = y_pred

    df_results = pd.DataFrame(results)
    best_idx = df_results["r2"].idxmax()
    best_name = df_results.loc[best_idx, "model_name"]
    print(f"\n[FINAL] Régression · meilleur modèle = {best_name}")
    print(df_results.to_string(index=False))

    df_results.to_csv(REPORTS_DIR / "metrics_regression.csv", index=False)
    df_results.to_json(REPORTS_DIR / "metrics_regression.json", orient="records", indent=2)

    # Scatter prédiction vs vérité pour le best model.
    fig, ax = plt.subplots(figsize=(8, 7))
    y_pred_best = pred_data[best_name]
    sample_idx = np.random.RandomState(RANDOM_STATE).choice(
        len(y_test), size=min(2500, len(y_test)), replace=False
    )
    ax.scatter(
        y_test.values[sample_idx],
        y_pred_best[sample_idx],
        alpha=0.4,
        s=14,
        color=COLOR_EFREI_BLUE,
        edgecolors="none",
    )
    lo, hi = float(y.min()), float(y.max())
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, label="Prédiction parfaite")
    ax.set_xlabel("RUL réel (heures)", fontsize=11)
    ax.set_ylabel("RUL prédit (heures)", fontsize=11)
    ax.set_title(
        f"Régression RUL · {best_name} (R² = {df_results.loc[best_idx, 'r2']:.3f})",
        fontsize=13,
        fontweight="bold",
        color=COLOR_EFREI_DARK,
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(REPORTS_FIGURES_DIR / "regression_pred_vs_true.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Persistance · meilleur modèle régression.
    joblib.dump(
        joblib.load(MODELS_DIR / f"regression_{best_name}.joblib"),
        MODELS_DIR / "regression_final.joblib",
        compress=3,
    )
    (MODELS_DIR / "regression_final_name.txt").write_text(best_name, encoding="utf-8")
    print("Done.")


if __name__ == "__main__":
    main()
