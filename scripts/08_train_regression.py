# -*- coding: utf-8 -*-
"""Script 08 · Régression sur le RUL (Remaining Useful Life) · BONUS.

À QUOI ÇA SERT ?
----------------
Encore plus précis qu'une simple alerte oui/non · ce script prédit COMBIEN
D'HEURES il reste avant la prochaine panne. Cible continue `rul_hours` ∈
[0.5h, 99h]. C'est de la **régression**, pas de la classification.

DIFFÉRENCE AVEC LA CLASSIFICATION
---------------------------------
- **Cible** · valeur continue (12.5h, 47.3h, 0.8h, ...) au lieu d'une
  classe (0 ou 1).
- **Modèles** · Ridge (au lieu de LogReg) · les autres restent (RF, XGB, MLP).
- **Métriques** ·
    - **MAE (Mean Absolute Error)** · en heures · LA métrique métier ·
      "en moyenne, on se trompe de X heures sur la durée de vie restante".
    - **RMSE** · pénalise plus les grosses erreurs que MAE (utile si une
      erreur de 50h est bien pire qu'une erreur de 5h).
    - **R²** · pourcentage de variance expliquée. R²=0.65 = "le modèle
      capture 65% de la variabilité réelle du RUL".
- **Pas de stratify** · on ne stratifie pas en régression (`stratify=` ne
  marche que sur des classes discrètes).

INTÉRÊT MÉTIER
--------------
Le responsable maintenance peut planifier les interventions ·
  - "Cette pompe a 40h restantes · on la programme jeudi prochain."
  - "Ce robot a 4h restantes · stop urgent."
Plus actionnable qu'un simple flag binaire panne/sain.

CE QUI EST ENREGISTRÉ
---------------------
  - models/regression_{4 modèles}.joblib · 4 pipelines régression
  - models/regression_final.joblib · le meilleur R² (typiquement RF)
  - reports/08/metrics_regression.{csv,json} · MAE/RMSE/R² par modèle
  - reports/08/regression_pred_vs_true.png · scatter `prédit vs réel`,
    une ligne diagonale = prédiction parfaite

USAGE
-----
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

# Bootstrap · auto-install des dépendances manquantes (rend le repo
# clonable et exécutable sur n'importe quelle machine sans setup manuel).
from src.bootstrap import ensure_dependencies  # noqa: E402
ensure_dependencies()

import matplotlib.pyplot as plt  # noqa: E402

from src.config import (  # noqa: E402
    ALL_FEATURES,
    COLOR_EFREI_BLUE,
    COLOR_EFREI_DARK,
    MODELS_DIR,
    RANDOM_STATE,
    S08_DIR,
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

    df_results.to_csv(S08_DIR / "metrics_regression.csv", index=False)
    df_results.to_json(S08_DIR / "metrics_regression.json", orient="records", indent=2)

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
    fig.savefig(S08_DIR / "regression_pred_vs_true.png", dpi=150, bbox_inches="tight")
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
