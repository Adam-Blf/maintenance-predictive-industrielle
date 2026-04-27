# -*- coding: utf-8 -*-
"""Script · hyperparameter tuning Optuna (RF, XGB, MLP).

Recherche bayesienne TPE pour ne pas brûler du compute en grid
exhaustif. Le sous-échantillon (8000 lignes) est suffisant pour
identifier des bonnes regions de l'espace d'hyperparametres.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    ALL_FEATURES,
    DATA_PROCESSED_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    TARGET_BINARY,
    ensure_directories,
)
from src.data_loader import load_dataset  # noqa: E402
from src.tuning import tune_all  # noqa: E402


def main() -> None:
    ensure_directories()

    print("[TUNING] Chargement dataset...")
    df = load_dataset()
    X = df[ALL_FEATURES].copy()
    y = df[TARGET_BINARY].copy()

    # Sous-echantillon stratifie pour rester rapide (compromis qualite/temps).
    rng = np.random.RandomState(RANDOM_STATE)
    n_sample = min(8000, len(X))
    idx = rng.choice(len(X), size=n_sample, replace=False)
    X_sub = X.iloc[idx].reset_index(drop=True)
    y_sub = y.iloc[idx].reset_index(drop=True)
    print(f"  sous-echantillon · {len(X_sub):,} lignes (stratification implicite)")

    print("[TUNING] Lancement Optuna · ~3-5 minutes total...")
    results = tune_all(X_sub, y_sub, n_trials_each=20)

    output = REPORTS_DIR / "tuning_results.json"
    output.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\n[TUNING] Resultats sauvegardes dans {output}")

    # Resume console.
    print("\n=== Resume ===")
    for name, res in results.items():
        print(f"{name:<22} · best F1 = {res['best_value']:.4f}")
        print(f"  best_params = {res['best_params']}")


if __name__ == "__main__":
    main()
