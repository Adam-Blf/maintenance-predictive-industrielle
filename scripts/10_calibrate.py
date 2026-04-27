# -*- coding: utf-8 -*-
"""Script · calibration probabiliste + threshold tuning.

Utilise le modele final binaire (failure_within_24h) pour produire ·
  - Reliability diagram + Brier score
  - Courbe coût/seuil avec hypothèses métier (FN=1000€, FP=100€)
  - Threshold optimal sauvegardé dans models/optimal_threshold.json
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration import (  # noqa: E402
    cost_recall_curve,
    reliability_diagram,
    save_threshold,
)
from src.config import (  # noqa: E402
    DATA_PROCESSED_DIR,
    MODELS_DIR,
    ensure_directories,
)


def main() -> None:
    ensure_directories()

    final_name = (MODELS_DIR / "final_model_name.txt").read_text(encoding="utf-8").strip()
    print(f"[CALIB] Modele final · {final_name}")
    pipeline = joblib.load(MODELS_DIR / "final_model.joblib")

    X_test = pd.read_csv(DATA_PROCESSED_DIR / "X_test.csv")
    y_test = pd.read_csv(DATA_PROCESSED_DIR / "y_test.csv").iloc[:, 0]
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    print("[CALIB] Reliability diagram + Brier score...")
    rel_path, brier = reliability_diagram(y_test.values, y_proba, final_name)
    print(f"  Brier score = {brier:.4f} (plus c'est bas, mieux c'est calibre)")
    print(f"  Sauvegarde · {rel_path}")

    print("[CALIB] Optimisation du seuil metier (FN=1000 EUR, FP=100 EUR)...")
    cost_path, optimal_t, info = cost_recall_curve(
        y_test.values, y_proba, final_name, cost_fn=1000.0, cost_fp=100.0
    )
    print(f"  Seuil optimal · {optimal_t:.3f}")
    print(f"  Cout total au seuil optimal · {info['optimal_cost_eur']:.0f} EUR")
    print(
        f"  Cout au seuil 0.5 (default) · "
        f"{1000*info['fn_at_default_0_5'] + 100*info['fp_at_default_0_5']:.0f} EUR"
    )

    save_threshold(optimal_t, info, MODELS_DIR / "optimal_threshold.json")
    print(f"  Threshold persiste dans {MODELS_DIR / 'optimal_threshold.json'}")
    print("Done.")


if __name__ == "__main__":
    main()
