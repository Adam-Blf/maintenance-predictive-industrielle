# -*- coding: utf-8 -*-
"""Script · calibration probabiliste + threshold tuning du modèle binaire.

Rôle dans le pipeline
----------------------
Script n°10, à exécuter APRES le script 03 (entraînement binaire).
Complète l'évaluation avec deux analyses avancées :
  1. Qualité de la calibration probabiliste (reliability diagram + Brier score).
  2. Optimisation du seuil de décision selon les coûts métier.

Entrées
-------
models/final_model.joblib
    Pipeline du meilleur modèle binaire (issu du script 03).
models/final_model_name.txt
    Nom textuel du modèle final.
data/processed/X_test.csv
    Features du test set (sauvegardées par le script 03).
data/processed/y_test.csv
    Labels du test set (sauvegardées par le script 03).

Sorties
-------
reports/figures/reliability_diagram_{model}.png
    Diagramme de fiabilité + Brier score. Un modèle parfaitement calibré
    suit la diagonale.
reports/figures/cost_threshold_{model}.png
    Courbe coût total (FN + FP) en fonction du seuil de décision.
models/optimal_threshold.json
    Seuil optimal et info associée (coûts, FN/FP comparés). Consommé
    par l'API FastAPI et le dashboard Streamlit pour appliquer le bon seuil.

Pré-requis
----------
Scripts 01 et 03 exécutés.

Lien cahier des charges
-----------------------
Répond à l'exigence de "démarche métier" du cahier des charges : justifier
le seuil de décision par une analyse coût-bénéfice plutôt que d'utiliser
le seuil par défaut de 0.5.

Hypothèses métier utilisées
-----------------------------
- Coût faux négatif (panne ratée) · 1000 EUR (arrêt de production).
- Coût faux positif (intervention inutile) · 100 EUR (main d'oeuvre).
- Ratio 10:1 cohérent avec la littérature industrielle.

Usage ·
    python scripts/10_calibrate.py
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
    """Évalue la calibration et optimise le seuil de décision métier."""
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
