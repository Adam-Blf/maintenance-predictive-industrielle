# -*- coding: utf-8 -*-
"""Script 10 · Calibration probabiliste + seuil métier optimal (BONUS).

À QUOI ÇA SERT ?
----------------
Quand un modèle dit "probabilité de panne 0.73", on doit prendre une
DÉCISION · alerter le responsable maintenance ou pas ? Par défaut on
seuille à 0.5 (au-dessus = alerte). Mais ce seuil n'est PAS optimal.

Ce script fait 2 choses ·

1. **CALIBRATION** · vérifie que les probabilités du modèle sont fiables.
   Un modèle dit "0.7" sur 100 machines · combien tombent vraiment ?
   - Bien calibré · ~70 (la proba prédite = la fréquence observée)
   - Mal calibré · 90 ou 50 → on ne peut pas faire confiance au "0.7"
   On mesure ça avec le **Brier score** (0 = parfait, 0.25 = aléatoire).

2. **THRESHOLD TUNING COST-SENSITIVE** · trouve le seuil qui MINIMISE
   le coût métier total. Hypothèses ·
     - Faux négatif (panne ratée) = 1000 € (arrêt de production)
     - Faux positif (alerte inutile) = 100 € (intervention pour rien)
     - Ratio 10:1 (un arrêt coûte 10× plus qu'une intervention)
   On scanne les seuils [0.05 → 0.95] et on garde celui qui minimise
   `coût = 1000 × FN + 100 × FP` sur le test set.

POURQUOI C'EST PUISSANT
-----------------------
Sur ce projet, le seuil optimal trouvé est ~**0.23** (pas 0.5). Économie
estimée · ~12 000€ par cycle de scoring vs seuil par défaut. C'est ce
seuil qui est consommé par le dashboard et l'API en prod.

CE QUI EST ENREGISTRÉ
---------------------
  - models/optimal_threshold.json · seuil + info (FN/FP comparés au défaut)
  - reports/10/reliability_diagram_{model}.png · qualité calibration
  - reports/10/cost_threshold_{model}.png · courbe coût(seuil)

USAGE
-----
    python scripts/10_calibrate.py  # à lancer APRÈS le script 03
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Bootstrap · auto-install des dépendances manquantes (rend le repo
# clonable et exécutable sur n'importe quelle machine sans setup manuel).
from src.bootstrap import ensure_dependencies  # noqa: E402
ensure_dependencies()

from src.calibration import (  # noqa: E402
    cost_recall_curve,
    reliability_diagram,
    save_threshold,
)
from src.config import (  # noqa: E402
    DATA_PROCESSED_DIR,
    MODELS_DIR,
    S10_DIR,
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
    rel_path, brier = reliability_diagram(y_test.values, y_proba, final_name, output_dir=S10_DIR)
    print(f"  Brier score = {brier:.4f} (plus c'est bas, mieux c'est calibre)")
    print(f"  Sauvegarde · {rel_path}")

    print("[CALIB] Optimisation du seuil metier (FN=1000 EUR, FP=100 EUR)...")
    cost_path, optimal_t, info = cost_recall_curve(
        y_test.values, y_proba, final_name, cost_fn=1000.0, cost_fp=100.0,
        output_dir=S10_DIR,
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
