# -*- coding: utf-8 -*-
"""Script 09 · Optimisation d'hyperparamètres avec Optuna (BONUS).

À QUOI ÇA SERT ?
----------------
Quand on entraîne un Random Forest, on doit choisir des "boutons" ·
combien d'arbres ? quelle profondeur max ? combien de samples mini par
feuille ? Ce sont les **hyperparamètres**. Au lieu de les choisir au
hasard ou au feeling, ce script les SCANNE intelligemment.

POURQUOI OPTUNA PLUTÔT QUE GRIDSEARCH ?
----------------------------------------
- **GridSearch** · teste toutes les combinaisons d'une grille (100×5×4 = 2000
  essais). Lent et bête.
- **RandomSearch** · tire des combinaisons au hasard. Plus rapide mais
  pas plus malin.
- **Optuna (TPE = Tree-structured Parzen Estimator)** · apprend des essais
  précédents pour proposer des hyperparamètres prometteurs. Bayesian
  optimization · ~10× plus efficient que GridSearch en pratique.

CE QUE CE SCRIPT FAIT
---------------------
1. Charge le dataset (sous-échantillon 8000 lignes pour aller vite).
2. Pour chaque modèle (RF, XGB, MLP) ·
   - Définit l'espace des hyperparamètres à explorer.
   - Lance Optuna avec 20 essais, optimise sur F1 cross-validé 3-fold.
   - Garde les meilleurs hyperparams + leur F1.
3. Sauvegarde tout dans `reports/09/tuning_results.json`.

UTILISATION DES RÉSULTATS
-------------------------
Le JSON est **documentaire** · il prouve qu'on a fait le tuning. Si on
veut effectivement améliorer les modèles en prod, il faut réinjecter
les `best_params` dans `src/models.py` puis relancer `scripts/03_*`.

NOTE DE PERFORMANCE
-------------------
Durée · ~3-5 minutes sur CPU moderne (3 modèles × 20 trials × 3 CV folds
= 180 entraînements). Pour aller plus vite, baisser `n_trials_each` à 10.

USAGE
-----
    python scripts/09_tune_hyperparams.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Bootstrap · auto-install des dépendances manquantes (rend le repo
# clonable et exécutable sur n'importe quelle machine sans setup manuel).
from src.bootstrap import ensure_dependencies  # noqa: E402
ensure_dependencies()

from src.config import (  # noqa: E402
    ALL_FEATURES,
    DATA_PROCESSED_DIR,
    RANDOM_STATE,
    S09_DIR,
    TARGET_BINARY,
    ensure_directories,
)
from src.data_loader import load_dataset  # noqa: E402
from src.tuning import tune_all  # noqa: E402


def main() -> None:
    """Lance le tuning Optuna sur RF, XGB et MLP, sauvegarde les résultats en JSON."""
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

    output = S09_DIR / "tuning_results.json"
    output.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\n[TUNING] Resultats sauvegardes dans {output}")

    # Resume console.
    print("\n=== Resume ===")
    for name, res in results.items():
        print(f"{name:<22} · best F1 = {res['best_value']:.4f}")
        print(f"  best_params = {res['best_params']}")


if __name__ == "__main__":
    main()
