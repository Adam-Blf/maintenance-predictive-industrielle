# -*- coding: utf-8 -*-
"""Script · hyperparameter tuning Optuna (RF, XGB, MLP).

Rôle dans le pipeline
----------------------
Script optionnel, peut être lancé après le script 01. Son résultat
(tuning_results.json) est documentaire : les meilleurs hyperparamètres
trouvés par Optuna peuvent être réinjectés manuellement dans
`src/models.py` si une amélioration significative est observée.

Entrées
-------
data/raw/predictive_maintenance_v3.csv
    Dataset complet. Un sous-échantillon stratifié de 8000 lignes est
    utilisé pour accélérer le tuning sans dégrader la qualité.

Sorties
-------
reports/tuning_results.json
    Dictionnaire {model_name: {best_params, best_value}} pour les 3
    modèles tunés (RF, XGB, MLP). Consultable sans ré-exécuter le script.

Pré-requis
----------
- Script 01 exécuté.
- Package `optuna` installé (`pip install optuna`).

Lien cahier des charges
-----------------------
Démontre la démarche scientifique rigoureuse (section "Optimisation des
hyperparamètres") · justifie le choix des hyperparamètres finaux par
rapport aux valeurs par défaut.

Note de performance
-------------------
Durée typique · 3-5 minutes sur CPU moderne (3 modèles x 20 essais
x 3 folds). Réduire n_trials_each pour accélérer si nécessaire.

Usage ·
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
