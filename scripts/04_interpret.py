# -*- coding: utf-8 -*-
"""Script · interprétabilité du modèle final candidat (3 niveaux).

Rôle dans le pipeline
----------------------
Script n°4, à exécuter APRES le script 03 (entraînement binaire).
Produit les visualisations d'explicabilité requises par le cahier des
charges pour satisfaire le critère d'interprétabilité.

Entrées
-------
models/final_model.joblib
    Pipeline sklearn du meilleur modèle binaire (issu du script 03).
models/final_model_name.txt
    Nom textuel du modèle final (ex. "random_forest").
data/processed/X_test.csv
    Features du test set (sauvegardées par le script 03).
data/processed/y_test.csv
    Labels du test set (sauvegardées par le script 03).

Sorties (dans reports/04/)
---------------------------------
feature_importance_native_{model}.png
    Importance des features par réduction d'impureté (si RF/XGB).
permutation_importance_{model}.png
    Perte de F1 lors de la permutation de chaque feature (tous modèles).
shap_summary_{model}.png
    SHAP dot plot · contribution de chaque feature par observation.
shap_bar_{model}.png
    SHAP bar plot · importance globale (moyenne des |SHAP|).

Pré-requis
----------
- Scripts 01, 02, 03 exécutés.
- Package `shap` installé (`pip install shap`) pour le niveau avancé.

Lien cahier des charges
-----------------------
Répond aux exigences d'explicabilité C3.1 (feature importance basique),
C3.2 (permutation, recommandé), C3.3 (SHAP, avancé). Les 3 niveaux
sont produits systématiquement pour couvrir tous les critères.

Usage ·
    python scripts/04_interpret.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    ALL_FEATURES,
    DATA_PROCESSED_DIR,
    MODELS_DIR,
    S04_DIR,
    ensure_directories,
)
from src.interpretability import (  # noqa: E402
    compute_shap_values,
    plot_native_feature_importance,
    plot_permutation_importance,
)
from src.preprocessing import get_feature_names  # noqa: E402


def main() -> None:
    """Orchestre les 3 niveaux d'interprétabilité sur le modèle final."""
    ensure_directories()

    # Lecture du nom du modèle final + chargement du pipeline.
    name_file = MODELS_DIR / "final_model_name.txt"
    if not name_file.exists():
        raise FileNotFoundError("final_model_name.txt manquant. Lancer scripts/03_train_models.py.")
    final_name = name_file.read_text(encoding="utf-8").strip()
    print(f"[INTERPRET] Modèle final · {final_name}")

    pipeline = joblib.load(MODELS_DIR / "final_model.joblib")

    # Reload du test set sauvegardé par 03_train_models.py.
    X_test = pd.read_csv(DATA_PROCESSED_DIR / "X_test.csv")
    y_test = pd.read_csv(DATA_PROCESSED_DIR / "y_test.csv").iloc[:, 0]

    # Récupération des noms de features post-One-Hot pour SHAP.
    preprocessor = pipeline.named_steps["preprocessor"]
    feature_names_processed = get_feature_names(preprocessor)

    # ------------------------------------------------------------------
    # 1. Feature importance native (si applicable).
    # ------------------------------------------------------------------
    print("[INTERPRET] 1/3 · Feature importance native...")
    native_path = plot_native_feature_importance(
        pipeline, feature_names_processed, final_name, output_dir=S04_DIR
    )
    if native_path:
        print(f"  -> {native_path}")
    else:
        print("  Modele sans feature_importances_ natif (skip).")

    # ------------------------------------------------------------------
    # 2. Permutation importance (raw features pour interprétation métier).
    # ------------------------------------------------------------------
    print("[INTERPRET] 2/3 · Permutation importance...")
    perm_path = plot_permutation_importance(
        pipeline,
        X_test,
        y_test,
        feature_names_raw=ALL_FEATURES,
        model_name=final_name,
        output_dir=S04_DIR,
    )
    print(f"  -> {perm_path}")

    # ------------------------------------------------------------------
    # 3. SHAP (avancé · explicabilité locale + globale).
    # ------------------------------------------------------------------
    print("[INTERPRET] 3/3 · SHAP values (peut prendre 1-2 min)...")
    shap_summary, shap_bar = compute_shap_values(
        pipeline,
        X_sample=X_test,
        feature_names_processed=feature_names_processed,
        model_name=final_name,
        max_samples=400,
        output_dir=S04_DIR,
    )
    if shap_summary:
        print(f"  -> {shap_summary}")
        print(f"  -> {shap_bar}")

    print("\nDone.")


if __name__ == "__main__":
    main()
