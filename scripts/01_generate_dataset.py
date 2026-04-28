# -*- coding: utf-8 -*-
"""Script · vérification du dataset Kaggle officiel.

Rôle dans le pipeline
----------------------
Script n°1, étape de validation avant tout traitement. Vérifie que le
CSV Kaggle officiel `predictive_maintenance_v3.csv` est bien présent
dans `data/raw/`, qu'il respecte le schéma attendu (15 colonnes), et
affiche un résumé statistique pour validation visuelle.

**Règle stricte** · ce script ne crée JAMAIS de CSV. Le projet doit
toujours utiliser le dataset Kaggle officiel CC0 ·
`https://www.kaggle.com/datasets/tatheerabbas/industrial-machine-predictive-maintenance`
(téléchargement manuel ou via MCP Kaggle).

Si le CSV est absent, le script échoue avec un message d'erreur clair
indiquant comment l'obtenir. Aucune génération synthétique en
production · `src.data_loader.generate_synthetic_dataset()` reste
disponible **uniquement** pour les tests unitaires (`tests/test_*.py`).

Entrées
-------
data/raw/predictive_maintenance_v3.csv
    Dataset Kaggle officiel · 24 042 lignes, 15 colonnes.

Sorties
-------
Aucun fichier produit. Console uniquement (résumé stats).

Pré-requis
----------
- CSV Kaggle déjà présent dans `data/raw/`.
- Package `src` accessible (ajouté via sys.path.insert).

Lien cahier des charges
-----------------------
Répond à C3.1 (préparation des données) en validant la qualité des
données AVANT modélisation. Cohérent avec la règle anti-data-leakage ·
on charge le dataset complet pour validation, sans le modifier.

Usage ·
    python scripts/01_generate_dataset.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# On ajoute la racine du projet au PYTHONPATH pour permettre
# l'import du package `src.*` sans avoir à installer le projet
# en mode éditable (pratique pour un livrable étudiant).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATASET_KAGGLE_REF, DATASET_PATH, ensure_directories  # noqa: E402
from src.data_loader import load_dataset  # noqa: E402


def main() -> None:
    """Vérifie la présence et la validité du CSV Kaggle officiel.

    Comportement ·
    1. Vérifie l'existence du fichier `data/raw/predictive_maintenance_v3.csv`.
    2. Si absent · affiche les instructions de téléchargement Kaggle puis
       sort en code 1 (ne génère JAMAIS de CSV synthétique).
    3. Si présent · charge via `load_dataset()` (qui valide le schéma à
       15 colonnes) et affiche un résumé descriptif.
    """
    ensure_directories()

    if not DATASET_PATH.exists():
        print("[ERROR] CSV Kaggle officiel introuvable.")
        print(f"        Chemin attendu · {DATASET_PATH}")
        print()
        print("  Pour l'obtenir ·")
        print(f"  1. Telecharger depuis Kaggle · {DATASET_KAGGLE_REF}")
        print("     (URL · https://www.kaggle.com/datasets/"
              "tatheerabbas/industrial-machine-predictive-maintenance)")
        print("  2. Extraire le fichier predictive_maintenance_v3.csv")
        print(f"  3. Le placer dans · {DATASET_PATH.parent}")
        print("  4. Relancer ce script pour valider.")
        print()
        print("  Note · ce script NE genere PAS de CSV synthetique en")
        print("  production. La fonction synthetique reste disponible")
        print("  uniquement pour les tests unitaires (tests/test_*.py).")
        sys.exit(1)

    size_kb = DATASET_PATH.stat().st_size / 1024
    print(f"[1/3] CSV present · {DATASET_PATH} ({size_kb:.0f} Ko)")

    # Charge + valide le schema (lance une exception si colonnes manquantes).
    print("[2/3] Validation du schema (15 colonnes officielles)...")
    df = load_dataset()
    print(f"      OK · shape = {df.shape}")

    # Resume descriptif pour validation visuelle rapide en console.
    print("[3/3] Statistiques rapides ·")
    print(f"  - Pannes 24h · {df['failure_within_24h'].mean():.2%}")
    print("  - Distribution failure_type ·")
    for k, v in df["failure_type"].value_counts().items():
        print(f"      {k:<15} · {v:>6} ({v/len(df):.1%})")
    print(
        "  - RUL hours · "
        f"min={df['rul_hours'].min()}, "
        f"median={df['rul_hours'].median():.0f}, "
        f"max={df['rul_hours'].max()}"
    )
    nan_cells = int(df.isna().sum().sum())
    nan_cols = int((df.isna().sum() > 0).sum())
    print(f"  - NaN total · {nan_cells:,} cellules sur {nan_cols} colonnes")
    print()
    print("Done. CSV pret pour le pipeline (script 02_eda.py et suivants).")


if __name__ == "__main__":
    main()
