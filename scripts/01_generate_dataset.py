# -*- coding: utf-8 -*-
"""Script · génération du dataset synthétique reproductible.

Rôle dans le pipeline
----------------------
Script n°1, à exécuter EN PREMIER si le fichier CSV Kaggle officiel
n'est pas disponible localement. Il génère un dataset synthétique au
schéma exact de `predictive_maintenance_v3.csv` pour que tous les
scripts suivants puissent tourner sans connexion Internet ni clé API.

Entrées
-------
Aucun fichier d'entrée requis.

Sorties
-------
data/raw/predictive_maintenance_v3.csv
    Dataset synthétique · 24 042 lignes, 15 colonnes, schéma officiel
    Kaggle v3.0, seedé à RANDOM_STATE=42 pour reproductibilité.

Pré-requis
----------
- Package `src` accessible (ajouté via sys.path.insert).
- Dossier `data/raw/` créé automatiquement par `ensure_directories()`.

Lien cahier des charges
-----------------------
Répond à la contrainte "données reproductibles hors ligne" imposée par
le sujet pour permettre la relecture par l'évaluateur sans accès Kaggle.

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

from src.config import DATASET_PATH, RANDOM_STATE, ensure_directories  # noqa: E402
from src.data_loader import generate_synthetic_dataset  # noqa: E402


def main() -> None:
    """Point d'entrée du script de génération du dataset synthétique.

    Comportement ·
    - Si `data/raw/predictive_maintenance_v3.csv` existe déjà (cas du
      CSV Kaggle officiel téléchargé manuellement), on saute la
      génération synthétique pour ne PAS écraser le vrai dataset.
      Pour forcer une régénération synthétique, supprimer le CSV avant
      de relancer le script.
    - Sinon · on génère la version synthétique de secours (seedée à 42)
      et on l'écrit. Affiche un résumé de validation rapide pour
      confirmer que les proportions de pannes et la distribution RUL
      sont cohérentes.
    """
    # Crée l'arborescence si elle n'existe pas (idempotent).
    ensure_directories()

    # Garde-fou anti-écrasement · si le CSV Kaggle officiel est déjà en
    # place, on ne le remplace surtout pas par une version synthétique.
    if DATASET_PATH.exists():
        size_kb = DATASET_PATH.stat().st_size / 1024
        print(f"[skip] CSV déjà présent · {DATASET_PATH} ({size_kb:.0f} Ko)")
        print("       Pour forcer une régénération synthétique, supprimer le")
        print("       fichier ci-dessus puis relancer ce script.")
        return

    print(f"[1/3] Génération de 24 042 enregistrements (seed={RANDOM_STATE})...")
    df = generate_synthetic_dataset(n_samples=24_042, seed=RANDOM_STATE)

    print(f"[2/3] Sauvegarde vers {DATASET_PATH}")
    df.to_csv(DATASET_PATH, index=False, encoding="utf-8")

    # Petit résumé pour validation visuelle rapide en console.
    print("[3/3] Statistiques rapides ·")
    print(f"  - Shape · {df.shape}")
    print(f"  - Pannes 24h · {df['failure_within_24h'].mean():.2%}")
    print("  - Distribution failure_type ·")
    for k, v in df["failure_type"].value_counts().items():
        print(f"      {k:<12} · {v:>6} ({v/len(df):.1%})")
    print(
        "  - RUL hours · "
        f"min={df['rul_hours'].min()}, "
        f"median={df['rul_hours'].median():.0f}, "
        f"max={df['rul_hours'].max()}"
    )
    print("Done.")


if __name__ == "__main__":
    main()
