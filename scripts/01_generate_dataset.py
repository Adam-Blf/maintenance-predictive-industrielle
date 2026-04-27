# -*- coding: utf-8 -*-
"""Script · génération du dataset synthétique reproductible.

Usage ·
    python scripts/01_generate_dataset.py

Produit `data/raw/industrial_machine_maintenance.csv` (24 042 lignes,
16 colonnes dont 3 cibles), reproduisant le schéma du dataset Kaggle
officiel mentionné dans le sujet. Le générateur est seedé à 42 pour
garantir la reproductibilité bit-à-bit.
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
    """Point d'entrée du script."""
    # Crée l'arborescence si elle n'existe pas (idempotent).
    ensure_directories()

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
