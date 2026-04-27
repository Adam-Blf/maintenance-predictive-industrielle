# -*- coding: utf-8 -*-
"""Script · génération des schémas pédagogiques pour le rapport.

Ces schémas (architecture, pipeline ML, biais-variance, workflow
décisionnel) sont des illustrations conceptuelles construites avec
matplotlib pur · aucune dépendance externe (PlantUML, Graphviz).
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ensure_directories  # noqa: E402
from src.diagrams import render_all_diagrams  # noqa: E402


def main() -> None:
    """Point d'entrée."""
    ensure_directories()
    print("[DIAGRAMS] Génération des 4 schémas pédagogiques...")
    paths = render_all_diagrams()
    for name, path in paths.items():
        print(f"  - {name:<20} · {path}")
    print("Done.")


if __name__ == "__main__":
    main()
