# -*- coding: utf-8 -*-
"""Script · génération des schémas pédagogiques pour le rapport.

Rôle dans le pipeline
----------------------
Script n°5, indépendant des données · peut être exécuté à tout moment.
Génère des illustrations conceptuelles intégrées dans le rapport PDF
par FPDF2 (script 06). Aucune dépendance au dataset ou aux modèles.

Schémas produits (dans reports/05/)
-----------------------------------------
pipeline_overview.png
    Schéma du pipeline ML complet · données → preprocessing → modèles
    → évaluation → calibration → API/dashboard.
architecture_diagram.png
    Architecture logicielle du projet (package src/, scripts, API,
    dashboard, rapport).
bias_variance_tradeoff.png
    Illustration du compromis biais-variance avec les 4 modèles positionnés.
decision_workflow.png
    Workflow décisionnel de maintenance (capteurs → prédiction → action).

Choix technique
---------------
Matplotlib pur, sans dépendance externe (PlantUML, Graphviz, draw.io).
Garantit que le script tourne dans tout environnement minimal
avec uniquement matplotlib installé.

Pré-requis
----------
Aucun script préalable requis.

Lien cahier des charges
-----------------------
Ces schémas répondent aux exigences de documentation et de présentation
du pipeline imposées par le rapport final (section Architecture).

Usage ·
    python scripts/05_generate_diagrams.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Bootstrap · auto-install des dépendances manquantes (rend le repo
# clonable et exécutable sur n'importe quelle machine sans setup manuel).
from src.bootstrap import ensure_dependencies  # noqa: E402
ensure_dependencies()

from src.config import S05_DIR, ensure_directories  # noqa: E402
from src.diagrams import render_all_diagrams  # noqa: E402


def main() -> None:
    """Génère et sauvegarde les 4 schémas pédagogiques du rapport."""
    ensure_directories()
    print("[DIAGRAMS] Génération des 4 schémas pédagogiques...")
    paths = render_all_diagrams(output_dir=S05_DIR)
    for name, path in paths.items():
        print(f"  - {name:<20} · {path}")
    print("Done.")


if __name__ == "__main__":
    main()
