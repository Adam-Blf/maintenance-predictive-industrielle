# -*- coding: utf-8 -*-
"""Script · génération du rapport PDF final via FPDF2.

Rôle dans le pipeline
----------------------
Script n°6, à exécuter EN DERNIER dans la séquence principale. Compile
toutes les figures, métriques et textes en un rapport PDF structuré.

Entrées requises
----------------
reports/figures/*.png
    Toutes les figures générées par les scripts 02 à 05 :
    matrices de confusion, courbes ROC/PR, SHAP plots, schémas.
reports/metrics_*.csv / metrics_*.json
    Tableaux de métriques générés par les scripts 03, 07, 08.
models/final_model_name.txt
    Nom du modèle retenu (pour personnaliser les sections du rapport).
assets/logo_efrei*.png
    Logos EFREI pour la page de garde.

Sortie
------
reports/rapport_maintenance_predictive.pdf
    Rapport complet ~30 pages : page de garde, contexte, données,
    modélisation, résultats, interprétabilité, calibration, annexes.
    Encodé UTF-8 pour les caractères accentués.

Pré-requis
----------
Scripts 01 à 05 exécutés dans l'ordre. Sans les figures, FPDF2 génère
un rapport incomplet (les images manquantes sont ignorées silencieusement).

Lien cahier des charges
-----------------------
Le rapport PDF est le livrable principal de l'épreuve certifiante
RNCP40875 Bloc 2. Il doit démontrer la démarche scientifique complète.

Usage ·
    python scripts/06_generate_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ensure_directories  # noqa: E402
from src.report import render_full_report  # noqa: E402


def main() -> None:
    """Construit et sauvegarde le rapport PDF complet via FPDF2."""
    ensure_directories()
    print("[REPORT] Construction du rapport PDF FPDF2...")
    output = render_full_report()
    size_kb = output.stat().st_size / 1024.0
    print(f"[REPORT] Genere · {output} ({size_kb:.1f} Ko)")


if __name__ == "__main__":
    main()
