# -*- coding: utf-8 -*-
"""Script · generation du rapport PDF final via FPDF2.

Pre-requis · les scripts 01 a 05 doivent avoir ete executes pour
disposer de toutes les figures referencees dans le rapport.

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
    """Point d'entree."""
    ensure_directories()
    print("[REPORT] Construction du rapport PDF FPDF2...")
    output = render_full_report()
    size_kb = output.stat().st_size / 1024.0
    print(f"[REPORT] Genere · {output} ({size_kb:.1f} Ko)")


if __name__ == "__main__":
    main()
