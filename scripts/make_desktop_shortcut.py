# -*- coding: utf-8 -*-
"""Cree un raccourci Windows (.lnk) sur le Bureau de l'utilisateur courant.

Le raccourci pointe sur `python app.py` avec icone EFREI · double-clic =
demarre API FastAPI + dashboard Streamlit + Swagger UI + ouvre le PPTX
et le PDF de la soutenance, sans ligne de commande.

Specifique Windows · necessite `pywin32` (auto-installe via requirements
ou via `pip install pywin32` au premier run). Sur Mac/Linux, lancer
directement `python app.py` depuis un terminal.

Usage ·
    python scripts/make_desktop_shortcut.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable  # le meme python que celui qui execute ce script
TARGET_PY = ROOT / "app.py"
ICON = ROOT / "assets" / "logo_efrei.ico"

DESKTOP = Path(os.environ.get("USERPROFILE", os.path.expanduser("~"))) / "Desktop"
SHORTCUT = DESKTOP / "Maintenance Predictive - Demo.lnk"


def main() -> int:
    if sys.platform != "win32":
        print(
            "[INFO] Ce script genere un raccourci Windows (.lnk).\n"
            "Sur Mac/Linux, lance directement `python app.py` depuis un terminal."
        )
        return 0

    try:
        from win32com.client import Dispatch  # pywin32
    except ImportError:
        print(
            "[ERREUR] pywin32 non disponible.\n"
            "  pip install pywin32\n"
            "ou simplement `python app.py` depuis le terminal pour lancer la demo."
        )
        return 1

    DESKTOP.mkdir(parents=True, exist_ok=True)
    shell = Dispatch("WScript.Shell")
    sc = shell.CreateShortCut(str(SHORTCUT))
    sc.TargetPath = PYTHON
    sc.Arguments = f'"{TARGET_PY}"'
    sc.WorkingDirectory = str(ROOT)
    sc.Description = "Lance API FastAPI + dashboard Streamlit + Swagger UI + PPTX + PDF"
    sc.WindowStyle = 1  # 1 = normal window (visible terminal pour voir les logs)
    if ICON.exists():
        sc.IconLocation = f"{ICON},0"
    sc.save()

    print(f"[OK] Raccourci cree : {SHORTCUT}")
    print(f"     Cible          : {PYTHON} {TARGET_PY}")
    print(f"     Icone          : {ICON if ICON.exists() else '(default)'}")
    print()
    print("Double-clique sur le raccourci pour lancer la demo complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
