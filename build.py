# -*- coding: utf-8 -*-
"""Build d'un executable Windows all-in-one via PyInstaller.

Genere `LANCER.exe` a la racine du repo · double-clic = demarre API +
dashboard + Swagger + ouvre le PPTX et le PDF, sans terminal manuel.

Usage ·
    python build.py

Pre-requis · PyInstaller (auto-installe via requirements.txt).

Sortie · `dist/LANCER.exe` puis copie a la racine pour acces direct.

Note · l'executable embarque le code de app.py et ses imports stdlib.
Les libs lourdes (sklearn, xgboost, streamlit, uvicorn) sont **toujours
appelees via subprocess** depuis app.py · elles doivent donc etre
disponibles dans le Python systeme. Cet exe est un wrapper double-clic,
pas un standalone.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "app.py"
ICON = ROOT / "assets" / "logo_efrei.ico"
DIST = ROOT / "dist"
BUILD = ROOT / "build"
SPEC = ROOT / "LANCER.spec"
EXE_NAME = "LANCER"
FINAL_EXE = ROOT / f"{EXE_NAME}.exe"


def main() -> int:
    if not SCRIPT.exists():
        print(f"[ERREUR] {SCRIPT} introuvable.")
        return 1
    if not ICON.exists():
        print(f"[WARN] Icone {ICON} manquante, build sans icone.")

    # Nettoyage build precedent (idempotent)
    for d in (DIST, BUILD):
        if d.exists():
            shutil.rmtree(d)
    if SPEC.exists():
        SPEC.unlink()

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",                 # un seul .exe (pas un dossier)
        "--name", EXE_NAME,
        "--console",                  # mode terminal · l'user voit les logs
        "--clean",
        "--noconfirm",
        # Hidden imports · les modules charges dynamiquement par bootstrap
        # ou pas detectes automatiquement par PyInstaller.
        "--hidden-import", "src.bootstrap",
        "--hidden-import", "win32com.client",
        # Optimisation · taille du binaire reduite
        "--strip" if sys.platform != "win32" else "--noupx",
    ]
    if ICON.exists():
        cmd.extend(["--icon", str(ICON)])
    cmd.append(str(SCRIPT))

    print("[build] Lancement PyInstaller (peut prendre 1-3 min)...")
    print("       Commande ·", " ".join(cmd))
    print()
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        print(f"[ERREUR] PyInstaller a echoue (exit={result.returncode}).")
        return result.returncode

    built_exe = DIST / f"{EXE_NAME}.exe"
    if not built_exe.exists():
        print(f"[ERREUR] {built_exe} non genere.")
        return 1

    # Copie du .exe a la racine du repo pour double-clic facile
    shutil.copy2(built_exe, FINAL_EXE)

    size_mb = FINAL_EXE.stat().st_size / (1024 * 1024)
    print()
    print("=" * 70)
    print(f"[OK] Executable cree · {FINAL_EXE}")
    print(f"     Taille          · {size_mb:.1f} Mo")
    print(f"     Build artefacts · {DIST}/, {BUILD}/, {SPEC}")
    print("=" * 70)
    print()
    print("Double-clic sur LANCER.exe (a la racine du repo) pour demarrer ·")
    print("  - API FastAPI + dashboard Streamlit + Swagger UI + ReDoc")
    print("  - Ouverture du PPTX et du PDF de la soutenance")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
