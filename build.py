# -*- coding: utf-8 -*-
"""Build de l'executable Windows via PyInstaller.

Usage · ``python build.py``

Produit ``dist/MaintenancePredictive.exe`` · double-clic pour lancer
l'API + Dashboard + ouverture automatique du navigateur.

Convention · ``python build.py`` (jamais .bat ni .ps1, EDR hospitaliers
bloquent les scripts shell).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
EXE_NAME = "MaintenancePredictive"


def _ensure_pyinstaller() -> None:
    """Installe PyInstaller si absent."""
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("[BUILD] Installation de PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])


def _clean_old_build() -> None:
    """Supprime les anciens artefacts pour eviter les caches obsoletes."""
    for d in ("build", "dist", f"{EXE_NAME}.spec"):
        path = PROJECT_ROOT / d
        if path.is_dir():
            print(f"  cleanup · {path}")
            shutil.rmtree(path, ignore_errors=True)
        elif path.is_file():
            path.unlink(missing_ok=True)


def main() -> int:
    """Point d'entree."""
    print(f"[BUILD] Construction de {EXE_NAME}.exe via PyInstaller...")
    _ensure_pyinstaller()
    _clean_old_build()

    # Construction de la ligne de commande PyInstaller.
    # `--collect-all` est essentiel pour Streamlit / sklearn / xgboost qui
    # chargent dynamiquement des sous-modules non detectables statiquement.
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--onedir",  # OneDir plus fiable que --onefile pour Streamlit
        "--console",
        "--name",
        EXE_NAME,
        # Inclure le code source comme data (pour que le launcher trouve
        # `dashboard/app.py` et `api/main.py`).
        "--add-data",
        "dashboard;dashboard",
        "--add-data",
        "api;api",
        "--add-data",
        "src;src",
        "--add-data",
        "assets;assets",
        "--add-data",
        "models;models",
        "--add-data",
        "data;data",
        "--add-data",
        "reports/figures;reports/figures",
        # Hidden imports critiques pour Streamlit + ML.
        "--hidden-import",
        "streamlit",
        "--hidden-import",
        "uvicorn",
        "--hidden-import",
        "sklearn.ensemble",
        "--hidden-import",
        "sklearn.linear_model",
        "--hidden-import",
        "sklearn.neural_network",
        "--hidden-import",
        "xgboost",
        "--hidden-import",
        "joblib",
        # Collect-all pour les bibliotheques avec chargement dynamique.
        "--collect-all",
        "streamlit",
        "--collect-submodules",
        "sklearn",
        "--collect-submodules",
        "xgboost",
        # Note · pas de --collect-all sur xgboost / shap car leur module
        # `testing` requiert `hypothesis` (dependance dev).
        "--exclude-module",
        "hypothesis",
        "--exclude-module",
        "pytest",
        "--exclude-module",
        "tests",
        # Icone optionnelle (logo EFREI · le converti en .ico si dispo).
        # "--icon", "assets/logo_efrei.ico",
        "launcher.py",
    ]

    print()
    print("[BUILD] Commande PyInstaller ·")
    print("  " + " ".join(cmd))
    print()
    print("[BUILD] Compilation (peut prendre 5-10 minutes)...")
    print()

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print(f"[BUILD] Echec PyInstaller (code {result.returncode})")
        return result.returncode

    exe_path = PROJECT_ROOT / "dist" / EXE_NAME / f"{EXE_NAME}.exe"
    if exe_path.exists():
        size_mb = exe_path.stat().st_size / (1024 * 1024)
        print()
        print("=" * 70)
        print(f"[BUILD] Succes · {exe_path}")
        print(f"        Taille executable seul · {size_mb:.1f} Mo")
        print(f"        Lancer · double-clic sur {exe_path}")
        print("=" * 70)
        return 0

    print(f"[BUILD] Executable non trouve a {exe_path}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
