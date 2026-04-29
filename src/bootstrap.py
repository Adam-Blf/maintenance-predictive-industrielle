# -*- coding: utf-8 -*-
"""Module de bootstrap · garantit que toutes les dépendances sont installées
avant l'exécution des scripts.

À QUOI ÇA SERT ?
----------------
Quand un camarade clone le repo et lance `python scripts/02_eda.py`, il
peut tomber sur `ModuleNotFoundError: No module named 'pandas'` parce qu'il
n'a pas fait `pip install -r requirements.txt`. Ce module évite ça ·
appeler `ensure_dependencies()` au début de chaque script déclenche un
`pip install` automatique des packages manquants.

PORTABILITÉ DES CHEMINS
-----------------------
Tous les chemins du projet sont calculés depuis `Path(__file__).resolve()`,
ce qui les rend ABSOLUS et indépendants du dossier courant ·

    PROJECT_ROOT = Path(__file__).resolve().parent.parent

Que l'utilisateur lance `python scripts/02_eda.py` depuis ·
  - C:/Users/adam/Documents/...
  - /home/emilien/projects/...
  - /Users/jury/Desktop/...
... le code trouve toujours `data/raw/`, `models/`, `reports/02/`, etc.
Aucun chemin Windows ou Linux n'est hardcodé.

USAGE
-----
    from src.bootstrap import ensure_dependencies
    ensure_dependencies()  # appelé en haut des scripts/*.py

Idempotent · si tout est déjà installé, ne fait rien (pas de network call).
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

# Racine du projet calculée depuis l'emplacement de CE fichier ·
# `src/bootstrap.py`.parent.parent → racine du repo. Portable Win/Linux/Mac.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
REQUIREMENTS_FILE: Path = PROJECT_ROOT / "requirements.txt"

# Mapping {nom_pip → nom_import_python}.
# Pour la plupart des packages le nom pip == nom import (`pandas` → `import pandas`),
# mais certains diffèrent (`scikit-learn` → `import sklearn`).
# On vérifie la présence via le NOM D'IMPORT, on installe via le NOM PIP.
_PIP_TO_IMPORT_NAME: dict[str, str] = {
    "scikit-learn": "sklearn",
    "Pillow": "PIL",
    "PyYAML": "yaml",
    "fpdf2": "fpdf",
    "python-pptx": "pptx",
    "python-dotenv": "dotenv",
    "pytest-cov": "pytest_cov",
    "uvicorn": "uvicorn",
    # pywin32 est un meta-package · ses modules s'importent via win32com,
    # win32api, win32con, etc. On utilise win32com comme sentinelle.
    "pywin32": "win32com",
}


def _is_installed(import_name: str) -> bool:
    """Retourne True si le package est importable, False sinon.

    On utilise `importlib.util.find_spec` plutôt que `import` direct ·
    plus rapide (pas de chargement réel) et n'importe rien dans le namespace.
    """
    return importlib.util.find_spec(import_name) is not None


def _parse_requirements() -> list[str]:
    """Lit `requirements.txt` et retourne la liste des noms pip (sans version).

    Exemple · "pandas>=2.0.0" → "pandas"
    Ignore les commentaires et lignes vides.
    """
    if not REQUIREMENTS_FILE.exists():
        return []
    requirements: list[str] = []
    for raw in REQUIREMENTS_FILE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        # Skip commentaires et lignes vides.
        if not line or line.startswith("#"):
            continue
        # Strip les markers PEP 508 (environment markers) AVANT le split
        # operator · ex. `pywin32>=306; sys_platform == "win32"` doit donner
        # `pywin32`, pas `pywin32>=306; sys_platform` (le == du marker
        # confond le parser).
        if ";" in line:
            line = line.split(";", 1)[0].strip()
        # Extraction du nom · "pandas>=2.0.0" → "pandas".
        # On split sur les opérateurs de version usuels.
        for sep in ("==", ">=", "<=", "~=", ">", "<"):
            if sep in line:
                line = line.split(sep)[0]
                break
        # Retirer les extras style "uvicorn[standard]" → "uvicorn"
        # (les extras sont gérés par pip mais ne sont pas un nom d'import).
        if "[" in line:
            line = line.split("[")[0]
        # Retirer les tirets/underscores qui rendent l'import différent du nom pip ·
        # ex. "pytest-cov" → import "pytest_cov", "python-dotenv" → import "dotenv".
        requirements.append(line.strip())
    return requirements


def ensure_dependencies(verbose: bool = True) -> None:
    """Installe les dépendances manquantes en lançant `pip install`.

    Algorithme ·
      1. Lire requirements.txt → liste des packages requis.
      2. Pour chaque package, vérifier s'il est déjà importable.
      3. Si certains manquent · `pip install -r requirements.txt`.
      4. Sinon · ne rien faire (idempotent, pas de coût réseau).

    Parameters
    ----------
    verbose : bool
        Si True, affiche les actions à l'écran. Mettre False pour les
        appels en mode batch (ex. dans une CI ou dans Streamlit).
    """
    requirements = _parse_requirements()
    if not requirements:
        if verbose:
            print("[bootstrap] requirements.txt introuvable · skip.")
        return

    # Liste des packages absents.
    missing: list[str] = []
    for pip_name in requirements:
        import_name = _PIP_TO_IMPORT_NAME.get(pip_name, pip_name)
        if not _is_installed(import_name):
            missing.append(pip_name)

    if not missing:
        return  # Tout est déjà installé · no-op.

    if verbose:
        print(f"[bootstrap] {len(missing)} package(s) manquant(s) · {missing}")
        print("[bootstrap] Lancement de `pip install -r requirements.txt`...")

    # `sys.executable` · garantit qu'on utilise le même Python que celui
    # qui exécute le script (utile en venv pour ne pas installer dans le
    # Python système par accident).
    cmd = [sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE)]
    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        print(
            "[bootstrap] L'installation pip a échoué. Lance manuellement ·\n"
            f"    {' '.join(cmd)}",
            file=sys.stderr,
        )
        sys.exit(1)

    if verbose:
        print("[bootstrap] Dépendances installées avec succès.")
