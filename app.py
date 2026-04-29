# -*- coding: utf-8 -*-
"""Orchestrateur unifie · lance API FastAPI + dashboard Streamlit + Swagger UI.

Usage ·
    python app.py

Sequence d'execution ·
    1. Demarrage de l'API FastAPI (uvicorn) sur 127.0.0.1:8000
    2. Attente de /health 200 (max 30s)
    3. Demarrage du dashboard Streamlit sur localhost:8501,
       connecte a l'API via la variable d'environnement API_BASE_URL
    4. Ouverture des 3 onglets navigateur ·
       - Swagger UI (visualisation API pour la soutenance) · /docs
       - Dashboard metier
       - ReDoc (alternative Swagger, plus presentable) · /redoc

Arret propre via Ctrl+C · les 2 sous-processus sont termines avec un
SIGTERM, puis SIGKILL si timeout.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

# Detection du mode frozen PyInstaller · quand le programme tourne depuis
# `LANCER.exe`, sys.frozen=True et le code est extrait dans un temp _MEIxxxx.
# Dans ce cas, ROOT doit pointer sur le dossier qui CONTIENT le .exe (= repo
# clone ou le user a place LANCER.exe), pas sur le temp d'extraction.
if getattr(sys, "frozen", False):
    ROOT = Path(sys.executable).resolve().parent
else:
    ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.bootstrap import ensure_dependencies  # noqa: E402

# verbose=True · l'utilisateur voit les packages manquants et la progression
# de pip install au lancement. Idempotent · aucun appel reseau si tout est
# deja installe (le test est `importlib.util.find_spec` par package).
print("[0/3] Verification des dependances (requirements.txt)...")
ensure_dependencies(verbose=True)
print("      Dependances OK.\n")


# ---------------------------------------------------------------------------
# Auto-installation du raccourci bureau (Windows uniquement, idempotent)
# ---------------------------------------------------------------------------
def _autoinstall_desktop_shortcut() -> None:
    """Cree le raccourci `Maintenance Predictive - Demo.lnk` sur le bureau
    si on est sous Windows et qu'il n'existe pas deja. Resilient · ignore
    silencieusement toute erreur (pywin32 manquant, drive readonly, etc.)
    pour ne jamais bloquer le lancement du serveur.

    Idempotent · si le raccourci existe deja avec la bonne cible, ne fait
    rien. Si il existe avec une cible obsolete (autre python.exe), il est
    re-ecrit. C'est cela qui rend la solution **portable** entre machines ·
    cloner le repo + lancer `python app.py` une fois suffit a obtenir un
    raccourci bureau valide sur N'IMPORTE QUELLE machine Windows.
    """
    if sys.platform != "win32":
        return  # Mac/Linux · pas de .lnk, lancer `python app.py` directement
    try:
        from win32com.client import Dispatch  # type: ignore

        desktop_root = os.environ.get("USERPROFILE") or os.path.expanduser("~")
        desktop = Path(desktop_root) / "Desktop"
        desktop.mkdir(parents=True, exist_ok=True)
        shortcut_path = desktop / "Maintenance Predictive - Demo.lnk"

        target_python = sys.executable
        target_arg = f'"{Path(__file__).resolve()}"'
        icon = Path(__file__).resolve().parent / "assets" / "logo_efrei.ico"

        # Si le raccourci existe deja et pointe sur ce python.exe, on
        # considere qu'il est a jour (evite une re-ecriture inutile a chaque
        # lancement).
        if shortcut_path.exists():
            try:
                shell_check = Dispatch("WScript.Shell")
                existing = shell_check.CreateShortCut(str(shortcut_path))
                if existing.TargetPath == target_python and existing.Arguments == target_arg:
                    return  # raccourci deja a jour, no-op
            except Exception:
                pass  # on tentera la re-ecriture quand meme

        shell = Dispatch("WScript.Shell")
        sc = shell.CreateShortCut(str(shortcut_path))
        sc.TargetPath = target_python
        sc.Arguments = target_arg
        sc.WorkingDirectory = str(Path(__file__).resolve().parent)
        sc.Description = "Lance API FastAPI + dashboard Streamlit + Swagger + PPTX + PDF"
        sc.WindowStyle = 1  # 1 = fenetre normale (terminal visible)
        if icon.exists():
            sc.IconLocation = f"{icon},0"
        sc.save()
        print(f"[setup] Raccourci bureau cree/mis a jour · {shortcut_path}")
    except Exception as exc:
        # Pas de pywin32 ou autre · on log et on continue, ce n'est pas bloquant
        print(f"[setup] Raccourci bureau non cree (non bloquant) · {exc}")


_autoinstall_desktop_shortcut()

# ---------------------------------------------------------------------------
# Configuration des services
# ---------------------------------------------------------------------------
API_HOST = "127.0.0.1"
API_PORT = 8000
DASHBOARD_PORT = 8501

API_BASE_URL = f"http://{API_HOST}:{API_PORT}"
DASHBOARD_URL = f"http://localhost:{DASHBOARD_PORT}"
SWAGGER_URL = f"{API_BASE_URL}/docs"
REDOC_URL = f"{API_BASE_URL}/redoc"

# Livrables ouverts automatiquement pour la soutenance
PPTX_PATH = ROOT / "reports" / "11" / "presentation.pptx"
PDF_PATH = ROOT / "reports" / "06" / "rapport_projet_data_science.pdf"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _wait_for(url: str, timeout: float = 30.0, interval: float = 0.5) -> bool:
    """Boucle jusqu'a ce que `url` reponde HTTP 200, ou timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(interval)
    return False


def _print_banner(title: str, lines: list[str]) -> None:
    width = max(len(title), *(len(line) for line in lines)) + 4
    bar = "=" * width
    print(bar)
    print(f"  {title}")
    print(bar)
    for line in lines:
        print(f"  {line}")
    print(bar)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    _print_banner(
        "Maintenance Predictive · orchestrateur unifie",
        [
            f"API REST       · {API_BASE_URL}",
            f"Swagger UI     · {SWAGGER_URL}",
            f"ReDoc          · {REDOC_URL}",
            f"Dashboard      · {DASHBOARD_URL}",
            "Ctrl+C pour tout arreter proprement.",
        ],
    )
    print()

    # ---------------------- 1. Lancement de l'API ----------------------
    print("[1/3] Demarrage de l'API FastAPI (uvicorn)...")
    api_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "api.main:app",
        "--host",
        API_HOST,
        "--port",
        str(API_PORT),
        "--log-level",
        "warning",
    ]
    api_proc = subprocess.Popen(api_cmd, cwd=str(ROOT))

    print(f"      Attente de {API_BASE_URL}/health (max 30s)...")
    if not _wait_for(f"{API_BASE_URL}/health", timeout=30.0):
        print("[ERREUR] L'API n'a pas repondu dans le delai. Arret.")
        api_proc.terminate()
        return 1
    print("      API operationnelle.")

    # ---------------------- 2. Lancement du dashboard ----------------------
    print("\n[2/3] Demarrage du dashboard Streamlit (branche sur l'API)...")
    env = os.environ.copy()
    env["API_BASE_URL"] = API_BASE_URL  # consomme par dashboard/app.py
    dashboard_cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "dashboard/app.py",
        "--server.port",
        str(DASHBOARD_PORT),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]
    dash_proc = subprocess.Popen(dashboard_cmd, cwd=str(ROOT), env=env)

    print(f"      Attente du dashboard sur {DASHBOARD_URL}...")
    if not _wait_for(DASHBOARD_URL, timeout=20.0):
        # Streamlit prend parfois 5-10s · on ne kill pas, on continue.
        print("      (Le dashboard demarre encore, on ouvre quand meme.)")
    else:
        print("      Dashboard operationnel.")

    # ---------------------- 3. Ouverture des navigateurs ----------------------
    print("\n[3/3] Ouverture des interfaces dans le navigateur...")
    # Swagger UI · interface interactive de l'API (utile pour la soutenance)
    webbrowser.open(SWAGGER_URL)
    time.sleep(0.6)
    # Dashboard metier
    webbrowser.open(DASHBOARD_URL)
    time.sleep(0.6)
    # ReDoc · vue documentaire alternative, plus lisible pour le jury
    webbrowser.open(REDOC_URL)
    time.sleep(0.6)

    # Ouverture des livrables pour la soutenance (PowerPoint + PDF)
    if PPTX_PATH.exists():
        try:
            os.startfile(str(PPTX_PATH))
            print(f"      Slides    · {PPTX_PATH.name} ouvert dans PowerPoint")
        except Exception as exc:
            print(f"      [warn] impossible d'ouvrir le pptx · {exc}")
    if PDF_PATH.exists():
        try:
            os.startfile(str(PDF_PATH))
            print(f"      Rapport   · {PDF_PATH.name} ouvert dans le viewer PDF")
        except Exception as exc:
            print(f"      [warn] impossible d'ouvrir le pdf · {exc}")

    print()
    _print_banner(
        "Tout est en route",
        [
            f"  Swagger UI   · {SWAGGER_URL}",
            f"  ReDoc        · {REDOC_URL}",
            f"  Dashboard    · {DASHBOARD_URL}",
            "Ctrl+C pour arreter l'API et le dashboard.",
        ],
    )

    # ---------------------- Boucle de garde ----------------------
    try:
        while True:
            if api_proc.poll() is not None:
                print("[app] L'API s'est arretee de maniere inattendue.")
                break
            if dash_proc.poll() is not None:
                print("[app] Le dashboard s'est arrete de maniere inattendue.")
                break
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[app] Ctrl+C detecte · arret des sous-processus...")

    # ---------------------- Arret propre ----------------------
    for proc, label in ((dash_proc, "dashboard"), (api_proc, "api")):
        if proc.poll() is None:
            print(f"[app] Kill {label}...")
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
    print("[app] Termine.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
