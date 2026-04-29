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

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.bootstrap import ensure_dependencies  # noqa: E402

ensure_dependencies(verbose=False)

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
