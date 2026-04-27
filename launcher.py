# -*- coding: utf-8 -*-
"""Launcher de la plateforme Maintenance Predictive.

Demarre en parallele ·
  - L'API FastAPI (uvicorn) sur http://127.0.0.1:8000
  - Le Dashboard Streamlit sur http://127.0.0.1:8501

Puis ouvre automatiquement le navigateur sur le dashboard.

Compile en `.exe` avec ``python build.py`` (PyInstaller). L'executable
resultant est lance en double-cliquant ; les serveurs s'arretent
proprement quand on ferme la console.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path


def _resolve_root() -> Path:
    """Localise la racine projet (compatible PyInstaller bundle)."""
    if getattr(sys, "frozen", False):
        # Mode PyInstaller · les ressources sont extraites dans _MEIPASS.
        return Path(getattr(sys, "_MEIPASS", Path(__file__).parent))
    return Path(__file__).resolve().parent


PROJECT_ROOT = _resolve_root()


def _wait_for_port(host: str, port: int, timeout: float = 30.0) -> bool:
    """Attend que le port ecoute (serveur pret)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.4)
    return False


def _start_uvicorn() -> subprocess.Popen:
    """Demarre l'API FastAPI via uvicorn en sous-processus."""
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "api.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
    ]
    return subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
    )


def _start_streamlit() -> subprocess.Popen:
    """Demarre le Dashboard Streamlit en sous-processus."""
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(PROJECT_ROOT / "dashboard" / "app.py"),
        "--server.port",
        "8501",
        "--server.address",
        "127.0.0.1",
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]
    return subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
    )


def _print_banner() -> None:
    """Affiche le banner de demarrage."""
    print("=" * 70)
    print("  MAINTENANCE PREDICTIVE INDUSTRIELLE")
    print("  Adam Beloucif & Emilien Morice  |  EFREI 2025-2026")
    print("=" * 70)
    print()
    print("Demarrage de l'API et du Dashboard...")
    print("  API           · http://127.0.0.1:8000  (Swagger · /docs)")
    print("  Dashboard     · http://127.0.0.1:8501")
    print()
    print("Fermez cette fenetre pour arreter les services.")
    print("=" * 70)
    print()


def main() -> int:
    """Point d'entree."""
    _print_banner()

    # Verification de l'integrite du projet · le modele final doit exister.
    final_model = PROJECT_ROOT / "models" / "final_model.joblib"
    if not final_model.exists():
        print("[ERROR] Modele final introuvable.")
        print(f"        Attendu · {final_model}")
        print()
        print("Lancer la pipeline d'entrainement avant ·")
        print("  python scripts/01_generate_dataset.py")
        print("  python scripts/03_train_models.py")
        input("\nAppuyez sur Entree pour quitter...")
        return 1

    # Demarrage des deux serveurs.
    api_proc = _start_uvicorn()
    print("[1/2] API FastAPI demarree (PID %d)" % api_proc.pid)

    streamlit_proc = _start_streamlit()
    print("[2/2] Dashboard Streamlit demarre (PID %d)" % streamlit_proc.pid)
    print()

    # Attente que l'API soit pret avant d'ouvrir le navigateur.
    print("Attente du demarrage des serveurs...")
    if _wait_for_port("127.0.0.1", 8000, timeout=30):
        print("  API   prete sur :8000")
    else:
        print("  [WARN] API pas encore prete apres 30s, on continue.")

    if _wait_for_port("127.0.0.1", 8501, timeout=45):
        print("  Dashboard pret sur :8501")
    else:
        print("  [WARN] Dashboard pas encore pret apres 45s, on continue.")

    # Ouvre le navigateur sur le dashboard.
    print()
    print("Ouverture du navigateur sur http://127.0.0.1:8501")
    webbrowser.open("http://127.0.0.1:8501")

    print()
    print("Services en cours d'execution. Ctrl+C pour arreter.")
    print()

    # Boucle d'attente · on garde le processus vivant tant que les enfants
    # tournent. Au Ctrl+C / fermeture, on termine proprement.
    try:
        while True:
            time.sleep(2)
            if api_proc.poll() is not None:
                print("[WARN] L'API s'est arretee. Code:", api_proc.returncode)
                break
            if streamlit_proc.poll() is not None:
                print("[WARN] Le Dashboard s'est arrete. Code:", streamlit_proc.returncode)
                break
    except KeyboardInterrupt:
        print()
        print("Arret demande par l'utilisateur...")

    # Termination des sous-processus.
    for proc, label in ((api_proc, "API"), (streamlit_proc, "Dashboard")):
        if proc.poll() is None:
            print(f"  Arret du {label}...")
            try:
                if os.name == "nt":
                    proc.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    proc.terminate()
                proc.wait(timeout=8)
            except Exception:
                proc.kill()

    print("Au revoir.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
