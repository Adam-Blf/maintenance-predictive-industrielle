# -*- coding: utf-8 -*-
"""Wrapper CodeCarbon · mesure de l'empreinte carbone des entrainements.

Le sujet impose explicitement (C4.3 du RNCP40875) d'evaluer le
*"degré d'écoresponsabilité"* des modeles. CodeCarbon fournit une
estimation du CO2 emis basee sur la consommation electrique de la
machine et l'intensite carbone du mix energetique national (defaut
France · ~80 gCO2/kWh, parmi les plus bas au monde grace au nucleaire).
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

try:
    from codecarbon import EmissionsTracker

    CODECARBON_AVAILABLE = True
except Exception:  # pragma: no cover - dependance optionnelle
    CODECARBON_AVAILABLE = False
    EmissionsTracker = None  # type: ignore


@contextmanager
def track_emissions(label: str, output_dir: Path):
    """Context manager qui mesure les emissions CO2 d'un bloc de code.

    Usage ·
        with track_emissions("train_random_forest", REPORTS_DIR) as t:
            model.fit(X, y)
        # t.final_emissions = grammes de CO2 equivalent
    """
    if not CODECARBON_AVAILABLE:
        # Fallback silencieux · si CodeCarbon n'est pas dispo on yield un
        # objet stub avec final_emissions=0 pour que le code consommateur
        # reste fonctionnel.
        class _Stub:
            final_emissions = 0.0

        yield _Stub()
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    tracker = EmissionsTracker(
        project_name=label,
        output_dir=str(output_dir),
        log_level="error",  # silencieux en console
        save_to_file=True,
        country_iso_code="FRA",  # France · ~80 gCO2/kWh
        tracking_mode="machine",
    )
    tracker.start()
    try:
        yield tracker
    finally:
        try:
            tracker.stop()
        except Exception:
            pass
