# -*- coding: utf-8 -*-
"""Wrapper CodeCarbon · mesure de l'empreinte carbone des entrainements.

Contexte RNCP / cahier des charges
------------------------------------
Le bloc de compétences BC2 (RNCP40875) impose explicitement au critère
C4.3 d'évaluer le *"degré d'écoresponsabilité"* des choix algorithmiques.
CodeCarbon (https://codecarbon.io) répond à cette exigence en mesurant ·
  - la consommation électrique de la machine pendant l'entraînement
    (via l'API Intel RAPL ou powermetrics),
  - multipliée par l'intensité carbone du mix énergétique du pays
    déclaré (France · ~80 gCO2eq/kWh, l'un des plus bas au monde
    grâce à la prépondérance du nucléaire).

Pourquoi CodeCarbon plutôt qu'un simple chronomètre ?
------------------------------------------------------
Le temps de calcul est un proxy imparfait de la consommation : un GPU
peut consommer 200-500 W pendant quelques secondes, là où un CPU
consomme 20-50 W pendant plusieurs minutes. CodeCarbon mesure les watts
réels, pas un proxy temporel.

Sortie · fichier `emissions.csv` dans `output_dir`, une ligne par
entraînement. Le Brier score et les métriques de performance sont
croisés avec cette donnée pour produire la figure "coût computationnel"
du rapport (cf. `evaluation.plot_training_time_barplot`).
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

    Encapsule le cycle start/stop de CodeCarbon dans un `with` pour
    garantir que le tracker est toujours arrêté, même en cas d'exception
    dans le bloc entraîné.

    Usage
    -----
        with track_emissions("train_random_forest", REPORTS_DIR) as t:
            model.fit(X, y)
        # t.final_emissions contient les grammes de CO2 equivalent emis

    Parameters
    ----------
    label : str
        Nom du projet CodeCarbon (apparaît dans emissions.csv). Doit
        être unique par modèle pour faciliter la comparaison post-hoc.
    output_dir : Path
        Dossier où CodeCarbon écrit son CSV de sortie. Le dossier est
        créé automatiquement s'il n'existe pas.

    Yields
    ------
    EmissionsTracker | _Stub
        L'objet tracker CodeCarbon (ou un stub si CodeCarbon n'est pas
        installé). L'attribut `final_emissions` contient le résultat
        en kgCO2eq après le `with`.

    Notes
    -----
    Le `tracking_mode="machine"` mesure la consommation totale du CPU
    (et GPU si présent), pas seulement le processus Python, ce qui donne
    une borne supérieure conservative plutôt qu'une sous-estimation.
    """
    if not CODECARBON_AVAILABLE:
        # Fallback silencieux · si CodeCarbon n'est pas installé (environnement
        # CI, machine sans accès RAPL), on yield un objet stub avec
        # final_emissions=0 pour que le code consommateur reste fonctionnel
        # sans lever d'ImportError.
        class _Stub:
            final_emissions = 0.0

        yield _Stub()
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    tracker = EmissionsTracker(
        project_name=label,
        output_dir=str(output_dir),
        log_level="error",  # silencieux en console pour ne pas polluer les logs
        save_to_file=True,
        country_iso_code="FRA",  # France · ~80 gCO2eq/kWh (nucléaire dominant)
        tracking_mode="machine",  # mesure CPU + GPU, pas seulement le process Python
    )
    tracker.start()
    try:
        yield tracker
    finally:
        # On englobe stop() dans un try/except pour ne pas masquer une
        # exception levée dans le bloc utilisateur par une erreur CodeCarbon.
        try:
            tracker.stop()
        except Exception:
            pass
