# -*- coding: utf-8 -*-
"""Keep-alive anti-endormissement pour le free tier Render.

Render suspend un service free apres ~15 minutes sans trafic HTTP entrant
(cold start ~50s a la requete suivante). Ce module lance un thread daemon
qui pinge la propre URL publique du service toutes les 10 minutes : le ping
passe par le load balancer Render et compte comme trafic entrant.

Activation automatique uniquement sur Render : la variable d'environnement
`RENDER_EXTERNAL_URL` est injectee par la plateforme. En local, en CI ou en
mode desktop, elle est absente et le module ne fait rien.

Kill-switch : poser `KEEPALIVE_DISABLED=1` sur le service pour couper le
ping sans redeployer de code (utile pour re-laisser dormir le service et
economiser les 750 h/mois d'instance du free tier).

Zero dependance : urllib + threading (stdlib), pas de httpx/requests requis.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import urllib.request

logger = logging.getLogger("keepalive")

# 10 min < 15 min (seuil d'endormissement Render) avec de la marge.
PING_INTERVAL_SECONDS = 600
PING_PATH = "/health"


def start_keepalive() -> bool:
    """Demarre le thread de self-ping si le service tourne sur Render.

    Returns:
        True si le thread a ete demarre, False sinon (hors Render ou
        explicitement desactive).
    """
    if os.environ.get("KEEPALIVE_DISABLED"):
        logger.info("Keep-alive desactive (KEEPALIVE_DISABLED).")
        return False

    base_url = os.environ.get("RENDER_EXTERNAL_URL", "").rstrip("/")
    if not base_url:
        # Pas sur Render (local / CI / desktop) : inutile de se pinger.
        return False

    ping_url = f"{base_url}{PING_PATH}"

    def _ping_loop() -> None:
        while True:
            time.sleep(PING_INTERVAL_SECONDS)
            try:
                with urllib.request.urlopen(ping_url, timeout=30) as response:
                    logger.info("Keep-alive ping %s -> HTTP %s", ping_url, response.status)
            except Exception as exc:  # reseau instable : on retente au tour suivant
                logger.warning("Keep-alive ping echoue (%s), retry dans %ss.", exc, PING_INTERVAL_SECONDS)

    thread = threading.Thread(target=_ping_loop, daemon=True, name="render-keepalive")
    thread.start()
    logger.info("Keep-alive demarre : ping %s toutes les %ss.", ping_url, PING_INTERVAL_SECONDS)
    return True
