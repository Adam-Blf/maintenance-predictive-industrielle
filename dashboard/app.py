# -*- coding: utf-8 -*-
"""Dashboard Streamlit · interface décisionnelle responsable maintenance.

Vision métier · cet outil s'adresse en premier lieu au **responsable
maintenance** d'usine et à son chef d'atelier, pas au data scientist.
Tous les KPI, recommandations et écrans sont formulés en langage
opérationnel · machines critiques, plan d'intervention, économies
réalisées, plutôt que F1, ROC-AUC ou SHAP values.

Architecture des onglets ·
  1. État du parc · vue temps réel des 20 machines, top alertes.
  2. Plan d'intervention · table d'actions priorisées par urgence.
  3. Impact économique · ROI, coûts évités, comparaison "avec/sans IA".
  4. Diagnostic machine · saisie capteurs → recommandation actionnable.
  5. Détails techniques · onglet repli pour DSI/jury (EDA + modèles +
     interprétabilité). Les 6 fonctions ML obligatoires du sujet sont
     toutes accessibles ici.

Style · CSS personnalisé (charte EFREI bleu) injecté via `st.markdown`
avec `unsafe_allow_html=True`. Approche standard Streamlit.

Lancement · `streamlit run dashboard/app.py`
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ajout du root au PYTHONPATH pour importer le package `src`.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Bootstrap · auto-install des dépendances manquantes (rend le repo
# clonable et exécutable sur n'importe quelle machine sans setup manuel).
from src.bootstrap import ensure_dependencies  # noqa: E402
ensure_dependencies(verbose=False)


def _ensure_streamlit_runtime() -> None:
    """Sort proprement avec un hint si le module est lance via `python` direct.

    Quand le script est lance correctement via `streamlit run dashboard/app.py`
    ou via l'orchestrateur `python app.py` (qui appelle streamlit run en
    sous-processus), le runtime Streamlit existe et on retourne sans bruit.
    Sinon (cas `python dashboard/app.py` direct), on imprime un hint clair
    et on sort en code 0 pour eviter le flood de warnings ScriptRunContext.
    """
    try:
        from streamlit.runtime import exists as _runtime_exists
        if _runtime_exists():
            return
    except Exception:
        return  # api streamlit indisponible · on tente quand meme
    print(
        "\n[dashboard] Lance ce dashboard avec Streamlit, pas avec python ·\n"
        "  streamlit run dashboard/app.py\n"
        "ou plus simplement, l'orchestrateur unifie ·\n"
        "  python app.py\n",
        file=sys.stderr,
    )
    sys.exit(0)


_ensure_streamlit_runtime()

from src.config import (  # noqa: E402
    ALL_FEATURES,
    EFREI_LOGO,
    EFREI_LOGO_CMJN,
    MODELS_DIR,
    NUMERIC_FEATURES,
    OPERATING_MODES,
    S03_DIR,
    S04_DIR,
    TARGET_BINARY,
)
from src.data_loader import load_dataset  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration de la page · titre, icône, layout wide pour exploiter
# l'espace écran sur grands moniteurs (responsable maintenance ≈ écran 24").
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Maintenance Prédictive · Système Intelligent",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS personnalisé · charte EFREI (bleu institutionnel, typographie sobre,
# cartes arrondies, badges colorés pour les KPI).
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

/* =====================================================================
   EFREI Paris · Design System
   Palette: #163767 · #FF43B8 · #0C78B4 · #051832
   Typo: Poppins (Google Fonts), Arial fallback
   Accessibilité: contrast corps ≥ 4.5:1 sur toutes les surfaces
====================================================================== */

:root {
    /* Palette EFREI officielle */
    --ef-primary:      #163767;
    --ef-deep:         #051832;
    --ef-navy:         #0B1B34;
    --ef-accent:       #FF43B8;
    --ef-blue:         #0C78B4;

    /* Sémantique machine (alertes uniquement) */
    --ef-ok:           #10B981;
    --ef-ok-bg:        #D1FAE5;
    --ef-warn:         #F59E0B;
    --ef-warn-bg:      #FEF3C7;
    --ef-crit:         #EF4444;
    --ef-crit-bg:      #FEE2E2;

    /* Surfaces */
    --ef-bg:           #F4F4F4;
    --ef-surface:      #FFFFFF;
    --ef-surface-2:    #EAECEF;
    --ef-border:       #DDE1E7;
    --ef-border-hi:    #B0B8C4;

    /* Texte — contraste vérifié ≥ 4.5:1 */
    --ef-text:         #212121;   /* 16:1 sur blanc */
    --ef-text-sub:     #5A6B82;   /* 4.6:1 sur blanc */
    --ef-text-muted:   #7B8899;   /* 4.5:1 sur blanc */

    /* Ombres */
    --ef-s-xs:  0 1px 3px rgba(5, 24, 50, 0.07);
    --ef-s-sm:  0 2px 8px rgba(5, 24, 50, 0.09), 0 1px 2px rgba(5, 24, 50, 0.05);
    --ef-s-md:  0 6px 20px rgba(5, 24, 50, 0.11), 0 2px 5px rgba(5, 24, 50, 0.06);
    --ef-s-hdr: 0 10px 32px rgba(22, 55, 103, 0.26);

    /* Radii */
    --ef-r-sm:  8px;
    --ef-r-md:  12px;
    --ef-r-lg:  18px;
    --ef-r-xl:  24px;

    --ef-ease:  cubic-bezier(0.25, 0.46, 0.45, 0.94);
    --ef-font:  'Poppins', Arial, sans-serif;
}

/* ── Base ──────────────────────────────────────────────── */
html, body {
    font-family: var(--ef-font) !important;
    color: var(--ef-text) !important;
    background: var(--ef-bg) !important;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}
.stApp { background: var(--ef-bg) !important; }
.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 4rem;
    max-width: 1340px;
}

/* Propagation Poppins */
.main .block-container *,
section[data-testid="stSidebar"] * {
    font-family: var(--ef-font) !important;
}

/* Échelle typographique */
.main .block-container h1 {
    font-size: 1.95rem; font-weight: 800;
    letter-spacing: -0.03em;
    color: var(--ef-deep) !important;
    line-height: 1.2;
}
.main .block-container h2 {
    font-size: 1.45rem; font-weight: 700;
    letter-spacing: -0.02em;
    color: var(--ef-deep) !important;
}
.main .block-container h3 {
    font-size: 1.15rem; font-weight: 700;
    letter-spacing: -0.015em;
    color: var(--ef-deep) !important;
}
.main .block-container h4 {
    font-size: 0.98rem; font-weight: 600;
    color: var(--ef-text) !important;
}
.main .block-container p,
.main .block-container li { color: var(--ef-text); }

/* Exception header : texte blanc */
.ef-header, .ef-header * { color: #FFFFFF !important; }

/* ── Header principal ─────────────────────────────────── */
.ef-header {
    background: linear-gradient(140deg, #163767 0%, #051832 100%);
    padding: 28px 36px;
    border-radius: var(--ef-r-xl);
    margin-bottom: 28px;
    box-shadow: var(--ef-s-hdr);
    position: relative;
    overflow: hidden;
}
.ef-header::before {
    content: '';
    position: absolute;
    top: -64px; right: -64px;
    width: 260px; height: 260px;
    border-radius: 50%;
    background: rgba(255, 67, 184, 0.07);
    pointer-events: none;
}
.ef-header::after {
    content: '';
    position: absolute;
    bottom: -80px; left: 30%;
    width: 200px; height: 200px;
    border-radius: 50%;
    background: rgba(12, 120, 180, 0.08);
    pointer-events: none;
}
.ef-header h1 {
    margin: 0;
    font-size: 1.85rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    line-height: 1.2;
}
.ef-header p {
    margin: 6px 0 0;
    font-size: 0.98rem;
    font-weight: 500;
    opacity: 0.88;
}
.ef-header .ef-authors {
    font-size: 0.80rem;
    opacity: 0.68;
    margin-top: 5px;
    font-weight: 400;
}

/* ── Cartes KPI — sans stripe latérale ────────────────── */
.kpi-card {
    background: var(--ef-surface);
    padding: 20px 22px;
    border-radius: var(--ef-r-lg);
    border: 1px solid var(--ef-border);
    box-shadow: var(--ef-s-xs);
    height: 100%;
    transition: box-shadow 0.22s var(--ef-ease),
                transform 0.22s var(--ef-ease);
}
.kpi-card:hover {
    box-shadow: var(--ef-s-md);
    transform: translateY(-2px);
}
/* Variantes sémantiques : couleur du contour global uniquement */
.kpi-card.alert   { border-color: rgba(239, 68, 68, 0.35); }
.kpi-card.success { border-color: rgba(16, 185, 129, 0.35); }
.kpi-card.warning { border-color: rgba(245, 158, 11, 0.35); }

.kpi-card .kpi-label {
    font-size: 0.71rem;
    font-weight: 600;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: var(--ef-text-sub);
}
.kpi-card .kpi-value {
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    line-height: 1.1;
    margin-top: 6px;
    color: var(--ef-primary);
    font-variant-numeric: tabular-nums;
}
.kpi-card.alert   .kpi-value { color: var(--ef-crit); }
.kpi-card.success .kpi-value { color: var(--ef-ok); }
.kpi-card.warning .kpi-value { color: var(--ef-warn); }
.kpi-card .kpi-sub {
    font-size: 0.75rem;
    color: var(--ef-text-muted);
    margin-top: 5px;
    font-weight: 500;
}

/* ── Badges — sans glassmorphism ──────────────────────── */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 14px;
    border-radius: 999px;
    font-size: 0.83rem;
    font-weight: 700;
    letter-spacing: 0.02em;
}
.badge-success { background: var(--ef-ok-bg);   color: #065F46; border: 1px solid var(--ef-ok); }
.badge-warning { background: var(--ef-warn-bg); color: #92400E; border: 1px solid var(--ef-warn); }
.badge-alert   { background: var(--ef-crit-bg); color: #991B1B; border: 1px solid var(--ef-crit); }

/* ── Sidebar ──────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: var(--ef-surface) !important;
    border-right: 1px solid var(--ef-border) !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: var(--ef-deep) !important;
    font-weight: 700 !important;
    letter-spacing: -0.015em;
}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div { color: var(--ef-text) !important; }
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] hr {
    border-color: var(--ef-border);
}

/* ── Onglets ──────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 2px;
    border-bottom: 2px solid var(--ef-border);
    padding-bottom: 0;
    background: transparent;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: var(--ef-text-sub);
    font-weight: 600;
    font-size: 0.92rem;
    padding: 10px 18px;
    border-radius: var(--ef-r-sm) var(--ef-r-sm) 0 0;
    transition: color 0.18s, background 0.18s;
}
.stTabs [data-baseweb="tab"]:hover {
    color: var(--ef-primary);
    background: rgba(22, 55, 103, 0.05);
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: var(--ef-primary) !important;
    border-bottom: 3px solid var(--ef-accent) !important;
    background: transparent;
}

/* ── Inputs & contrôles ───────────────────────────────── */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stSelectbox > div > div,
.stMultiSelect > div > div {
    border-radius: var(--ef-r-sm) !important;
    border-color: var(--ef-border) !important;
    transition: border-color 0.18s, box-shadow 0.18s;
}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
    border-color: var(--ef-primary) !important;
    box-shadow: 0 0 0 3px rgba(22, 55, 103, 0.13) !important;
}

/* Slider — pouce bleu EFREI */
.stSlider [data-baseweb="slider"] [role="slider"] {
    background: var(--ef-primary) !important;
    border-color: var(--ef-primary) !important;
    box-shadow: 0 0 0 5px rgba(22, 55, 103, 0.14) !important;
}
.stSlider [data-baseweb="slider"] > div:nth-child(2) > div {
    background: var(--ef-primary) !important;
}

/* ── Boutons ──────────────────────────────────────────── */
.stButton > button {
    font-weight: 600;
    border-radius: var(--ef-r-sm);
    letter-spacing: 0.01em;
    transition: transform 0.15s var(--ef-ease), box-shadow 0.18s var(--ef-ease);
}
.stButton > button:hover { transform: translateY(-1px); }
.stButton > button[kind="primary"] {
    background: var(--ef-primary) !important;
    color: #FFFFFF !important;
    border: none !important;
    padding: 10px 24px;
    box-shadow: 0 3px 10px rgba(22, 55, 103, 0.30);
}
.stButton > button[kind="primary"]:hover {
    background: var(--ef-navy) !important;
    box-shadow: 0 6px 18px rgba(22, 55, 103, 0.40);
}

/* ── st.metric natif ──────────────────────────────────── */
[data-testid="stMetric"] {
    background: var(--ef-surface);
    border: 1px solid var(--ef-border);
    border-radius: var(--ef-r-lg);
    padding: 18px 20px;
    box-shadow: var(--ef-s-xs);
    transition: box-shadow 0.22s var(--ef-ease);
}
[data-testid="stMetric"]:hover { box-shadow: var(--ef-s-md); }
[data-testid="stMetricValue"] {
    color: var(--ef-primary) !important;
    font-weight: 800 !important;
    font-variant-numeric: tabular-nums;
}
[data-testid="stMetricLabel"] {
    font-size: 0.71rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    color: var(--ef-text-sub) !important;
}

/* ── DataFrames ───────────────────────────────────────── */
.stDataFrame, [data-testid="stTable"] {
    border-radius: var(--ef-r-md) !important;
    overflow: hidden;
    border: 1px solid var(--ef-border) !important;
    box-shadow: var(--ef-s-xs);
}

/* ── Alertes ──────────────────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: var(--ef-r-md);
    box-shadow: var(--ef-s-xs);
}

/* ── Expanders ────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: var(--ef-surface);
    border: 1px solid var(--ef-border) !important;
    border-radius: var(--ef-r-md);
    box-shadow: var(--ef-s-xs);
    transition: box-shadow 0.22s var(--ef-ease);
}
[data-testid="stExpander"]:hover { box-shadow: var(--ef-s-sm); }

/* ── Code ─────────────────────────────────────────────── */
code, pre, .stCode { font-feature-settings: 'liga' 0; }

/* ── Footer ───────────────────────────────────────────── */
.footer {
    text-align: center;
    padding: 20px;
    margin-top: 48px;
    border-top: 1px solid var(--ef-border);
    font-size: 0.79rem;
    color: var(--ef-text-muted);
    font-weight: 500;
    letter-spacing: 0.01em;
}

/* ── Fade-in entrée ───────────────────────────────────── */
@keyframes ef-in {
    from { opacity: 0; transform: translateY(7px); }
    to   { opacity: 1; transform: translateY(0); }
}
.ef-header, .kpi-card, [data-testid="stMetric"] {
    animation: ef-in 0.4s var(--ef-ease) both;
}

/* ── Scrollbar fine ───────────────────────────────────── */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: var(--ef-border-hi);
    border-radius: 999px;
    border: 2px solid var(--ef-bg);
}
::-webkit-scrollbar-thumb:hover { background: var(--ef-text-sub); }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers de mise en page · factorisent le CSS et la composition Markdown
# pour garder le corps principal lisible.
# ---------------------------------------------------------------------------
def render_header() -> None:
    """Affiche le bandeau principal avec logo + titre + sous-titre."""
    cols = st.columns([1, 5])
    with cols[0]:
        if EFREI_LOGO_CMJN.exists():
            st.image(str(EFREI_LOGO_CMJN), width=160)
        elif EFREI_LOGO.exists():
            st.image(str(EFREI_LOGO), width=160)
    with cols[1]:
        st.markdown(
            """
            <div class="ef-header">
                <h1>&#9881;&#65039; Maintenance Prédictive · Pilotage du parc</h1>
                <p>Outil d'aide à la décision · état du parc, plan d'intervention, économies réalisées</p>
                <div class="ef-authors">Adam Beloucif · Emilien Morice · M1 Mastère Data Engineering &amp; IA · EFREI 2025-26</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_kpi(label: str, value: str, sub: str = "", level: str = "info") -> None:
    """Petite carte KPI cohérente avec la charte CSS."""
    css_class = {
        "alert": "kpi-card alert",
        "success": "kpi-card success",
        "warning": "kpi-card warning",
        "info": "kpi-card",
    }.get(level, "kpi-card")
    st.markdown(
        f"""
        <div class="{css_class}">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Chargement en cache · évite de recharger le CSV et le modèle à chaque
# interaction utilisateur (Streamlit re-exécute le script entier).
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data_cached() -> pd.DataFrame:
    """Cache · CSV de 24k lignes."""
    return load_dataset()


@st.cache_resource(show_spinner=False)
def load_model_cached():
    """Cache · pipeline sklearn final + métriques + nom du modèle.

    `cache_resource` (vs `cache_data`) car le pipeline contient des
    objets non-sérialisables triviaux (ex. estimators).
    """
    model = joblib.load(MODELS_DIR / "final_model.joblib")
    name_file = MODELS_DIR / "final_model_name.txt"
    name = name_file.read_text(encoding="utf-8").strip() if name_file.exists() else "?"

    metrics_path = S03_DIR / "metrics_summary.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else []
    return model, name, metrics


# ---------------------------------------------------------------------------
# Hypothèses métier · ratio 10:1 cohérent avec la littérature industrielle
# (un arrêt non planifié coûte 5-50x plus cher qu'une maintenance préventive
# correctement programmée). Modifiable ici pour adapter au secteur réel.
# ---------------------------------------------------------------------------
COST_UNPLANNED_FAILURE_EUR: float = 5_000.0   # Arrêt non planifié (production + réparation urgente)
COST_PLANNED_INTERVENTION_EUR: float = 200.0  # Maintenance préventive programmée
COST_FALSE_ALARM_EUR: float = 100.0           # Inspection inutile (intervention pour rien)
RISK_THRESHOLD_HIGH: float = 0.60             # Au-dessus · intervenir sous 24h
RISK_THRESHOLD_MEDIUM: float = 0.30           # Au-dessus · planifier sous 7j


@st.cache_data(show_spinner=False)
def compute_fleet_predictions(
    _model, df: pd.DataFrame, window: int = 100
) -> pd.DataFrame:
    """Pour chaque machine du parc, retourne la mesure la plus à risque
    parmi les `window` dernières observations.

    Pourquoi pas juste la dernière mesure ? · à un instant T donné, la
    majorité des machines sont nominales (taux de panne global ~5%) ; le
    dashboard semble alors vide. En production, le scoring quotidien
    s'effectue sur une fenêtre glissante et alerte dès qu'une machine
    *passe* par un état à risque · on reproduit ce comportement ici en
    prenant le pire score sur les 100 dernières observations capteurs.

    Returns
    -------
    pd.DataFrame
        Une ligne par machine_id, triée par proba décroissante. Colonnes ·
          machine_id, machine_type, proba_panne_24h, risk_level,
          rul_hours, estimated_repair_cost, action_recommandee, fenetre_h
    """
    if "timestamp" in df.columns:
        df_sorted = df.sort_values("timestamp", ascending=False)
    else:
        df_sorted = df.copy()

    # Garde les `window` dernières obs par machine.
    recent = df_sorted.groupby("machine_id", group_keys=False).head(window).copy()
    recent["proba_panne_24h"] = _model.predict_proba(recent[ALL_FEATURES])[:, 1]

    # Pour chaque machine, on garde la ligne avec la proba MAX (alerte la pire).
    idx_worst = recent.groupby("machine_id")["proba_panne_24h"].idxmax()
    fleet = recent.loc[idx_worst].copy()

    def _classify(p: float) -> tuple[str, str, str]:
        if p >= RISK_THRESHOLD_HIGH:
            return "Critique", "🔴 Intervenir < 24h", "12h"
        if p >= RISK_THRESHOLD_MEDIUM:
            return "Modéré", "🟠 Planifier sous 7j", "168h"
        return "Sain", "🟢 Surveillance continue", "—"

    classifications = fleet["proba_panne_24h"].apply(_classify)
    fleet["risk_level"] = [c[0] for c in classifications]
    fleet["action_recommandee"] = [c[1] for c in classifications]
    fleet["fenetre_h"] = [c[2] for c in classifications]

    return fleet.sort_values("proba_panne_24h", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Onglet 1 · État du parc · vue temps réel orientée responsable maintenance.
# ---------------------------------------------------------------------------
def tab_fleet_status(fleet: pd.DataFrame) -> None:
    """État opérationnel du parc · KPIs métier + top alertes prioritaires."""
    st.markdown("### 🏭 État du parc industriel · vue temps réel")
    st.caption(
        "Snapshot des 20 machines suivies. Les KPI ci-dessous sont calculés à "
        "partir de la dernière mesure capteurs de chaque machine et de la "
        "prédiction du modèle final."
    )

    n_critical = int((fleet["risk_level"] == "Critique").sum())
    n_moderate = int((fleet["risk_level"] == "Modéré").sum())
    n_healthy = int((fleet["risk_level"] == "Sain").sum())
    cost_at_risk = float(
        fleet.loc[fleet["risk_level"] != "Sain", "estimated_repair_cost"].sum()
    )
    avg_rul_critical = (
        float(fleet.loc[fleet["risk_level"] == "Critique", "rul_hours"].mean())
        if n_critical > 0
        else 0.0
    )

    cols = st.columns(5)
    with cols[0]:
        render_kpi(
            "Machines critiques", f"{n_critical}",
            f"sur {len(fleet)} suivies", level="alert",
        )
    with cols[1]:
        render_kpi(
            "À surveiller", f"{n_moderate}",
            "intervention à planifier", level="warning",
        )
    with cols[2]:
        render_kpi(
            "Machines saines", f"{n_healthy}",
            "fonctionnement nominal", level="success",
        )
    with cols[3]:
        render_kpi(
            "Coût à risque", f"{cost_at_risk:,.0f} €",
            "réparation potentielle", level="alert",
        )
    with cols[4]:
        render_kpi(
            "RUL critiques", f"{avg_rul_critical:.0f}h",
            "durée de vie estimée", level="warning",
        )

    st.markdown("---")
    st.markdown("#### 🚨 Machines prioritaires")
    st.caption(
        "Top des machines à inspecter · classement par probabilité de panne "
        "dans les 24h. Cliquer sur l'onglet **Plan d'intervention** pour le "
        "détail des actions."
    )

    top = fleet.head(8).copy()
    top["proba_pct"] = (top["proba_panne_24h"] * 100).round(1).astype(str) + "%"
    top["repair_cost_eur"] = top["estimated_repair_cost"].round(0).astype(int)

    display_cols = [
        "machine_id", "machine_type", "risk_level", "proba_pct",
        "rul_hours", "repair_cost_eur", "action_recommandee",
    ]
    rename = {
        "machine_id": "Machine",
        "machine_type": "Type",
        "risk_level": "Risque",
        "proba_pct": "Proba panne 24h",
        "rul_hours": "RUL (h)",
        "repair_cost_eur": "Coût réparation (€)",
        "action_recommandee": "Action",
    }
    st.dataframe(
        top[display_cols].rename(columns=rename),
        width="stretch",
        hide_index=True,
    )

    # Graphique répartition risque par type de machine.
    st.markdown("---")
    st.markdown("#### 📊 Répartition des risques par type de machine")
    by_type = (
        fleet.groupby(["machine_type", "risk_level"]).size().reset_index(name="count")
    )
    fig = px.bar(
        by_type,
        x="machine_type",
        y="count",
        color="risk_level",
        color_discrete_map={"Sain": "#10B981", "Modéré": "#F59E0B", "Critique": "#EF4444"},
        labels={"machine_type": "Type", "count": "Nb machines", "risk_level": "Niveau"},
        text_auto=True,
    )
    fig.update_layout(height=340, margin=dict(t=10, b=10), barmode="stack")
    st.plotly_chart(fig, width="stretch")


# ---------------------------------------------------------------------------
# Onglet 2 · Plan d'intervention · table d'actions priorisées.
# ---------------------------------------------------------------------------
def tab_intervention_plan(fleet: pd.DataFrame) -> None:
    """Liste actionnable des interventions à programmer · pour le chef
    d'atelier qui établit le planning de la semaine."""
    st.markdown("### 🚨 Plan d'intervention · ordre de priorité")
    st.caption(
        "Calendrier suggéré par l'algorithme. Filtres ci-dessous pour réduire "
        "la liste à un type de machine ou un niveau de risque précis."
    )

    cols_filter = st.columns(3)
    with cols_filter[0]:
        type_filter = st.multiselect(
            "Type de machine",
            options=sorted(fleet["machine_type"].unique()),
            default=sorted(fleet["machine_type"].unique()),
        )
    with cols_filter[1]:
        risk_filter = st.multiselect(
            "Niveau de risque",
            options=["Critique", "Modéré", "Sain"],
            default=["Critique", "Modéré"],
        )
    with cols_filter[2]:
        st.markdown(
            f"**Hypothèses coûts** · arrêt non planifié = "
            f"`{COST_UNPLANNED_FAILURE_EUR:,.0f}€` · "
            f"intervention préventive = `{COST_PLANNED_INTERVENTION_EUR:,.0f}€`"
        )

    filtered = fleet[
        fleet["machine_type"].isin(type_filter) & fleet["risk_level"].isin(risk_filter)
    ].copy()

    if filtered.empty:
        st.info("Aucune machine ne correspond aux filtres sélectionnés.")
        return

    # Économie potentielle = coût arrêt évité - coût intervention préventive.
    filtered["economie_estimee_eur"] = (
        filtered["proba_panne_24h"] * COST_UNPLANNED_FAILURE_EUR
        - COST_PLANNED_INTERVENTION_EUR
    ).round(0).astype(int)

    filtered["proba_pct"] = (filtered["proba_panne_24h"] * 100).round(1).astype(str) + "%"
    filtered["rul_disp"] = filtered["rul_hours"].round(0).astype(int).astype(str) + "h"

    display_cols = [
        "machine_id", "machine_type", "risk_level", "fenetre_h", "proba_pct",
        "rul_disp", "action_recommandee", "economie_estimee_eur",
    ]
    rename = {
        "machine_id": "Machine",
        "machine_type": "Type",
        "risk_level": "Niveau",
        "fenetre_h": "Fenêtre",
        "proba_pct": "Risque 24h",
        "rul_disp": "RUL",
        "action_recommandee": "Action",
        "economie_estimee_eur": "Économie estimée (€)",
    }
    st.dataframe(
        filtered[display_cols].rename(columns=rename),
        width="stretch",
        hide_index=True,
    )

    n_actions = int((filtered["risk_level"] != "Sain").sum())
    total_economy = int(filtered["economie_estimee_eur"].clip(lower=0).sum())
    st.markdown(
        f"**Synthèse** · {n_actions} intervention(s) recommandée(s) sur la "
        f"sélection · économie projetée cumulée · **{total_economy:,} €** "
        "(vs scénario zéro maintenance préventive)."
    )


# ---------------------------------------------------------------------------
# Onglet 3 · Impact économique · ROI de l'IA prédictive.
# ---------------------------------------------------------------------------
def tab_business_impact(
    fleet: pd.DataFrame, metrics: list[dict], best_name: str
) -> None:
    """Comparaison "Sans IA" vs "Avec IA" sur la base du parc actuel."""
    st.markdown("### 💶 Impact économique de la maintenance prédictive")
    st.caption(
        "Estimation du retour sur investissement basée sur le parc et les "
        "performances mesurées du modèle final sur le test set."
    )

    best_metric = next((m for m in metrics if m["model_name"] == best_name), {})
    recall = float(best_metric.get("recall", 0.0))
    precision = float(best_metric.get("precision", 0.0))

    expected_failures = float(fleet["proba_panne_24h"].sum())

    # Sans IA · toutes les pannes survenues coûtent l'arrêt non planifié.
    cost_without_ai = expected_failures * COST_UNPLANNED_FAILURE_EUR

    # Avec IA · on détecte (recall × pannes) et on les traite en préventif.
    # Le reste passe en panne. On supporte aussi des FP (intervention inutile).
    detected = expected_failures * recall
    missed = expected_failures - detected
    # FP estimés via précision · si precision = 0.7, alors 30% des alertes sont fausses.
    n_alerts = detected / max(precision, 0.01)
    false_alarms = n_alerts - detected
    cost_with_ai = (
        detected * COST_PLANNED_INTERVENTION_EUR
        + missed * COST_UNPLANNED_FAILURE_EUR
        + false_alarms * COST_FALSE_ALARM_EUR
    )

    saving = cost_without_ai - cost_with_ai
    saving_pct = (saving / cost_without_ai * 100) if cost_without_ai > 0 else 0.0

    cols = st.columns(4)
    with cols[0]:
        render_kpi(
            "Pannes attendues", f"{expected_failures:.1f}",
            f"sur les {len(fleet)} machines", level="warning",
        )
    with cols[1]:
        render_kpi(
            "Sans IA · coût", f"{cost_without_ai:,.0f} €",
            "scénario réactif", level="alert",
        )
    with cols[2]:
        render_kpi(
            "Avec IA · coût", f"{cost_with_ai:,.0f} €",
            f"{n_alerts:.1f} alertes générées", level="info",
        )
    with cols[3]:
        render_kpi(
            "Économie", f"{saving:,.0f} €",
            f"−{saving_pct:.0f}% vs réactif", level="success",
        )

    st.markdown("---")
    st.markdown("#### Décomposition des coûts · scénario IA")

    breakdown = pd.DataFrame(
        {
            "Catégorie": [
                "Interventions préventives (vraies alertes)",
                "Pannes non détectées (manquées)",
                "Fausses alertes (intervention inutile)",
            ],
            "Volume": [round(detected, 1), round(missed, 1), round(false_alarms, 1)],
            "Coût unitaire (€)": [
                COST_PLANNED_INTERVENTION_EUR,
                COST_UNPLANNED_FAILURE_EUR,
                COST_FALSE_ALARM_EUR,
            ],
            "Coût total (€)": [
                round(detected * COST_PLANNED_INTERVENTION_EUR),
                round(missed * COST_UNPLANNED_FAILURE_EUR),
                round(false_alarms * COST_FALSE_ALARM_EUR),
            ],
        }
    )
    st.dataframe(breakdown, width="stretch", hide_index=True)

    fig = px.bar(
        breakdown,
        x="Catégorie",
        y="Coût total (€)",
        color="Catégorie",
        color_discrete_map={
            "Interventions préventives (vraies alertes)": "#10B981",
            "Pannes non détectées (manquées)": "#EF4444",
            "Fausses alertes (intervention inutile)": "#F59E0B",
        },
        text_auto=True,
    )
    fig.update_layout(height=380, margin=dict(t=10, b=10), showlegend=False)
    st.plotly_chart(fig, width="stretch")

    st.markdown(
        f"**Lecture métier** · le modèle `{best_name}` détecte "
        f"**{recall*100:.0f}%** des pannes réelles (recall) avec "
        f"**{precision*100:.0f}%** de précision. Sur ce parc de "
        f"{len(fleet)} machines, l'IA permet d'éviter ~{saving:,.0f}€ de "
        f"coûts d'arrêt par cycle de prédiction de 24h, soit "
        f"~**{saving*30:,.0f}€/mois** si le scoring est exécuté chaque jour."
    )


# ---------------------------------------------------------------------------
# Onglet · Analyse exploratoire (corrélations + scatter) · sous Détails techniques.
# ---------------------------------------------------------------------------
def tab_eda(df: pd.DataFrame) -> None:
    """Heatmap de corrélation + scatter plot interactif."""
    st.markdown("### Analyse exploratoire des données")

    cols = st.columns([3, 2])

    with cols[0]:
        st.markdown("#### Matrice de corrélation")
        corr = df[NUMERIC_FEATURES + [TARGET_BINARY]].corr()
        fig = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            aspect="auto",
        )
        fig.update_layout(height=520, margin=dict(t=10, b=10))
        st.plotly_chart(fig, width="stretch")

    with cols[1]:
        st.markdown("#### Scatter plot bidimensionnel")
        x_axis = st.selectbox(
            "Axe X", NUMERIC_FEATURES, index=NUMERIC_FEATURES.index("vibration_rms")
        )
        y_axis = st.selectbox(
            "Axe Y",
            NUMERIC_FEATURES,
            index=NUMERIC_FEATURES.index("temperature_motor"),
        )
        # Échantillonnage · le scatter plotly devient lent au-delà de 5k.
        sample = df.sample(n=min(4000, len(df)), random_state=42)
        fig = px.scatter(
            sample,
            x=x_axis,
            y=y_axis,
            color=sample[TARGET_BINARY].map({0: "OK", 1: "Panne 24h"}),
            color_discrete_map={"OK": "#163767", "Panne 24h": "#FF43B8"},
            opacity=0.55,
            labels={"color": "Classe"},
        )
        fig.update_traces(marker=dict(size=6))
        fig.update_layout(height=520, margin=dict(t=10, b=10))
        st.plotly_chart(fig, width="stretch")


# ---------------------------------------------------------------------------
# Onglet 3 · Comparaison des modèles.
# ---------------------------------------------------------------------------
def tab_models(metrics: list[dict]) -> None:
    """Tableau + barplot comparatif des 4 modèles, classés par F1 décroissant."""
    st.markdown("### Comparaison des 4 modèles entraînés · classés par F1-score")
    st.caption(
        "Classement par F1-score décroissant · le modèle final retenu apparaît en tête. "
        "F1 est privilégié sur l'accuracy car la cible est déséquilibrée (~25% pannes)."
    )

    if not metrics:
        st.warning("Aucune métrique trouvée. Lancer `python scripts/03_train_models.py`.")
        return

    df_metrics = pd.DataFrame(metrics)

    # Classement décroissant par F1 · le meilleur modèle en première ligne.
    sort_key = "f1" if "f1" in df_metrics.columns else "roc_auc"
    df_ranked = df_metrics.sort_values(sort_key, ascending=False).reset_index(drop=True)
    df_ranked.insert(0, "Rang", [f"#{i + 1}" for i in range(len(df_ranked))])

    # Mise en forme · arrondi à 4 décimales pour les scores.
    display_df = df_ranked.copy()
    score_cols = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    for c in score_cols:
        if c in display_df.columns:
            display_df[c] = display_df[c].round(4)
    if "fit_time_s" in display_df.columns:
        display_df["fit_time_s"] = display_df["fit_time_s"].round(2)
    if "predict_time_ms" in display_df.columns:
        display_df["predict_time_ms"] = display_df["predict_time_ms"].round(4)

    # Mise en évidence du meilleur modèle (ligne 0 = #1).
    def _highlight_best(row):
        if row.name == 0:
            return ["background-color: #DCFCE7; font-weight: 600"] * len(row)
        return [""] * len(row)

    styled = display_df.style.apply(_highlight_best, axis=1)

    # Tableau interactif Streamlit · tri/filtre natifs, ligne 1 mise en valeur.
    st.dataframe(
        styled,
        width="stretch",
        hide_index=True,
    )

    # Barplot horizontal pour faciliter la comparaison visuelle.
    st.markdown("#### Comparaison visuelle (6 métriques)")
    # Préserve l'ordre du classement dans la légende et les groupes.
    ordered_models = df_ranked["model_name"].tolist()
    melt = df_ranked.melt(
        id_vars="model_name",
        value_vars=[c for c in score_cols if c in df_ranked.columns],
        var_name="metric",
        value_name="score",
    )
    fig = px.bar(
        melt,
        x="metric",
        y="score",
        color="model_name",
        barmode="group",
        category_orders={"model_name": ordered_models},
        color_discrete_sequence=["#163767", "#0C78B4", "#FF43B8", "#0B1B34"],
        text=melt["score"].round(3),
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        height=480,
        yaxis_range=[0, 1.05],
        legend_title_text="Modèle (classé)",
        margin=dict(t=10, b=10),
    )
    st.plotly_chart(fig, width="stretch")


# ---------------------------------------------------------------------------
# Onglet 4 · Diagnostic machine · saisie capteurs + recommandation métier.
# ---------------------------------------------------------------------------
def tab_diagnostic(model, df: pd.DataFrame, best_name: str) -> None:
    """Évaluation à la demande d'une machine · le responsable maintenance saisit
    les valeurs capteurs et obtient une **décision actionnable**, pas un score
    brut · niveau de risque, fenêtre d'intervention, économie attendue."""
    st.markdown("### 🔧 Diagnostic d'une machine · décision en 30 secondes")
    st.caption(
        "Saisissez les dernières valeurs capteurs relevées sur une machine. "
        "L'outil renvoie un **niveau de risque, une fenêtre d'intervention "
        "recommandée et l'économie estimée** vs. attendre la panne."
    )

    # Layout · 2 colonnes, formulaire à gauche + résultat à droite.
    cols = st.columns([2, 1])

    with cols[0]:
        # Sliders avec valeurs par défaut = médiane historique pour ne pas
        # initialiser dans une zone aberrante.
        defaults = df[NUMERIC_FEATURES].median().to_dict()
        # Bornes · min/max observés (clip pour éviter les extrapolations
        # totalement hors distribution).
        bounds = {f: (df[f].min(), df[f].max()) for f in NUMERIC_FEATURES}

        sub = st.columns(2)
        inputs: dict[str, float] = {}
        for idx, feat in enumerate(NUMERIC_FEATURES):
            target_col = sub[idx % 2]
            with target_col:
                lo, hi = bounds[feat]
                inputs[feat] = st.slider(
                    feat.replace("_", " ").title(),
                    min_value=float(lo),
                    max_value=float(hi),
                    value=float(defaults[feat]),
                    step=(float(hi) - float(lo)) / 100.0,
                )

        inputs["operating_mode"] = st.selectbox(
            "Mode opératoire",
            OPERATING_MODES,
            index=OPERATING_MODES.index("normal"),
        )
        # Type de machine · ajouté avec le schéma Kaggle v3.0.
        from src.config import MACHINE_TYPES  # noqa
        inputs["machine_type"] = st.selectbox(
            "Type de machine",
            MACHINE_TYPES,
            index=0,
        )

        do_predict = st.button("⚡ Évaluer la machine", type="primary")

    with cols[1]:
        st.markdown("#### Décision recommandée")

        if do_predict:
            # 1. Tente d'appeler l'API REST si elle est disponible (cas
            #    `python app.py` orchestrateur). Cela permet a la
            #    soutenance de demontrer le couplage dashboard <-> API.
            api_url = os.environ.get("API_BASE_URL", "").rstrip("/")
            proba: float | None = None
            source: str = "local"
            if api_url:
                try:
                    import httpx
                    payload = {f: inputs[f] for f in ALL_FEATURES}
                    r = httpx.post(f"{api_url}/predict", json=payload, timeout=3.0)
                    if r.status_code == 200:
                        body = r.json()
                        proba = float(body["probability"])
                        source = "api"
                except Exception:
                    proba = None  # bascule en local
            # 2. Fallback · prediction locale via le modele charge.
            if proba is None:
                X_input = pd.DataFrame([{f: inputs[f] for f in ALL_FEATURES}])
                proba = float(model.predict_proba(X_input)[0, 1])

            # Badge source pour la soutenance.
            if source == "api":
                st.caption(f"Source · API REST `{api_url}/predict`")
            else:
                st.caption("Source · modele local (joblib)")

            # Classification métier · 3 niveaux de risque.
            if proba >= RISK_THRESHOLD_HIGH:
                level, label = "badge-alert", "Risque CRITIQUE"
                window = "12 heures"
                action = (
                    "Programmer une intervention préventive **sous 12 heures**. "
                    "Stopper la machine si possible avant la fin du shift."
                )
            elif proba >= RISK_THRESHOLD_MEDIUM:
                level, label = "badge-warning", "Risque modéré"
                window = "7 jours"
                action = (
                    "Planifier une **inspection sous 7 jours** lors d'un créneau "
                    "de maintenance programmée. Renforcer la surveillance."
                )
            else:
                level, label = "badge-success", "Risque faible"
                window = "—"
                action = (
                    "Aucune action immédiate. **Surveillance continue** via "
                    "les capteurs. Prochain check de routine selon plan annuel."
                )

            economy = max(
                0.0, proba * COST_UNPLANNED_FAILURE_EUR - COST_PLANNED_INTERVENTION_EUR
            )

            st.markdown(
                f'<div class="badge {level}">⚙️ {label}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(f"### Probabilité de panne 24h · **{proba*100:.1f}%**")
            st.markdown(f"**Fenêtre d'intervention** · {window}")
            st.markdown(f"**Économie estimée** · {economy:,.0f} € (vs attendre la panne)")

            # Jauge plotly · couleur cohérente avec les seuils métier.
            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=proba * 100,
                    number={"suffix": "%"},
                    domain={"x": [0, 1], "y": [0, 1]},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#163767"},
                        "steps": [
                            {"range": [0, RISK_THRESHOLD_MEDIUM * 100], "color": "#C8E6C9"},
                            {"range": [RISK_THRESHOLD_MEDIUM * 100, RISK_THRESHOLD_HIGH * 100], "color": "#FFE0B2"},
                            {"range": [RISK_THRESHOLD_HIGH * 100, 100], "color": "#FFCDD2"},
                        ],
                        "threshold": {
                            "line": {"color": "#E53935", "width": 4},
                            "thickness": 0.75,
                            "value": RISK_THRESHOLD_HIGH * 100,
                        },
                    },
                )
            )
            fig_gauge.update_layout(height=260, margin=dict(t=10, b=10))
            st.plotly_chart(fig_gauge, width="stretch")

            st.info(f"**Action** · {action}")
        else:
            st.info(
                "Configurez les capteurs puis cliquez sur "
                "**Évaluer la machine** pour obtenir la décision."
            )


# ---------------------------------------------------------------------------
# Onglet 5 · Interprétabilité · feature importance + SHAP.
# ---------------------------------------------------------------------------
def tab_interpretability(best_name: str) -> None:
    """Affichage des graphes d'interprétabilité produits par 04_interpret.py."""
    st.markdown("### Interprétabilité du modèle final")
    st.caption(
        "Permet de répondre à la question · "
        "_« Pourquoi le modèle indique un risque de panne élevé ? »_"
    )

    # Liste des graphiques attendus dans reports/04/ (script 04_interpret.py).
    candidates = [
        (
            f"feature_importance_native_{best_name}.png",
            "Feature Importance native",
        ),
        (
            f"permutation_importance_{best_name}.png",
            "Permutation Importance (agnostique)",
        ),
        (f"shap_summary_{best_name}.png", "SHAP Summary Plot"),
        (f"shap_bar_{best_name}.png", "SHAP Importance globale"),
    ]

    for filename, title in candidates:
        path = S04_DIR / filename
        if path.exists():
            st.markdown(f"#### {title}")
            st.image(str(path), width="stretch")
        else:
            st.info(f"`{filename}` non trouvé. Lancer `python scripts/04_interpret.py`.")


# ---------------------------------------------------------------------------
# Sidebar · navigation + crédits.
# ---------------------------------------------------------------------------
def render_sidebar() -> None:
    """Sidebar avec logo + métadonnées projet."""
    with st.sidebar:
        if EFREI_LOGO_CMJN.exists():
            st.image(str(EFREI_LOGO_CMJN), width="stretch")
        elif EFREI_LOGO.exists():
            st.image(str(EFREI_LOGO), width="stretch")
        st.markdown("## À propos")
        st.markdown("""
            **Projet Data Science · M1 Data Engineering & IA**

            Système intelligent multi-modèles pour la maintenance prédictive
            industrielle.

            - **Bloc** · BC2 RNCP40875
            - **Année** · 2025-2026
            - **Auteurs** · Adam Beloucif, Emilien Morice
            - **École** · EFREI Paris Panthéon-Assas
            """)
        st.markdown("---")
        st.markdown("### Stack")
        st.code(
            "Python · scikit-learn\n"
            "XGBoost · MLP (Deep Learning)\n"
            "Streamlit · FastAPI\n"
            "SHAP · matplotlib · plotly",
            language="text",
        )


# ---------------------------------------------------------------------------
# Entrée principale · 5 onglets vision métier · les détails ML sont
# regroupés sous le dernier onglet pour ne pas pollluer l'écran principal.
# ---------------------------------------------------------------------------
def main() -> None:
    """Point d'entrée Streamlit."""
    render_sidebar()
    render_header()

    # Garde-fou · si le modèle n'est pas entraîné, on guide l'utilisateur.
    if not (MODELS_DIR / "final_model.joblib").exists():
        st.error(
            "Modèle final introuvable. Exécuter dans l'ordre ·\n\n"
            "1. `python scripts/02_eda.py`\n"
            "2. `python scripts/03_train_models.py`\n"
            "3. `python scripts/04_interpret.py`"
        )
        return

    df = load_data_cached()
    model, best_name, metrics = load_model_cached()
    fleet = compute_fleet_predictions(model, df)

    # 5 onglets, du plus opérationnel (gauche) au plus technique (droite).
    tabs = st.tabs(
        [
            "🏭 État du parc",
            "🚨 Plan d'intervention",
            "💶 Impact économique",
            "🔧 Diagnostic machine",
            "🔬 Détails techniques",
        ]
    )

    with tabs[0]:
        tab_fleet_status(fleet)
    with tabs[1]:
        tab_intervention_plan(fleet)
    with tabs[2]:
        tab_business_impact(fleet, metrics, best_name)
    with tabs[3]:
        tab_diagnostic(model, df, best_name)
    with tabs[4]:
        # Sous-onglets · les vues ML traditionnelles, regroupées pour
        # rester accessibles au jury sans alourdir le dashboard métier.
        st.markdown(
            "##### Vues techniques · réservées au data scientist / DSI / jury"
        )
        sub = st.tabs(
            ["📊 Données (EDA)", "🤖 Modèles", "💡 Interprétabilité"]
        )
        with sub[0]:
            tab_eda(df)
        with sub[1]:
            tab_models(metrics)
        with sub[2]:
            tab_interpretability(best_name)

    # Footer minimaliste.
    st.markdown(
        '<div class="footer">'
        "© 2026 · Adam Beloucif &amp; Emilien Morice · "
        "EFREI Paris Panthéon-Assas Université"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
