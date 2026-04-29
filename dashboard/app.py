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
import numpy as np
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
    MODELS_DIR,
    NUMERIC_FEATURES,
    OPERATING_MODES,
    S03_DIR,
    S04_DIR,
    TARGET_BINARY,
)
from src.data_loader import load_dataset  # noqa: E402
from src.preprocessing import get_feature_names  # noqa: E402

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
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap" rel="stylesheet">
<style>
    /* =====================================================================
       Design tokens · charte EFREI premium
       Une seule source de vérité pour les couleurs, ombres, radii, espacement.
       Inspiration · Apple HIG, Linear, Vercel (sobriété + densité d'information).
    ====================================================================== */
    :root {
        --ef-primary: #1E88E5;
        --ef-primary-deep: #0D47A1;
        --ef-primary-soft: #E3F2FD;
        --ef-success: #10B981;
        --ef-success-soft: #D1FAE5;
        --ef-warning: #F59E0B;
        --ef-warning-soft: #FEF3C7;
        --ef-danger: #EF4444;
        --ef-danger-soft: #FEE2E2;

        --ef-bg: #F8FAFC;
        --ef-surface: #FFFFFF;
        --ef-surface-2: #F1F5F9;
        --ef-border: #E2E8F0;
        --ef-border-strong: #CBD5E1;
        --ef-text: #0F172A;
        --ef-text-soft: #475569;
        --ef-text-muted: #94A3B8;

        --ef-radius-sm: 10px;
        --ef-radius-md: 14px;
        --ef-radius-lg: 20px;
        --ef-radius-xl: 28px;

        --ef-shadow-sm: 0 1px 2px rgba(15, 23, 42, 0.04);
        --ef-shadow-md: 0 4px 12px rgba(15, 23, 42, 0.06), 0 1px 3px rgba(15, 23, 42, 0.04);
        --ef-shadow-lg: 0 12px 28px rgba(15, 23, 42, 0.10), 0 4px 8px rgba(15, 23, 42, 0.04);
        --ef-shadow-blue: 0 8px 24px rgba(30, 136, 229, 0.18);

        --ef-easing: cubic-bezier(0.32, 0.72, 0, 1);
        --ef-font-sans: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        --ef-font-mono: 'JetBrains Mono', 'SF Mono', Menlo, Consolas, monospace;
    }

    /* =====================================================================
       Reset · base typographique et fond
    ====================================================================== */
    html, body {
        color: var(--ef-text) !important;
        background-color: var(--ef-bg) !important;
        font-family: var(--ef-font-sans);
        font-feature-settings: 'cv11', 'ss01', 'ss03';
        -webkit-font-smoothing: antialiased;
    }
    .stApp { background-color: var(--ef-bg) !important; }
    .main .block-container {
        padding-top: 1.4rem;
        padding-bottom: 4rem;
        max-width: 1320px;
    }
    .main .block-container p,
    .main .block-container li,
    .main .block-container span,
    .main .block-container label,
    .main .block-container div { color: var(--ef-text); }
    .main .block-container h1 {
        color: var(--ef-primary-deep) !important;
        font-weight: 800;
        letter-spacing: -0.025em;
    }
    .main .block-container h2,
    .main .block-container h3 {
        color: var(--ef-primary-deep) !important;
        font-weight: 700;
        letter-spacing: -0.015em;
    }
    .main .block-container h4 {
        color: var(--ef-text) !important;
        font-weight: 700;
    }
    /* Le header gradient garde son texte blanc · exception explicite. */
    .main-header, .main-header * { color: #FFFFFF !important; }

    /* =====================================================================
       Header principal · vague de bleu EFREI avec halo lumineux
    ====================================================================== */
    .main-header {
        position: relative;
        background:
            radial-gradient(circle at top right, rgba(255,255,255,0.18) 0%, transparent 55%),
            linear-gradient(135deg, #1E88E5 0%, #0D47A1 100%);
        padding: 32px 40px;
        border-radius: var(--ef-radius-xl);
        color: #FFFFFF;
        margin-bottom: 32px;
        box-shadow: var(--ef-shadow-blue);
        overflow: hidden;
    }
    .main-header::after {
        content: '';
        position: absolute;
        right: -120px; top: -120px;
        width: 320px; height: 320px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(255,255,255,0.10) 0%, transparent 70%);
        pointer-events: none;
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.05rem;
        font-weight: 800;
        letter-spacing: -0.035em;
        line-height: 1.15;
    }
    .main-header p {
        margin: 8px 0 0 0;
        opacity: 0.95;
        font-size: 1.05rem;
        font-weight: 500;
    }
    .main-header .authors {
        font-size: 0.86rem;
        opacity: 0.82;
        margin-top: 6px;
        font-weight: 500;
        letter-spacing: 0.01em;
    }

    /* =====================================================================
       Cartes KPI · ombre douce, hover lift, valeur en JetBrains Mono
    ====================================================================== */
    .kpi-card {
        background: var(--ef-surface);
        padding: 20px 22px;
        border-radius: var(--ef-radius-md);
        border: 1px solid var(--ef-border);
        border-left: 4px solid var(--ef-primary);
        box-shadow: var(--ef-shadow-sm);
        height: 100%;
        position: relative;
        transition: transform 0.25s var(--ef-easing),
                    box-shadow 0.25s var(--ef-easing),
                    border-color 0.25s var(--ef-easing);
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: var(--ef-shadow-md);
        border-color: var(--ef-border-strong);
    }
    .kpi-card.alert { border-left-color: var(--ef-danger); }
    .kpi-card.success { border-left-color: var(--ef-success); }
    .kpi-card.warning { border-left-color: var(--ef-warning); }
    .kpi-card .kpi-label {
        font-size: 0.74rem;
        color: var(--ef-text-soft);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
    }
    .kpi-card .kpi-value {
        font-family: var(--ef-font-mono);
        font-size: 1.95rem;
        font-weight: 700;
        color: var(--ef-primary-deep);
        margin-top: 6px;
        line-height: 1.05;
        letter-spacing: -0.02em;
        font-variant-numeric: tabular-nums;
    }
    .kpi-card.alert   .kpi-value { color: var(--ef-danger); }
    .kpi-card.success .kpi-value { color: var(--ef-success); }
    .kpi-card.warning .kpi-value { color: var(--ef-warning); }
    .kpi-card .kpi-sub {
        font-size: 0.78rem;
        color: var(--ef-text-muted);
        margin-top: 6px;
        font-weight: 500;
    }

    /* =====================================================================
       Badges de prédiction · pastille glassmorphism
    ====================================================================== */
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 14px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.86rem;
        letter-spacing: 0.02em;
        backdrop-filter: blur(8px);
    }
    .badge-success { background: var(--ef-success-soft); color: #065F46; border: 1px solid var(--ef-success); }
    .badge-warning { background: var(--ef-warning-soft); color: #92400E; border: 1px solid var(--ef-warning); }
    .badge-alert   { background: var(--ef-danger-soft);  color: #991B1B; border: 1px solid var(--ef-danger); }

    /* =====================================================================
       Sidebar · branding EFREI subtil, séparateurs nets
    ====================================================================== */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #FFFFFF 0%, var(--ef-surface-2) 100%) !important;
        border-right: 1px solid var(--ef-border);
    }
    section[data-testid="stSidebar"] *,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div { color: var(--ef-text) !important; }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: var(--ef-primary-deep) !important;
        font-weight: 700 !important;
        letter-spacing: -0.015em;
    }
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] hr {
        border-color: var(--ef-border);
    }

    /* =====================================================================
       Onglets · indicateur slide animé
    ====================================================================== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        border-bottom: 1px solid var(--ef-border);
        padding-bottom: 0;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        font-weight: 600;
        color: var(--ef-text-soft);
        padding: 12px 20px;
        border-radius: var(--ef-radius-sm) var(--ef-radius-sm) 0 0;
        transition: color 0.2s var(--ef-easing), background 0.2s var(--ef-easing);
        font-size: 0.95rem;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--ef-primary-deep);
        background: var(--ef-surface-2);
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: var(--ef-primary) !important;
        border-bottom: 3px solid var(--ef-primary);
        background: transparent;
    }

    /* =====================================================================
       Inputs · radius cohérent, focus EFREI ring
    ====================================================================== */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div,
    .stMultiSelect > div > div {
        border-radius: var(--ef-radius-sm) !important;
        border-color: var(--ef-border) !important;
        transition: border-color 0.2s, box-shadow 0.2s;
    }
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus {
        border-color: var(--ef-primary) !important;
        box-shadow: 0 0 0 3px rgba(30, 136, 229, 0.15) !important;
    }

    /* Sliders · pouce EFREI avec halo */
    .stSlider [data-baseweb="slider"] [role="slider"] {
        background: var(--ef-primary) !important;
        border-color: var(--ef-primary) !important;
        box-shadow: 0 0 0 6px rgba(30, 136, 229, 0.12) !important;
    }
    .stSlider [data-baseweb="slider"] > div:nth-child(2) > div {
        background: var(--ef-primary) !important;
    }

    /* =====================================================================
       Boutons · primary gradient EFREI, secondary outline
    ====================================================================== */
    .stButton > button {
        border-radius: var(--ef-radius-sm);
        font-weight: 600;
        transition: transform 0.15s var(--ef-easing), box-shadow 0.2s var(--ef-easing);
        letter-spacing: 0.01em;
    }
    .stButton > button:hover { transform: translateY(-1px); }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #1E88E5 0%, #0D47A1 100%);
        color: white;
        border: none;
        padding: 10px 24px;
        box-shadow: 0 4px 12px rgba(30, 136, 229, 0.25);
    }
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 8px 20px rgba(30, 136, 229, 0.35);
    }

    /* Métriques natives Streamlit · style cohérent KPI */
    [data-testid="stMetric"] {
        background: var(--ef-surface);
        border: 1px solid var(--ef-border);
        border-radius: var(--ef-radius-md);
        padding: 18px 20px;
        box-shadow: var(--ef-shadow-sm);
        transition: box-shadow 0.25s var(--ef-easing);
    }
    [data-testid="stMetric"]:hover { box-shadow: var(--ef-shadow-md); }
    [data-testid="stMetricValue"] {
        font-family: var(--ef-font-mono) !important;
        font-variant-numeric: tabular-nums;
        color: var(--ef-primary-deep) !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricLabel"] {
        color: var(--ef-text-soft) !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 0.74rem !important;
    }

    /* Tables · zebrage subtil, header EFREI */
    .stDataFrame, [data-testid="stTable"] {
        border-radius: var(--ef-radius-md);
        overflow: hidden;
        border: 1px solid var(--ef-border);
        box-shadow: var(--ef-shadow-sm);
    }
    .stDataFrame thead tr th {
        background: var(--ef-surface-2) !important;
        color: var(--ef-primary-deep) !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        font-size: 0.78rem !important;
    }

    /* Alerts (info/warning/error/success) · radius + ombre */
    [data-testid="stAlert"] {
        border-radius: var(--ef-radius-md);
        border: 1px solid var(--ef-border);
        box-shadow: var(--ef-shadow-sm);
    }

    /* Expanders · bord doux et hover */
    [data-testid="stExpander"] {
        border-radius: var(--ef-radius-md);
        border: 1px solid var(--ef-border) !important;
        background: var(--ef-surface);
        box-shadow: var(--ef-shadow-sm);
        transition: box-shadow 0.25s var(--ef-easing);
    }
    [data-testid="stExpander"]:hover { box-shadow: var(--ef-shadow-md); }

    /* Code blocks · monospace JetBrains */
    code, pre, .stCode {
        font-family: var(--ef-font-mono) !important;
        font-feature-settings: 'liga' 0;
    }

    /* Footer minimaliste */
    .footer {
        text-align: center;
        padding: 24px;
        color: var(--ef-text-muted);
        font-size: 0.82rem;
        margin-top: 56px;
        border-top: 1px solid var(--ef-border);
        font-weight: 500;
        letter-spacing: 0.01em;
    }

    /* Animation · fade-in subtil au chargement */
    @keyframes ef-fade-in {
        from { opacity: 0; transform: translateY(6px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .main-header, .kpi-card, [data-testid="stMetric"] {
        animation: ef-fade-in 0.45s var(--ef-easing) both;
    }

    /* Scrollbar fine et discrète (Webkit) */
    ::-webkit-scrollbar { width: 10px; height: 10px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb {
        background: var(--ef-border-strong);
        border-radius: 999px;
        border: 2px solid var(--ef-bg);
    }
    ::-webkit-scrollbar-thumb:hover { background: var(--ef-text-muted); }
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
        if EFREI_LOGO.exists():
            st.image(str(EFREI_LOGO), width=160)
    with cols[1]:
        st.markdown(
            """
            <div class="main-header">
                <h1>⚙️ Maintenance Prédictive · Pilotage du parc</h1>
                <p>Outil d'aide à la décision · état du parc, plan d'intervention, économies réalisées</p>
                <div class="authors">Adam Beloucif · Emilien Morice · M1 Mastère Data Engineering &amp; IA · EFREI 2025-26</div>
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

    n_at_risk = int((fleet["risk_level"] != "Sain").sum())
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
            color_discrete_map={"OK": "#43A047", "Panne 24h": "#E53935"},
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
        color_discrete_sequence=["#10B981", "#1E88E5", "#0D47A1", "#E53935"],
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
                        "bar": {"color": "#0D47A1"},
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
        if EFREI_LOGO.exists():
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
