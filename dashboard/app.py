# -*- coding: utf-8 -*-
"""Dashboard Streamlit · interface décisionnelle responsable maintenance.

Ce dashboard est l'outil opérationnel produit pour transformer les
prédictions du modèle final en décisions actionnables. Il couvre les
6 fonctions exigées par le sujet ·

  1. Visualisation des distributions des capteurs.
  2. Analyse des corrélations.
  3. Comparaison des performances des modèles.
  4. Simulation d'un scénario machine.
  5. Obtention d'une prédiction en temps réel.
  6. Analyse des variables les plus influentes.

Style · CSS personnalisé (charte EFREI bleu) injecté via `st.markdown`
avec `unsafe_allow_html=True`. Cette approche est standard dans la
documentation Streamlit pour ré-habiller l'interface par défaut.

Lancement · `streamlit run dashboard/app.py`
"""

from __future__ import annotations

import json
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


def _ensure_streamlit_runtime() -> None:
    """Sort proprement si le module est lancé en bare mode (`python app.py`).

    Streamlit nécessite son propre runtime (`streamlit run`) pour fournir
    le ScriptRunContext. Sans cela, chaque appel à `st.*` génère un
    warning "missing ScriptRunContext" et l'UI ne s'affiche pas.
    """
    from streamlit.runtime import exists as _runtime_exists

    if _runtime_exists():
        return
    print(
        "\n[dashboard] Lance ce dashboard avec Streamlit, pas avec python ·\n"
        f"  streamlit run {Path(__file__).relative_to(PROJECT_ROOT)}\n",
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
                <h1>⚙️ Système Intelligent · Maintenance Prédictive Industrielle</h1>
                <p>Plateforme d'aide à la décision · prédiction de panne 24h, simulation, interprétabilité</p>
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
# Onglet 1 · Vue d'ensemble (KPI + distribution capteurs).
# ---------------------------------------------------------------------------
def tab_overview(df: pd.DataFrame, metrics: list[dict], best_name: str) -> None:
    """KPI principaux + aperçu visuel du parc machine."""
    st.markdown("### Vue d'ensemble du parc industriel")

    # Calcul des KPI métiers · taux de panne, machines actives, etc.
    n_records = len(df)
    n_machines = df["machine_id"].nunique() if "machine_id" in df.columns else 200
    failure_rate = df[TARGET_BINARY].mean() * 100
    avg_rul = df["rul_hours"].mean()
    best_metric = next((m for m in metrics if m["model_name"] == best_name), None)
    best_f1 = best_metric["f1"] if best_metric else 0.0

    cols = st.columns(5)
    with cols[0]:
        render_kpi("Observations", f"{n_records:,}", "lignes capteurs")
    with cols[1]:
        render_kpi("Machines suivies", f"{n_machines}", "parc unique")
    with cols[2]:
        render_kpi(
            "Taux de panne 24h",
            f"{failure_rate:.1f}%",
            "moyenne historique",
            level="alert",
        )
    with cols[3]:
        render_kpi(
            "RUL moyenne",
            f"{avg_rul:.0f}h",
            "durée de vie restante",
            level="warning",
        )
    with cols[4]:
        render_kpi(
            "F1 modèle final",
            f"{best_f1:.3f}",
            f"{best_name}",
            level="success",
        )

    st.markdown("---")
    st.markdown("#### Distribution des principaux capteurs")
    # Sélecteur multi-feature pour comparer 2 distributions côte à côte.
    cols = st.columns(2)
    with cols[0]:
        feat_a = st.selectbox(
            "Capteur A",
            NUMERIC_FEATURES,
            index=NUMERIC_FEATURES.index("vibration_rms"),
        )
        fig = px.histogram(
            df,
            x=feat_a,
            nbins=50,
            color=TARGET_BINARY,
            color_discrete_map={0: "#43A047", 1: "#E53935"},
            barmode="overlay",
            opacity=0.7,
            labels={TARGET_BINARY: "Panne 24h"},
        )
        fig.update_layout(
            height=380,
            margin=dict(t=10, b=10),
            legend_title_text="Classe",
        )
        st.plotly_chart(fig, width="stretch")

    with cols[1]:
        feat_b = st.selectbox(
            "Capteur B",
            NUMERIC_FEATURES,
            index=NUMERIC_FEATURES.index("temperature_motor"),
        )
        fig = px.histogram(
            df,
            x=feat_b,
            nbins=50,
            color=TARGET_BINARY,
            color_discrete_map={0: "#43A047", 1: "#E53935"},
            barmode="overlay",
            opacity=0.7,
            labels={TARGET_BINARY: "Panne 24h"},
        )
        fig.update_layout(
            height=380,
            margin=dict(t=10, b=10),
            legend_title_text="Classe",
        )
        st.plotly_chart(fig, width="stretch")


# ---------------------------------------------------------------------------
# Onglet 2 · Analyse exploratoire (corrélations + scatter).
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
    """Tableau + barplot comparatif des 4 modèles."""
    st.markdown("### Comparaison des 4 modèles entraînés")

    if not metrics:
        st.warning("Aucune métrique trouvée. Lancer `python scripts/03_train_models.py`.")
        return

    df_metrics = pd.DataFrame(metrics)

    # Mise en forme · arrondi à 4 décimales pour les scores.
    display_df = df_metrics.copy()
    score_cols = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    for c in score_cols:
        if c in display_df.columns:
            display_df[c] = display_df[c].round(4)
    if "fit_time_s" in display_df.columns:
        display_df["fit_time_s"] = display_df["fit_time_s"].round(2)
    if "predict_time_ms" in display_df.columns:
        display_df["predict_time_ms"] = display_df["predict_time_ms"].round(4)

    # Tableau interactif Streamlit · tri/filtre natifs.
    st.dataframe(
        display_df,
        width="stretch",
        hide_index=True,
    )

    # Barplot horizontal pour faciliter la comparaison visuelle.
    st.markdown("#### Comparaison visuelle (6 métriques)")
    melt = df_metrics.melt(
        id_vars="model_name",
        value_vars=score_cols,
        var_name="metric",
        value_name="score",
    )
    fig = px.bar(
        melt,
        x="metric",
        y="score",
        color="model_name",
        barmode="group",
        color_discrete_sequence=["#1E88E5", "#0D47A1", "#E53935", "#43A047"],
        text=melt["score"].round(3),
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        height=480,
        yaxis_range=[0, 1.05],
        legend_title_text="Modèle",
        margin=dict(t=10, b=10),
    )
    st.plotly_chart(fig, width="stretch")


# ---------------------------------------------------------------------------
# Onglet 4 · Simulateur · prédiction temps réel sur scénario utilisateur.
# ---------------------------------------------------------------------------
def tab_simulator(model, df: pd.DataFrame, best_name: str) -> None:
    """Formulaire interactif · l'utilisateur ajuste les capteurs et obtient
    la prédiction de panne en temps réel."""
    st.markdown("### Simulateur · Scénario machine personnalisé")
    st.caption(
        f"Saisissez les valeurs de capteurs ci-dessous. Le modèle `{best_name}` "
        "renvoie immédiatement la probabilité de panne dans les 24 heures."
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

        do_predict = st.button("⚡ Lancer la prédiction", type="primary")

    with cols[1]:
        st.markdown("#### Résultat")

        if do_predict:
            X_input = pd.DataFrame([{f: inputs[f] for f in ALL_FEATURES}])
            proba = float(model.predict_proba(X_input)[0, 1])
            pred = int(proba >= 0.5)

            # Badge coloré selon le seuil.
            if proba < 0.30:
                level = "badge-success"
                label = "Risque faible"
            elif proba < 0.60:
                level = "badge-warning"
                label = "Risque modéré"
            else:
                level = "badge-alert"
                label = "Risque élevé"

            st.markdown(
                f'<div class="badge {level}">⚙️ {label}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(f"### Probabilité de panne 24h · **{proba*100:.1f}%**")

            # Jauge plotly pour visualiser le score.
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
                            {"range": [0, 30], "color": "#C8E6C9"},
                            {"range": [30, 60], "color": "#FFE0B2"},
                            {"range": [60, 100], "color": "#FFCDD2"},
                        ],
                        "threshold": {
                            "line": {"color": "#E53935", "width": 4},
                            "thickness": 0.75,
                            "value": 50,
                        },
                    },
                )
            )
            fig_gauge.update_layout(height=280, margin=dict(t=10, b=10))
            st.plotly_chart(fig_gauge, width="stretch")

            # Recommandation textuelle simple.
            if pred == 1:
                st.warning(
                    "Recommandation · planifier une intervention préventive "
                    "dans les 12-24 prochaines heures."
                )
            else:
                st.success(
                    "Recommandation · aucune action requise. " "Maintenir la surveillance continue."
                )
        else:
            st.info("Configurez les capteurs puis cliquez sur " "**Lancer la prédiction**.")


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
# Entrée principale · orchestre les 5 onglets fonctionnels.
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

    # Création des 5 onglets thématiques.
    tabs = st.tabs(
        [
            "📊 Vue d'ensemble",
            "🔍 EDA",
            "🤖 Modèles",
            "⚙️ Simulateur",
            "💡 Interprétabilité",
        ]
    )

    with tabs[0]:
        tab_overview(df, metrics, best_name)
    with tabs[1]:
        tab_eda(df)
    with tabs[2]:
        tab_models(metrics)
    with tabs[3]:
        tab_simulator(model, df, best_name)
    with tabs[4]:
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
