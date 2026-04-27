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

from src.config import (  # noqa: E402
    ALL_FEATURES,
    EFREI_LOGO,
    MODELS_DIR,
    NUMERIC_FEATURES,
    OPERATING_MODES,
    REPORTS_DIR,
    REPORTS_FIGURES_DIR,
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
<style>
    /* Police globale plus moderne que la sans-serif Streamlit par défaut */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Header principal · gradient bleu EFREI vers bleu nuit */
    .main-header {
        background: linear-gradient(135deg, #1E88E5 0%, #0D47A1 100%);
        padding: 28px 36px;
        border-radius: 18px;
        color: white;
        margin-bottom: 28px;
        box-shadow: 0 8px 24px rgba(13, 71, 161, 0.18);
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.0rem;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    .main-header p {
        margin: 6px 0 0 0;
        opacity: 0.92;
        font-size: 1.05rem;
    }
    .main-header .authors {
        font-size: 0.88rem;
        opacity: 0.78;
        margin-top: 4px;
    }

    /* Cartes KPI · bord coloré à gauche selon la sémantique */
    .kpi-card {
        background: white;
        padding: 18px 22px;
        border-radius: 14px;
        border-left: 5px solid #1E88E5;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
        height: 100%;
    }
    .kpi-card.alert { border-left-color: #E53935; }
    .kpi-card.success { border-left-color: #43A047; }
    .kpi-card.warning { border-left-color: #FB8C00; }
    .kpi-card .kpi-label {
        font-size: 0.82rem;
        color: #5F6368;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        font-weight: 600;
    }
    .kpi-card .kpi-value {
        font-size: 1.95rem;
        font-weight: 800;
        color: #0D47A1;
        margin-top: 4px;
        line-height: 1.1;
    }
    .kpi-card .kpi-sub {
        font-size: 0.82rem;
        color: #80868B;
        margin-top: 4px;
    }

    /* Badges de prédiction · pilule colorée */
    .badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.92rem;
        letter-spacing: 0.4px;
    }
    .badge-success { background: #E8F5E9; color: #1B5E20; border: 1px solid #43A047; }
    .badge-warning { background: #FFF3E0; color: #E65100; border: 1px solid #FB8C00; }
    .badge-alert { background: #FFEBEE; color: #B71C1C; border: 1px solid #E53935; }

    /* Sidebar · fond légèrement teinté pour la distinguer */
    section[data-testid="stSidebar"] {
        background: #F4F7FB;
        border-right: 1px solid #DDE5EE;
    }
    section[data-testid="stSidebar"] h2 {
        color: #0D47A1 !important;
        font-weight: 700 !important;
    }

    /* Onglets · ligne de soulignement EFREI */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        border-bottom: 2px solid #E3EAF2;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        font-weight: 600;
        color: #5F6368;
        padding: 10px 18px;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #1E88E5;
        border-bottom: 3px solid #1E88E5;
    }

    /* Footer minimaliste */
    .footer {
        text-align: center;
        padding: 18px;
        color: #80868B;
        font-size: 0.85rem;
        margin-top: 40px;
        border-top: 1px solid #E3EAF2;
    }

    /* Boutons primaires · couleur EFREI */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #1E88E5 0%, #0D47A1 100%);
        color: white;
        border: none;
        font-weight: 700;
        padding: 8px 22px;
        border-radius: 10px;
    }
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

    metrics_path = REPORTS_DIR / "metrics_summary.json"
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
        st.plotly_chart(fig, use_container_width=True)

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
        st.plotly_chart(fig, use_container_width=True)


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
        st.plotly_chart(fig, use_container_width=True)

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
        st.plotly_chart(fig, use_container_width=True)


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
        use_container_width=True,
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
    st.plotly_chart(fig, use_container_width=True)


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
            index=OPERATING_MODES.index("Normal"),
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
            st.plotly_chart(fig_gauge, use_container_width=True)

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

    # Liste des graphiques attendus dans reports/figures/.
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
        path = REPORTS_FIGURES_DIR / filename
        if path.exists():
            st.markdown(f"#### {title}")
            st.image(str(path), use_container_width=True)
        else:
            st.info(f"`{filename}` non trouvé. Lancer `python scripts/04_interpret.py`.")


# ---------------------------------------------------------------------------
# Sidebar · navigation + crédits.
# ---------------------------------------------------------------------------
def render_sidebar() -> None:
    """Sidebar avec logo + métadonnées projet."""
    with st.sidebar:
        if EFREI_LOGO.exists():
            st.image(str(EFREI_LOGO), use_container_width=True)
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
            "1. `python scripts/01_generate_dataset.py`\n"
            "2. `python scripts/02_eda.py`\n"
            "3. `python scripts/03_train_models.py`\n"
            "4. `python scripts/04_interpret.py`"
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
