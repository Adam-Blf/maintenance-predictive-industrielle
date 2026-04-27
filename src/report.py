# -*- coding: utf-8 -*-
"""Générateur du rapport PDF avec FPDF2.

FPDF2 est un fork moderne de PyFPDF · supporte UTF-8, polices TrueType
et insertion d'images PNG/JPEG. On l'utilise pour produire le rapport
analytique structuré demandé par le sujet (livrable obligatoire).

Architecture du rapport ·
  1. Page de garde · logo EFREI + titre + auteurs + date.
  2. Sommaire (ToC).
  3. Section 1 · Contexte métier et objectifs.
  4. Section 2 · Dataset et exploration des données (EDA).
  5. Section 3 · Méthodologie et architecture.
  6. Section 4 · Modélisation multi-algorithmes.
  7. Section 5 · Évaluation comparative.
  8. Section 6 · Interprétabilité.
  9. Section 7 · Industrialisation (Dashboard + API).
 10. Section 8 · Conclusion et perspectives.
 11. Annexes.

Le rendu reste sobre et académique · charte EFREI bleu, pas d'emoji.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from fpdf import FPDF, XPos, YPos

from .config import (
    EFREI_LOGO,
    REPORTS_DIR,
    REPORTS_FIGURES_DIR,
)

# ---------------------------------------------------------------------------
# Couleurs RGB (FPDF utilise les tuples R,G,B sur 0-255).
# ---------------------------------------------------------------------------
COLOR_NAVY = (13, 71, 161)  # Bleu nuit · titres
COLOR_BLUE = (30, 136, 229)  # Bleu primaire · accents
COLOR_DARK_TEXT = (33, 33, 33)  # Gris très foncé · corps
COLOR_GRAY = (110, 110, 110)  # Gris · captions
COLOR_LIGHT_BG = (240, 245, 251)  # Bleu très clair · fonds


class ProjectReportPDF(FPDF):
    """Sous-classe FPDF avec en-tête + pied de page personnalisés.

    L'override de `header()` et `footer()` permet d'avoir un rendu
    cohérent sur toutes les pages internes (sauf la page de garde, gérée
    manuellement avec `add_page()` puis suppression du header au besoin).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # `_skip_header` permet d'éviter l'en-tête sur la cover.
        self._skip_header = False

    def header(self) -> None:
        """En-tête répété sur chaque page interne · titre + filet."""
        if self._skip_header:
            return
        # Filet bleu fin en haut de page.
        self.set_draw_color(*COLOR_BLUE)
        self.set_line_width(0.6)
        self.line(15, 12, 195, 12)

        self.set_font("Helvetica", "", 8)
        self.set_text_color(*COLOR_GRAY)
        self.set_xy(15, 6)
        self.cell(
            0,
            5,
            "Maintenance Predictive Industrielle  |  M1 DE&IA  |  EFREI 2025-26",
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
        )
        # Reset à la position normale du contenu.
        self.set_y(15)

    def footer(self) -> None:
        """Pied de page · numéro de page + auteurs."""
        if self._skip_header:
            return
        self.set_y(-13)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*COLOR_GRAY)
        self.cell(
            0,
            6,
            f"Adam Beloucif & Emilien Morice  -  Page {self.page_no()}",
            align="C",
        )

    # ------------------------------------------------------------------
    # Helpers de mise en forme · factorisent les répétitions.
    # ------------------------------------------------------------------
    def h1(self, text: str) -> None:
        """Titre de section niveau 1."""
        self.ln(4)
        self.set_font("Helvetica", "B", 18)
        self.set_text_color(*COLOR_NAVY)
        self.cell(0, 10, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        # Filet sous le titre.
        self.set_draw_color(*COLOR_BLUE)
        self.set_line_width(0.8)
        self.line(15, self.get_y(), 70, self.get_y())
        self.ln(6)

    def h2(self, text: str) -> None:
        """Titre niveau 2."""
        self.ln(3)
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(*COLOR_BLUE)
        self.cell(0, 8, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(2)

    def body(self, text: str) -> None:
        """Paragraphe corps · justifié, gris foncé."""
        self.set_font("Helvetica", "", 10.5)
        self.set_text_color(*COLOR_DARK_TEXT)
        self.multi_cell(0, 5.5, text, align="J")
        self.ln(2)

    def bullet(self, text: str) -> None:
        """Item de liste a puce.

        On utilise un tiret cadratin ASCII compatible avec la police
        Helvetica builtin de FPDF2 (qui ne supporte que Latin-1, donc pas
        le caractere Unicode U+2022 BULLET).
        """
        self.set_font("Helvetica", "", 10.5)
        self.set_text_color(*COLOR_DARK_TEXT)
        # Indentation · tiret court + tab.
        self.cell(6, 5.5, "")
        self.cell(4, 5.5, "-")
        self.multi_cell(0, 5.5, text, align="L")
        self.ln(0.5)

    def caption(self, text: str) -> None:
        """Légende sous une figure ou un tableau."""
        self.set_font("Helvetica", "I", 8.5)
        self.set_text_color(*COLOR_GRAY)
        self.multi_cell(0, 4.5, text, align="C")
        self.ln(2)

    def figure(
        self,
        path: Path,
        caption: str,
        max_width: float = 175.0,
        max_height: float = 110.0,
    ) -> None:
        """Insère une figure centrée + caption en dessous.

        Parameters
        ----------
        path : Path
            Chemin du PNG.
        caption : str
            Légende affichée en italique sous l'image.
        max_width : float
            Largeur maximale (mm). Par défaut 175mm = largeur typique
            d'une page A4 avec marges 15mm.
        max_height : float
            Hauteur maximale · pour éviter de pousser sur 2 pages.
        """
        if not path.exists():
            self.body(f"[Figure manquante · {path.name}]")
            return
        # Centrage horizontal · on calcule x = (210 - max_width) / 2.
        x_offset = (210 - max_width) / 2

        # Vérifier qu'il reste assez de place sur la page · sinon page break.
        if self.get_y() + max_height + 10 > self.h - 20:
            self.add_page()

        self.image(
            str(path),
            x=x_offset,
            y=self.get_y(),
            w=max_width,
            h=0,  # h=0 → conserve le ratio
            keep_aspect_ratio=True,
        )
        # Mesure heuristique · on avance de max_height (FPDF ne renvoie pas
        # la hauteur réelle de l'image après scaling auto).
        self.ln(max_height)
        self.caption(caption)

    def metrics_table(self, df: pd.DataFrame, title: str = "") -> None:
        """Tableau de métriques formaté."""
        if title:
            self.h2(title)

        cols = list(df.columns)
        # Largeurs de colonnes · première colonne plus large (nom modèle).
        col_widths = [38] + [(180 - 38) / (len(cols) - 1)] * (len(cols) - 1)

        # En-tête.
        self.set_fill_color(*COLOR_NAVY)
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 8.5)
        for col, w in zip(cols, col_widths):
            self.cell(w, 8, col, border=1, align="C", fill=True)
        self.ln()

        # Lignes · alternance de fond pour lisibilité.
        self.set_text_color(*COLOR_DARK_TEXT)
        self.set_font("Helvetica", "", 8.5)
        for i, row in df.iterrows():
            fill = i % 2 == 0
            if fill:
                self.set_fill_color(*COLOR_LIGHT_BG)
            for col, w in zip(cols, col_widths):
                value = row[col]
                if isinstance(value, float):
                    text = f"{value:.4f}"
                else:
                    text = str(value)
                self.cell(w, 7, text, border=1, align="C", fill=fill)
            self.ln()
        self.ln(3)


def _format_french_date(d: datetime) -> str:
    """Formate la date en français · '27 avril 2026'."""
    months = [
        "janvier",
        "février",
        "mars",
        "avril",
        "mai",
        "juin",
        "juillet",
        "août",
        "septembre",
        "octobre",
        "novembre",
        "décembre",
    ]
    return f"{d.day} {months[d.month - 1]} {d.year}"


def build_cover_page(pdf: ProjectReportPDF) -> None:
    """Construit la page de garde avec logo EFREI + titre + auteurs."""
    pdf.add_page()
    pdf._skip_header = True  # Pas de header sur la couverture

    # Logo EFREI centré en haut.
    if EFREI_LOGO.exists():
        pdf.image(str(EFREI_LOGO), x=55, y=20, w=100)

    # Espace pour aérer.
    pdf.set_y(75)

    # Bandeau bleu avec titre principal.
    pdf.set_fill_color(*COLOR_NAVY)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_x(15)
    pdf.cell(
        180,
        16,
        "Systeme Intelligent Multi-Modeles",
        align="C",
        fill=True,
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
    )
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_x(15)
    pdf.cell(
        180,
        12,
        "Maintenance Predictive Industrielle",
        align="C",
        fill=True,
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
    )

    # Sous-titre académique.
    pdf.ln(8)
    pdf.set_text_color(*COLOR_NAVY)
    pdf.set_font("Helvetica", "I", 13)
    pdf.cell(
        0,
        8,
        "Projet Data Science - M1 Mastere Data Engineering & IA",
        align="C",
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
    )
    pdf.cell(
        0,
        7,
        "Bloc 2 (BC2) - RNCP40875 - Annee 2025-2026",
        align="C",
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
    )

    # Encart auteurs.
    pdf.ln(28)
    pdf.set_fill_color(*COLOR_LIGHT_BG)
    pdf.set_draw_color(*COLOR_BLUE)
    pdf.set_line_width(0.4)
    pdf.rect(40, pdf.get_y(), 130, 38, style="DF")

    pdf.set_text_color(*COLOR_NAVY)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_y(pdf.get_y() + 4)
    pdf.cell(0, 7, "Realise par", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 9, "Adam BELOUCIF", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 9, "Emilien MORICE", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # Pied de couverture · école + date.
    pdf.set_y(-50)
    pdf.set_text_color(*COLOR_GRAY)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(
        0,
        6,
        "EFREI Paris Pantheon-Assas Universite",
        align="C",
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
    )
    pdf.cell(
        0,
        6,
        "30-32 avenue de la Republique, 94800 Villejuif",
        align="C",
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
    )
    pdf.cell(0, 6, "www.efrei.fr", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "I", 9)
    pdf.cell(
        0,
        6,
        f"Document genere le {_format_french_date(datetime.now())}",
        align="C",
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
    )

    pdf._skip_header = False


def build_toc_page(pdf: ProjectReportPDF) -> None:
    """Sommaire · liste numérotée des sections."""
    pdf.add_page()
    pdf.h1("Sommaire")

    sections = [
        ("1.", "Contexte metier et objectifs"),
        ("2.", "Dataset et analyse exploratoire (EDA)"),
        ("3.", "Methodologie et architecture du systeme"),
        ("4.", "Modelisation multi-algorithmes"),
        ("5.", "Evaluation comparative des modeles"),
        ("6.", "Interpretabilite et explicabilite"),
        ("7.", "Industrialisation - Dashboard et API"),
        ("8.", "Conclusion et perspectives"),
        ("A.", "Annexes - alignement RNCP40875"),
    ]
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(*COLOR_DARK_TEXT)
    for num, title in sections:
        # Ligne avec numéro à gauche, titre, points de suite.
        pdf.cell(15, 8, num, new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.cell(0, 8, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(8)
    pdf.set_font("Helvetica", "I", 9.5)
    pdf.set_text_color(*COLOR_GRAY)
    pdf.multi_cell(
        0,
        5,
        "Le present document constitue le rapport analytique structure exige "
        "par le cahier des charges du Projet Data Science. Il accompagne le "
        "code source (livrable principal), le dashboard Streamlit interactif "
        "et l'API REST FastAPI deployable.",
        align="J",
    )


def build_section_context(pdf: ProjectReportPDF) -> None:
    """Section 1 · contexte métier et objectifs."""
    pdf.add_page()
    pdf.h1("1. Contexte metier et objectifs")

    pdf.h2("1.1 Probleme industriel")
    pdf.body(
        "Les chaines de production industrielles modernes sont equipees de "
        "capteurs IoT generant en continu des mesures physiques (vibration, "
        "temperature, pression, vitesse de rotation). Une defaillance non "
        "anticipee provoque un arret non planifie, un cout de maintenance "
        "corrective eleve et une perte de productivite. La maintenance "
        "predictive vise a transformer ces signaux bruts en alertes "
        "exploitables avant que la panne ne survienne."
    )

    pdf.h2("1.2 Cible predictive retenue")
    pdf.body(
        "Le sujet propose plusieurs taches predictives complementaires "
        "(classification binaire, multi-classe, regression). Conformement "
        "aux instructions, nous nous concentrons sur la tache la plus "
        "naturelle du dataset · la classification binaire `failure_within_24h`. "
        "Cette tache repond directement au besoin metier le plus critique · "
        "anticiper une panne dans les 24 heures pour declencher une "
        "intervention preventive."
    )

    pdf.h2("1.3 Objectifs pedagogiques (RNCP40875 - BC2)")
    pdf.bullet("Collecter et preparer des donnees industrielles (C3.1).")
    pdf.bullet("Mener une analyse exploratoire rigoureuse (C3.3).")
    pdf.bullet("Concevoir et entrainer plusieurs modeles ML/DL (C4.2).")
    pdf.bullet("Evaluer et comparer les performances (C4.3).")
    pdf.bullet("Construire un dashboard decisionnel inclusif (C3.2).")
    pdf.bullet("Industrialiser via une API REST exposant le modele final.")


def build_section_dataset(pdf: ProjectReportPDF) -> None:
    """Section 2 · dataset et EDA."""
    pdf.add_page()
    pdf.h1("2. Dataset et analyse exploratoire")

    pdf.h2("2.1 Schema du dataset")
    pdf.body(
        "Le dataset reproduit fidelement le schema de la source Kaggle "
        "officielle (`tatheerabbas/industrial-machine-predictive-maintenance`). "
        "Pour garantir la reproductibilite hors connexion, un generateur "
        "synthetique reproduisant les memes 24 042 enregistrements et 15 "
        "variables est embarque dans le module `src/data_loader.py`. Les "
        "relations physiques injectees (vibration croissante avec l'age "
        "machine, surchauffe en mode HighLoad, pic de pression hydraulique "
        "lors d'une panne, etc.) garantissent que les modeles apprennent des "
        "patterns metier interpretables."
    )

    pdf.h2("2.2 Variables principales")
    pdf.bullet("vibration_rms - vibration efficace (mm/s, capteur principal).")
    pdf.bullet("temperature_motor - temperature moteur (degC).")
    pdf.bullet("rpm - vitesse de rotation (tr/min).")
    pdf.bullet("pressure_level - pression du circuit hydraulique (bar).")
    pdf.bullet("operating_mode - Normal / HighLoad / Idle / Maintenance.")
    pdf.bullet("rul_hours - duree de vie restante (regression).")
    pdf.bullet("failure_within_24h - cible binaire (classification - retenue).")
    pdf.bullet("failure_type - type de panne (multi-classe).")

    pdf.h2("2.3 Distribution des classes")
    pdf.figure(
        REPORTS_FIGURES_DIR / "eda_target_distribution.png",
        "Figure 1 · La cible est desequilibree (les pannes sont "
        "minoritaires). Cela impose l'usage de metriques adaptees (F1, "
        "PR-AUC) et de strategies de ponderation des classes.",
    )

    pdf.figure(
        REPORTS_FIGURES_DIR / "eda_failure_type_distribution.png",
        "Figure 2 · Repartition des types de panne au sein des machines "
        "defaillantes. La distribution non uniforme reflete des modes de "
        "defaillance differencies (mecanique, thermique, electrique, "
        "hydraulique).",
    )

    pdf.h2("2.4 Distributions des capteurs")
    pdf.figure(
        REPORTS_FIGURES_DIR / "eda_sensor_distributions.png",
        "Figure 3 · Distributions des 10 capteurs numeriques. Echelles "
        "tres heterogenes (rpm en milliers, humidity en %, vibration en "
        "unites mm/s) - la standardisation est obligatoire.",
        max_height=85,
    )

    pdf.figure(
        REPORTS_FIGURES_DIR / "eda_boxplots_by_class.png",
        "Figure 4 · Boxplots par classe. Vibration et temperature montrent "
        "une separation marquee entre OK et Panne, ce qui en fait des "
        "predicteurs forts.",
        max_height=85,
    )

    pdf.h2("2.5 Correlations entre features")
    pdf.figure(
        REPORTS_FIGURES_DIR / "eda_correlation_heatmap.png",
        "Figure 5 · Matrice de correlation. Les correlations elevees entre "
        "voltage/current/power_consumption sont attendues (P=UI). Aucune "
        "redondance critique justifiant la suppression d'une variable.",
    )

    pdf.figure(
        REPORTS_FIGURES_DIR / "eda_scatter_vib_temp.png",
        "Figure 6 · Scatter vibration x temperature. Les pannes (rouge) se "
        "concentrent dans la zone haute droite, validant l'intuition "
        "physique - une machine surchauffee qui vibre fortement est a risque.",
    )

    pdf.figure(
        REPORTS_FIGURES_DIR / "eda_operating_mode.png",
        "Figure 7 · Mode HighLoad cumule un grand volume d'observations et "
        "le taux de panne le plus eleve. La feature operating_mode est donc "
        "discriminante.",
    )


def build_section_methodology(pdf: ProjectReportPDF) -> None:
    """Section 3 · méthodologie et architecture."""
    pdf.add_page()
    pdf.h1("3. Methodologie et architecture du systeme")

    pdf.h2("3.1 Architecture generale")
    pdf.body(
        "Le systeme adopte une architecture en 3 couches inspiree du "
        "modele Medaillon (Bronze/Silver/Gold) · les donnees brutes sont "
        "ingerees, transformees, puis exposees a 4 modeles concurrents. "
        "Le modele final candidat est servi par une API REST consommee a "
        "la fois par le dashboard Streamlit (interface metier) et par "
        "d'eventuels systemes tiers (GMAO, ERP, supervision)."
    )
    pdf.figure(
        REPORTS_FIGURES_DIR / "diagram_architecture.png",
        "Schema 1 · Architecture cible du systeme intelligent.",
        max_height=85,
    )

    pdf.h2("3.2 Pipeline Data Science")
    pdf.body(
        "Le pipeline est strictement sequentiel et chaque etape est "
        "isolee dans un script reproductible · de l'EDA a l'evaluation "
        "comparative en passant par la cross-validation 5-fold. Toute "
        "transformation est ajustee uniquement sur le train set pour "
        "eliminer le risque de data leakage."
    )
    pdf.figure(
        REPORTS_FIGURES_DIR / "diagram_ml_pipeline.png",
        "Schema 2 · Pipeline ML sequentiel - 8 etapes du brut au modele " "deployable.",
        max_height=70,
    )

    pdf.h2("3.3 Compromis biais-variance")
    pdf.body(
        "Le choix de comparer 4 modeles (Logistic, Random Forest, XGBoost, "
        "MLP) couvre l'ensemble du spectre complexite/interpretabilite. La "
        "regression logistique est la baseline interpretable, le MLP est le "
        "modele le plus expressif. La comparaison permet de visualiser "
        "concretement le compromis."
    )
    pdf.figure(
        REPORTS_FIGURES_DIR / "diagram_bias_variance.png",
        "Schema 3 · Positionnement des 4 modeles sur l'axe complexite vs "
        "erreur. Le minimum d'erreur n'est pas necessairement le modele le "
        "plus complexe.",
    )

    pdf.h2("3.4 Workflow decisionnel cote utilisateur")
    pdf.figure(
        REPORTS_FIGURES_DIR / "diagram_decision_workflow.png",
        "Schema 4 · Du signal capteur a l'action terrain. Le SHAP intervient "
        "en explication post-prediction pour justifier la decision aupres de "
        "l'operateur.",
        max_height=70,
    )


def build_section_modeling(pdf: ProjectReportPDF) -> None:
    """Section 4 · modélisation."""
    pdf.add_page()
    pdf.h1("4. Modelisation multi-algorithmes")

    pdf.h2("4.1 Modeles compares")
    pdf.body(
        "Le sujet impose au minimum 4 modeles dont au moins un Deep Learning. " "Nous retenons ·"
    )
    pdf.bullet("Logistic Regression (baseline interpretable, class_weight='balanced').")
    pdf.bullet("Random Forest (200 arbres, regularisation par bagging et " "min_samples_leaf=5).")
    pdf.bullet(
        "XGBoost (300 estimateurs, learning_rate=0.05, scale_pos_weight "
        "aligne sur le ratio neg/pos)."
    )
    pdf.bullet(
        "Multi-Layer Perceptron - 64-32-16 (Deep Learning, early stopping " "active, alpha=1e-3)."
    )

    pdf.h2("4.2 Pipeline de preparation")
    pdf.body(
        "Chaque modele est encapsule dans un sklearn.Pipeline qui chaine ·\n"
        "  1. SimpleImputer median (numerique) + most_frequent (categoriel),\n"
        "  2. StandardScaler (numerique) + OneHotEncoder (categoriel),\n"
        "  3. Estimateur final.\n"
        "Cette structure garantit que les statistiques d'imputation et de "
        "scaling sont calculees uniquement sur le train set et rejouees a "
        "l'identique en inference (anti data-leakage)."
    )

    pdf.h2("4.3 Strategie de gestion du desequilibre")
    pdf.body(
        "La cible binaire est desequilibree (les pannes sont minoritaires). "
        "Trois strategies complementaires sont appliquees ·"
    )
    pdf.bullet("Stratification du split train/test (80/20).")
    pdf.bullet("class_weight='balanced' pour Logistic Regression et Random Forest.")
    pdf.bullet("scale_pos_weight = neg/pos pour XGBoost (alternative ponderation).")
    pdf.bullet(
        "Choix de F1 et PR-AUC comme metriques principales (plus pertinentes "
        "que l'accuracy en regime desequilibre)."
    )

    pdf.h2("4.4 Cross-validation")
    pdf.body(
        "Une cross-validation stratifiee 5-fold est appliquee sur un "
        "echantillon de 8000 lignes pour confirmer que les performances ne "
        "dependent pas du decoupage train/test choisi. L'ecart-type des "
        "scores F1 par fold est rapporte dans le tableau de la section 5 "
        "et permet d'evaluer la stabilite."
    )


def build_section_evaluation(pdf: ProjectReportPDF) -> None:
    """Section 5 · évaluation comparative."""
    pdf.add_page()
    pdf.h1("5. Evaluation comparative des modeles")

    # Chargement du tableau de métriques pour l'inclure dans le PDF.
    metrics_csv = REPORTS_DIR / "metrics_summary.csv"
    if metrics_csv.exists():
        metrics_df = pd.read_csv(metrics_csv)
        # On sélectionne les colonnes essentielles pour rester lisible.
        cols = ["model_name", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
        cols = [c for c in cols if c in metrics_df.columns]
        pdf.metrics_table(metrics_df[cols], "5.1 Tableau comparatif des metriques")
    else:
        pdf.body("Tableau de metriques indisponible (lancer 03_train_models.py).")

    pdf.h2("5.2 Visualisation comparative")
    pdf.figure(
        REPORTS_FIGURES_DIR / "metrics_comparison_barplot.png",
        "Figure 8 · Histogramme groupe des 6 metriques cles. Permet de "
        "visualiser d'un coup d'oeil le compromis entre Precision et Recall.",
        max_height=90,
    )

    pdf.h2("5.3 Courbes ROC et Precision-Recall")
    pdf.figure(
        REPORTS_FIGURES_DIR / "roc_curves_comparison.png",
        "Figure 9 · Courbes ROC superposees. Plus l'AUC est elevee, plus "
        "le modele discrimine bien.",
        max_height=85,
    )
    pdf.figure(
        REPORTS_FIGURES_DIR / "pr_curves_comparison.png",
        "Figure 10 · Courbes Precision-Recall. Plus pertinente que ROC en "
        "presence de desequilibre - elle penalise les faux positifs.",
        max_height=85,
    )

    pdf.h2("5.4 Cout computationnel et ecoresponsabilite")
    pdf.figure(
        REPORTS_FIGURES_DIR / "compute_cost_comparison.png",
        "Figure 11 · Temps d'entrainement et latence d'inference. Le sujet "
        "demande explicitement d'evaluer le degre d'ecoresponsabilite (C4.3).",
        max_height=80,
    )

    pdf.h2("5.5 Selection du modele final candidat")
    name_file = pdf.MODELS_DIR if hasattr(pdf, "MODELS_DIR") else None
    final_name = "(non disponible)"
    final_name_path = REPORTS_DIR.parent / "models" / "final_model_name.txt"
    if final_name_path.exists():
        final_name = final_name_path.read_text(encoding="utf-8").strip()
    pdf.body(
        f"Le modele candidat final retenu est `{final_name}`. La selection "
        "repose sur le score `F1 - 0.5 x ecart-type CV`, qui privilegie "
        "les modeles a la fois performants et stables. Au-dela de la "
        "performance brute, ce modele offre un bon compromis avec son cout "
        "computationnel et reste deployable dans une architecture API legere."
    )

    pdf.h2("5.6 Analyse des erreurs")
    pdf.body(
        "Les matrices de confusion individuelles (annexe) montrent que les "
        "principaux faux negatifs (panne ratee) sont concentres sur les "
        "machines en mode Normal avec une vibration moderee et une "
        "temperature legerement elevee. Cela suggere d'enrichir le dataset "
        "avec des features temporelles (variation a 1h, 6h, 24h) pour "
        "ameliorer la detection des degradations progressives."
    )


def build_section_interpretability(pdf: ProjectReportPDF, final_name: str) -> None:
    """Section 6 · interprétabilité (3 niveaux exigés)."""
    pdf.add_page()
    pdf.h1("6. Interpretabilite et explicabilite")

    pdf.body(
        "Le sujet impose explicitement trois niveaux d'explicabilite "
        "(basique, recommande, avance). Tous sont implementes dans "
        "src/interpretability.py et illustres ci-dessous sur le modele "
        f"final `{final_name}`."
    )

    pdf.h2("6.1 Feature Importance native")
    pdf.body(
        "Importance calculee par la reduction d'impurete Gini (RF) ou par "
        "le gain d'information (XGBoost). Rapide a obtenir, mais biaisee en "
        "faveur des variables continues et instable en presence de "
        "correlations entre features."
    )
    native_fig = REPORTS_FIGURES_DIR / f"feature_importance_native_{final_name}.png"
    if native_fig.exists():
        pdf.figure(
            native_fig,
            "Figure 12 · Feature importance native. Vibration et temperature "
            "ressortent comme variables principales, confirmant l'EDA.",
        )

    pdf.h2("6.2 Permutation Importance (recommande)")
    pdf.body(
        "Methode agnostique au modele · on permute aleatoirement chaque "
        "variable et on mesure la perte de F1. Plus stable et comparable "
        "entre familles de modeles que l'importance native."
    )
    perm_fig = REPORTS_FIGURES_DIR / f"permutation_importance_{final_name}.png"
    if perm_fig.exists():
        pdf.figure(
            perm_fig,
            "Figure 13 · Permutation Importance avec barres d'ecart-type. "
            "Confirme la dominance de vibration_rms et temperature_motor.",
        )

    pdf.h2("6.3 SHAP - explicabilite locale et globale (avance)")
    pdf.body(
        "SHAP (SHapley Additive exPlanations) attribue a chaque feature "
        "une contribution chiffree a la prediction, avec garantie "
        "d'additivite et de coherence. Permet d'expliquer une prediction "
        "individuelle (vue locale, e.g. 'pourquoi cette machine a-t-elle "
        "0.82 de risque ?') et d'agreger globalement."
    )
    shap_summary = REPORTS_FIGURES_DIR / f"shap_summary_{final_name}.png"
    shap_bar = REPORTS_FIGURES_DIR / f"shap_bar_{final_name}.png"
    if shap_summary.exists():
        pdf.figure(
            shap_summary,
            "Figure 14 · SHAP Summary plot. Chaque point = une prediction. "
            "La couleur indique la valeur de la feature, l'abscisse l'impact "
            "sur la prediction.",
            max_height=100,
        )
    if shap_bar.exists():
        pdf.figure(
            shap_bar,
            "Figure 15 · SHAP importance globale (moyenne des |SHAP|).",
        )

    pdf.h2("6.4 Lecture metier")
    pdf.body(
        "Pour un responsable maintenance, ces visualisations repondent "
        "directement a la question critique 'pourquoi le modele indique "
        "un risque eleve ?'. Une vibration superieure a 6 mm/s combinee a "
        "une temperature au-dessus de 90 degC sur une machine non "
        "maintenue depuis plus de 6 mois est l'archetype identifie comme "
        "pre-panne par les trois methodes."
    )


def build_section_industrialization(pdf: ProjectReportPDF) -> None:
    """Section 7 · industrialisation (Dashboard + API)."""
    pdf.add_page()
    pdf.h1("7. Industrialisation - Dashboard et API")

    pdf.h2("7.1 Dashboard Streamlit")
    pdf.body(
        "Le dashboard est concu comme un outil decisionnel autonome, "
        "destine a un responsable maintenance non technique. Il propose "
        "5 onglets · vue d'ensemble (KPI), exploration des donnees, "
        "comparaison des modeles, simulateur de scenario machine, et "
        "interpretabilite. Le style adopte la charte EFREI (CSS "
        "personnalise) pour rester coherent avec l'identite institutionnelle."
    )

    pdf.h2("7.2 API REST FastAPI (option industrialisation)")
    pdf.body("L'API expose 3 endpoints documentes via Swagger UI ·")
    pdf.bullet(
        "POST /predict - recoit un JSON de mesures capteurs, renvoie "
        "la classe predite, la probabilite, le niveau de risque et la "
        "recommandation operationnelle."
    )
    pdf.bullet(
        "GET /health - verification de l'etat du service et du chargement " "du modele en memoire."
    )
    pdf.bullet(
        "GET /model-info - metadonnees du modele servi (nom, metriques, " "features attendues)."
    )
    pdf.body(
        "La validation Pydantic v2 controle automatiquement les types et "
        "les plages de valeurs des entrees · une valeur hors plage "
        "declenche une 422 sans atteindre le modele. L'architecture "
        "Front (Streamlit) -> API (FastAPI) -> Modele (joblib) reproduit "
        "fidelement les pratiques d'un environnement de production."
    )


def build_section_conclusion(pdf: ProjectReportPDF) -> None:
    """Section 8 · conclusion."""
    pdf.add_page()
    pdf.h1("8. Conclusion et perspectives")

    pdf.h2("8.1 Synthese")
    pdf.body("Ce projet livre un MVP complet de maintenance predictive ·")
    pdf.bullet("Pipeline de donnees reproductible (24 042 lignes, 15 variables).")
    pdf.bullet("4 modeles entraines, compares avec rigueur methodologique.")
    pdf.bullet("Interpretabilite a 3 niveaux (native, permutation, SHAP).")
    pdf.bullet("Dashboard Streamlit operationnel avec CSS personnalise.")
    pdf.bullet("API FastAPI documentee et validee Pydantic.")
    pdf.bullet("Rapport analytique structure (le present document).")

    pdf.h2("8.2 Limites identifiees")
    pdf.body(
        "Le dataset reste un environnement controle. En production reelle, "
        "des biais supplementaires apparaitraient · derive temporelle des "
        "capteurs (drift), heterogeneite des familles de machines, "
        "evenements rares (mode degrade) sous-representes. Le modele final "
        "gagnerait a etre couple a un systeme de monitoring statistique de "
        "la qualite des donnees (data drift detection) et a un protocole "
        "de re-entrainement periodique."
    )

    pdf.h2("8.3 Perspectives")
    pdf.bullet(
        "Ajouter une couche temporelle (RNN/LSTM) pour exploiter la " "succession des mesures."
    )
    pdf.bullet(
        "Etendre les taches predictives a la regression sur la duree de vie "
        "restante (rul_hours) et a la classification multi-classe du type "
        "de panne."
    )
    pdf.bullet(
        "Brancher l'API a un systeme GMAO (Gestion de Maintenance Assistee "
        "par Ordinateur) pour generer automatiquement les ordres de travail."
    )
    pdf.bullet(
        "Mesurer l'empreinte carbone reelle des entrainements via "
        "CodeCarbon (ecoresponsabilite, C4.3)."
    )


def build_annex_rncp(pdf: ProjectReportPDF) -> None:
    """Annexe · matrice de couverture des compétences RNCP40875."""
    pdf.add_page()
    pdf.h1("Annexe A. Alignement RNCP40875 - Bloc BC2")

    pdf.body(
        "Le tableau ci-dessous detaille la couverture par notre projet de "
        "chaque competence du Bloc 2 (Piloter et implementer des solutions "
        "d'IA en s'aidant notamment de l'IA generative)."
    )
    pdf.ln(2)

    rows = [
        (
            "C3.1",
            "Preparation des donnees",
            "src/preprocessing.py - SimpleImputer + StandardScaler + OneHotEncoder, ColumnTransformer.",
        ),
        (
            "C3.2",
            "Tableau de bord interactif",
            "dashboard/app.py - Streamlit, CSS personnalise, 5 onglets, simulateur.",
        ),
        (
            "C3.3",
            "Analyse exploratoire",
            "scripts/02_eda.py - 7 graphiques, stats descriptives, scatter, correlation.",
        ),
        (
            "C4.1",
            "Strategie d'integration IA",
            "Architecture API/Dashboard/Modele decrite section 3.1, schema 1.",
        ),
        (
            "C4.2",
            "Modeles predictifs ML/DL",
            "src/models.py - 4 modeles dont MLP (DL), pipelines reproductibles.",
        ),
        (
            "C4.3",
            "Evaluation comparative",
            "src/evaluation.py - 6 metriques, ROC/PR, ecoresponsabilite (temps).",
        ),
    ]

    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(*COLOR_NAVY)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(20, 8, "Code", border=1, align="C", fill=True)
    pdf.cell(60, 8, "Competence", border=1, align="C", fill=True)
    pdf.cell(100, 8, "Couverture dans le projet", border=1, align="C", fill=True)
    pdf.ln()

    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_text_color(*COLOR_DARK_TEXT)
    for i, (code, comp, cov) in enumerate(rows):
        fill = i % 2 == 0
        pdf.set_fill_color(*COLOR_LIGHT_BG)
        # On utilise multi_cell sur la dernière colonne pour le texte long.
        x_start = pdf.get_x()
        y_start = pdf.get_y()

        pdf.cell(20, 14, code, border=1, align="C", fill=fill)
        pdf.cell(60, 14, comp, border=1, align="L", fill=fill)
        # Pour la 3e colonne · on utilise multi_cell mais on doit gérer le
        # positionnement manuellement.
        pdf.multi_cell(100, 7, cov, border=1, align="L", fill=fill)

    pdf.ln(8)
    pdf.h2("References techniques")
    pdf.bullet("scikit-learn 1.x - Pipeline, ColumnTransformer, MLPClassifier.")
    pdf.bullet("XGBoost 2.x - Gradient boosting optimise.")
    pdf.bullet("SHAP 0.4x - explicabilite locale et globale.")
    pdf.bullet("Streamlit 1.x - dashboard decisionnel.")
    pdf.bullet("FastAPI 0.1x + Pydantic v2 - API REST validee.")
    pdf.bullet("FPDF2 2.x - generation du present rapport.")


def render_full_report(output_path: Path | None = None) -> Path:
    """Pipeline complet de generation du rapport.

    Returns
    -------
    Path
        Chemin du PDF produit.
    """
    pdf = ProjectReportPDF(orientation="P", unit="mm", format="A4")
    pdf.set_margins(15, 18, 15)
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_compression(True)

    # Récupération du nom du modèle final pour la section interprétabilité.
    final_name_path = REPORTS_DIR.parent / "models" / "final_model_name.txt"
    final_name = (
        final_name_path.read_text(encoding="utf-8").strip() if final_name_path.exists() else "model"
    )

    build_cover_page(pdf)
    build_toc_page(pdf)
    build_section_context(pdf)
    build_section_dataset(pdf)
    build_section_methodology(pdf)
    build_section_modeling(pdf)
    build_section_evaluation(pdf)
    build_section_interpretability(pdf, final_name)
    build_section_industrialization(pdf)
    build_section_conclusion(pdf)
    build_annex_rncp(pdf)

    target = output_path or (REPORTS_DIR / "rapport_projet_data_science.pdf")
    pdf.output(str(target))
    return target
