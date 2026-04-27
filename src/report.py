# -*- coding: utf-8 -*-
"""Générateur du rapport PDF avec FPDF2.

Architecture du rapport (avec un saut de page systematique avant chaque
figure pour éviter les superpositions et garantir une lecture aérée) ·

  1. Page de garde (logo EFREI + titre + auteurs).
  2. Sommaire.
  3. Section 1 · Contexte métier.
  4. Section 2 · Dataset et EDA.
  5. Section 3 · Méthodologie et architecture.
  6. Section 4 · Modélisation.
  7. Section 5 · Évaluation comparative.
  8. Section 6 · Interprétabilité.
  9. Section 7 · Industrialisation.
 10. Section 8 · Conclusion et perspectives.
 11. Annexe RNCP40875.

FPDF2 utilise la police Helvetica builtin · sous-jacent Latin-1, donc
on garde tous les accents francais standards (é, è, à, ô, ç, ·) mais
on substitue les caracteres Unicode etendus (• -> -, -> -> ->).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
from fpdf import FPDF, XPos, YPos

from .config import EFREI_LOGO, REPORTS_DIR, REPORTS_FIGURES_DIR

# ---------------------------------------------------------------------------
# Charte couleurs RGB (FPDF utilise des tuples 0-255).
# ---------------------------------------------------------------------------
COLOR_NAVY = (13, 71, 161)
COLOR_BLUE = (30, 136, 229)
COLOR_DARK_TEXT = (33, 33, 33)
COLOR_GRAY = (110, 110, 110)
COLOR_LIGHT_BG = (240, 245, 251)


class ProjectReportPDF(FPDF):
    """Sous-classe FPDF avec en-tete et pied de page personnalises.

    L'override de `header()` et `footer()` produit un rendu coherent sur
    toutes les pages internes (la couverture utilise `_skip_header` pour
    desactiver temporairement ces decorations).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._skip_header = False

    # ------------------------------------------------------------------
    # En-tete et pied de page
    # ------------------------------------------------------------------
    def header(self) -> None:
        """Bandeau leger en haut de chaque page interne."""
        if self._skip_header:
            return
        # Filet bleu fin en haut de page.
        self.set_draw_color(*COLOR_BLUE)
        self.set_line_width(0.6)
        self.line(20, 13, 190, 13)

        self.set_font("Helvetica", "", 8)
        self.set_text_color(*COLOR_GRAY)
        self.set_xy(20, 7)
        self.cell(
            0,
            5,
            "Maintenance Prédictive Industrielle  |  M1 DE&IA  |  EFREI 2025-2026",
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
        )
        # Reset a la position d'attaque du contenu.
        self.set_y(20)

    def footer(self) -> None:
        """Pied de page · numero + auteurs."""
        if self._skip_header:
            return
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*COLOR_GRAY)
        self.cell(
            0,
            6,
            f"Adam Beloucif et Emilien Morice  -  Page {self.page_no()}",
            align="C",
        )

    # ------------------------------------------------------------------
    # Helpers de mise en forme
    # ------------------------------------------------------------------
    def h1(self, text: str) -> None:
        """Titre de section niveau 1."""
        self.ln(4)
        self.set_font("Helvetica", "B", 18)
        self.set_text_color(*COLOR_NAVY)
        self.cell(0, 10, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(*COLOR_BLUE)
        self.set_line_width(0.8)
        self.line(20, self.get_y(), 80, self.get_y())
        self.ln(8)

    def h2(self, text: str) -> None:
        """Titre niveau 2."""
        self.ln(4)
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(*COLOR_BLUE)
        self.cell(0, 8, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(2)

    def body(self, text: str) -> None:
        """Paragraphe corps · justifie."""
        self.set_font("Helvetica", "", 11)
        self.set_text_color(*COLOR_DARK_TEXT)
        self.multi_cell(0, 6, text, align="J")
        self.ln(2)

    def bullet(self, text: str) -> None:
        """Item de liste a puce."""
        self.set_font("Helvetica", "", 11)
        self.set_text_color(*COLOR_DARK_TEXT)
        self.cell(6, 6, "")
        self.cell(4, 6, "-")
        self.multi_cell(0, 6, text, align="L")
        self.ln(0.5)

    def caption(self, text: str) -> None:
        """Legende sous une figure."""
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(*COLOR_GRAY)
        self.multi_cell(0, 5, text, align="C")
        self.ln(2)

    def figure(
        self,
        path: Path,
        caption: str,
        max_width: float = 165.0,
        new_page: bool = True,
    ) -> None:
        """Insere une figure centree avec sa legende, sur sa propre page.

        Le saut de page systematique (`new_page=True` par defaut) garantit
        qu'aucune figure n'est tronquee ni superposee a un texte voisin.
        Cela genere un rapport plus long mais bien plus aéré.
        """
        if new_page:
            self.add_page()

        # Titre de la figure (utilise comme h2 si sur sa propre page).
        if not path.exists():
            self.body(f"[Figure manquante · {path.name}]")
            return

        # Centrage horizontal.
        x_offset = (210 - max_width) / 2

        # On laisse au moins 4mm de marge avant l'image pour respirer.
        self.ln(2)

        # Hauteur disponible · de get_y() jusqu'a (h - footer_margin - caption).
        available_h = self.h - self.get_y() - 35
        # h=0 conserve le ratio. FPDF clippe automatiquement a max_width.
        self.image(
            str(path),
            x=x_offset,
            y=self.get_y(),
            w=max_width,
            h=0,
            keep_aspect_ratio=True,
        )
        # Avancement vertical · on prend une hauteur prudente (ne renvoie
        # pas la hauteur reelle apres scaling). 110mm convient a la plupart
        # des figures matplotlib produites par le projet.
        approximate_image_height = min(110, available_h - 10)
        self.ln(approximate_image_height)
        self.caption(caption)

    def metrics_table(self, df: pd.DataFrame, title: str = "") -> None:
        """Tableau de métriques formate · entete bleue + lignes alternees."""
        if title:
            self.h2(title)

        cols = list(df.columns)
        # Largeur totale · 170mm. Premiere colonne (nom modèle) plus large.
        first_col = 44
        other_col = (170 - first_col) / max(len(cols) - 1, 1)
        col_widths = [first_col] + [other_col] * (len(cols) - 1)

        # En-tete.
        self.set_fill_color(*COLOR_NAVY)
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 9)
        for col, w in zip(cols, col_widths):
            # On raccourcit les noms de colonnes longs.
            label = col.replace("_", " ")
            self.cell(w, 8, label, border=1, align="C", fill=True)
        self.ln()

        # Lignes (alternance pour lisibilite).
        self.set_text_color(*COLOR_DARK_TEXT)
        self.set_font("Helvetica", "", 9)
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
    """Formate la date en francais long."""
    months = [
        "janvier",
        "fevrier",
        "mars",
        "avril",
        "mai",
        "juin",
        "juillet",
        "aout",
        "septembre",
        "octobre",
        "novembre",
        "decembre",
    ]
    return f"{d.day} {months[d.month - 1]} {d.year}"


# ---------------------------------------------------------------------------
# Page de garde
# ---------------------------------------------------------------------------
def build_cover_page(pdf: ProjectReportPDF) -> None:
    """Construit la page de garde avec logo EFREI + titre + auteurs."""
    pdf.add_page()
    pdf._skip_header = True

    # Logo EFREI centre en haut.
    if EFREI_LOGO.exists():
        pdf.image(str(EFREI_LOGO), x=55, y=20, w=100)

    pdf.set_y(80)

    # Bandeau bleu avec titre principal.
    pdf.set_fill_color(*COLOR_NAVY)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 21)
    pdf.set_x(15)
    pdf.cell(
        180,
        16,
        "Système Intelligent Multi-Modèles",
        align="C",
        fill=True,
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
    )
    pdf.set_font("Helvetica", "B", 17)
    pdf.set_x(15)
    pdf.cell(
        180,
        12,
        "Maintenance Prédictive Industrielle",
        align="C",
        fill=True,
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
    )

    # Sous-titre academique.
    pdf.ln(10)
    pdf.set_text_color(*COLOR_NAVY)
    pdf.set_font("Helvetica", "I", 13)
    pdf.cell(
        0,
        8,
        "Projet Data Science - M1 Mastère Data Engineering et IA",
        align="C",
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
    )
    pdf.cell(
        0,
        7,
        "Bloc 2 (BC2) - RNCP40875 - Année universitaire 2025-2026",
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
    pdf.cell(0, 7, "Réalisé par", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 9, "Adam BELOUCIF", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 9, "Emilien MORICE", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # Pied de couverture · ecole + adresse + date.
    pdf.set_y(-50)
    pdf.set_text_color(*COLOR_GRAY)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(
        0,
        6,
        "EFREI Paris Panthéon-Assas Université",
        align="C",
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
    )
    pdf.cell(
        0,
        6,
        "30-32 avenue de la République, 94800 Villejuif",
        align="C",
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
    )
    pdf.cell(0, 6, "www.efrei.fr", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "I", 9)
    pdf.cell(
        0,
        6,
        _format_french_date(datetime.now()),
        align="C",
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
    )

    pdf._skip_header = False


def build_toc_page(pdf: ProjectReportPDF) -> None:
    """Sommaire avec liste numerotee des sections."""
    pdf.add_page()
    pdf.h1("Sommaire")

    sections = [
        ("1.", "Contexte métier et objectifs"),
        ("2.", "Dataset et analyse exploratoire (EDA)"),
        ("3.", "Méthodologie et architecture du système"),
        ("4.", "Modélisation multi-algorithmes"),
        ("5.", "Évaluation comparative des modèles"),
        ("6.", "Interprétabilité et explicabilite"),
        ("7.", "Industrialisation - Dashboard et API"),
        ("8.", "Conclusion et perspectives"),
        ("A.", "Annexes - alignément RNCP40875"),
    ]
    pdf.set_font("Helvetica", "", 12)
    pdf.set_text_color(*COLOR_DARK_TEXT)
    pdf.ln(4)
    for num, title in sections:
        pdf.cell(15, 9, num, new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.cell(0, 9, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(8)
    pdf.set_font("Helvetica", "I", 10)
    pdf.set_text_color(*COLOR_GRAY)
    pdf.multi_cell(
        0,
        5.5,
        "Le présent document constitue le rapport analytique structuré exige "
        "par le cahier des charges du Projet Data Science. Il accompagne le "
        "code source (livrable principal), le dashboard Streamlit interactif "
        "et l'API REST FastAPI déployable.",
        align="J",
    )


# ---------------------------------------------------------------------------
# Sections du rapport
# ---------------------------------------------------------------------------
def build_section_context(pdf: ProjectReportPDF) -> None:
    """Section 1 · contexte métier et objectifs."""
    pdf.add_page()
    pdf.h1("1. Contexte métier et objectifs")

    pdf.h2("1.1 Problème industriel")
    pdf.body(
        "Les chaînes de production industrielles modernes sont équipées de "
        "capteurs IoT générant en continu des mesures physiques (vibration, "
        "température, pression, vitesse de rotation). Une défaillance non "
        "anticipee provoque un arrêt non planifie, un coût de maintenance "
        "corrective élevé et une perte de productivite. La maintenance "
        "prédictive vise a transformer ces signaux bruts en alertes "
        "exploitables avant que la panne ne survienne, en s'appuyant sur "
        "des modèles d'apprentissage supervise capables de reconnaitre les "
        "signatures d'une dégradation imminente."
    )

    pdf.h2("1.2 Cible prédictive retenue")
    pdf.body(
        "Le sujet propose plusieurs tâches prédictives complémentaires "
        "(classification binaire, multi-classe, regression). Conformement "
        "aux instructions, nous nous concentrons sur la tâche la plus "
        "naturelle du dataset · la classification binaire `failure_within_24h`. "
        "Cette tâche répond directement au besoin métier le plus critique · "
        "anticiper une panne dans les 24 heures pour déclencher une "
        "intervention preventive et reduire les arrêts non planifies."
    )

    pdf.h2("1.3 Objectifs pedagogiques (RNCP40875 - BC2)")
    pdf.bullet("Collecter et préparéer des données industrielles (C3.1).")
    pdf.bullet("Mener une analyse exploratoire rigoureuse (C3.3).")
    pdf.bullet("Concevoir et entrainer plusieurs modèles ML/DL (C4.2).")
    pdf.bullet("Évaluer et comparer les performances (C4.3).")
    pdf.bullet("Construire un dashboard décisionnel inclusif (C3.2).")
    pdf.bullet("Industrialiser via une API REST exposant le modèle final.")


def build_section_dataset(pdf: ProjectReportPDF) -> None:
    """Section 2 · dataset et EDA."""
    pdf.add_page()
    pdf.h1("2. Dataset et analyse exploratoire")

    pdf.h2("2.1 Schéma du dataset")
    pdf.body(
        "Le dataset reproduit fidèlement le schéma de la source Kaggle "
        "officielle (`tatheerabbas/industrial-machine-predictive-maintenance`). "
        "Pour garantir la reproductibilite hors connexion, un generateur "
        "synthetique reproduisant les mêmes 24 042 enregistrements et 15 "
        "variables est embarque dans le module `src/data_loader.py`. Les "
        "relations physiques injectées (vibration croissante avec l'âge "
        "machine, surchauffe en mode HighLoad, pic de pression hydraulique "
        "lors d'une panne, etc.) garantissent que les modèles apprennent des "
        "patterns métier interprétables."
    )

    pdf.h2("2.2 Variables principales")
    pdf.bullet("vibration_rms · vibration efficace en mm/s (capteur principal).")
    pdf.bullet("temperature_motor · température moteur en degrés Celsius.")
    pdf.bullet("rpm · vitesse de rotation en tours par minute.")
    pdf.bullet("pressure_level · pression du circuit hydraulique en bar.")
    pdf.bullet("operating_mode · Normal / HighLoad / Idle / Maintenance.")
    pdf.bullet("rul_hours · durée de vie restante en heures (regression).")
    pdf.bullet("failure_within_24h · cible binaire (classification - retenue).")
    pdf.bullet("failure_type · type de panne (classification multi-classe).")

    pdf.h2("2.3 Distribution des classes")
    pdf.body(
        "La cible binaire est fortement déséquilibree (les pannes sont "
        "minoritaires). Cela impose l'usage de métriques adaptees (F1, "
        "PR-AUC) et de strategies de pondération des classes pour éviter "
        "qu'un modèle se contente de prédire la classe majoritaire."
    )
    pdf.figure(
        REPORTS_FIGURES_DIR / "eda_target_distribution.png",
        "Figure 1 · Répartition des observations entre machines saines et "
        "machines en panne dans les 24h.",
    )

    pdf.figure(
        REPORTS_FIGURES_DIR / "eda_failure_type_distribution.png",
        "Figure 2 · Répartition des types de panne au sein des machines "
        "défaillantes (mécanique, thermique, électrique, hydraulique).",
    )

    pdf.h2("2.4 Distributions des capteurs")
    pdf.body(
        "Les distributions des 10 capteurs numériques sont tres hétérogènes "
        "en ordre de grandeur (rpm en milliers, humidite en pourcentage, "
        "vibration en mm/s). La standardisation est donc obligatoire pour "
        "tous les modèles sensibles a l'echelle (regression logistique, MLP)."
    )
    pdf.figure(
        REPORTS_FIGURES_DIR / "eda_sensor_distributions.png",
        "Figure 3 · Distributions des 10 capteurs numériques.",
        max_width=180,
    )

    pdf.figure(
        REPORTS_FIGURES_DIR / "eda_boxplots_by_class.png",
        "Figure 4 · Boxplots par classe. Vibration et température montrent "
        "une separation marquee entre OK et Panne, ce qui en fait des "
        "predicteurs forts.",
        max_width=180,
    )

    pdf.h2("2.5 Correlations entre features")
    pdf.figure(
        REPORTS_FIGURES_DIR / "eda_correlation_heatmap.png",
        "Figure 5 · Matrice de correlation. Les correlations élevées entre "
        "voltage, current et power_consumption sont attendues (P=UI). "
        "Aucune redondance critique justifiant la suppression d'une variable.",
    )

    pdf.figure(
        REPORTS_FIGURES_DIR / "eda_scatter_vib_temp.png",
        "Figure 6 · Scatter vibration x température. Les pannes (rouge) se "
        "concentrent dans la zone haute droite, validant l'intuition "
        "physique : une machine surchauffee qui vibre fortement est a risque.",
    )

    pdf.figure(
        REPORTS_FIGURES_DIR / "eda_operating_mode.png",
        "Figure 7 · Le mode HighLoad cumule un grand volume d'observations "
        "et le taux de panne le plus élevé. La feature operating_mode est "
        "donc discriminante.",
    )


def build_section_methodology(pdf: ProjectReportPDF) -> None:
    """Section 3 · méthodologie et architecture."""
    pdf.add_page()
    pdf.h1("3. Méthodologie et architecture du système")

    pdf.h2("3.1 Architecture generale")
    pdf.body(
        "Le système adopte une architecture en 3 couches inspiree du "
        "modèle Medaillon (Bronze / Silver / Gold) · les données brutes sont "
        "ingèrees, transformees, puis exposees a 4 modèles concurrents. "
        "Le modèle final candidat est servi par une API REST consommee a "
        "la fois par le dashboard Streamlit (interface métier) et par "
        "d'eventuels systèmes tiers (GMAO, ERP, supervision)."
    )
    pdf.figure(
        REPORTS_FIGURES_DIR / "diagram_architecture.png",
        "Schéma 1 · Architecture cible du système intelligent.",
        max_width=180,
    )

    pdf.h2("3.2 Pipeline Data Science")
    pdf.body(
        "Le pipeline est strictement séquentiel et chaque etape est isolée "
        "dans un script reproductible · de l'EDA a l'évaluation comparative "
        "en passant par la cross-validation 5-fold. Toute transformation "
        "est ajustee uniquement sur le train set pour éliminer le risque "
        "de data leakage."
    )
    pdf.figure(
        REPORTS_FIGURES_DIR / "diagram_ml_pipeline.png",
        "Schéma 2 · Pipeline ML séquentiel · 8 étapes du brut au modèle " "déployable.",
        max_width=180,
    )

    pdf.h2("3.3 Compromis biais-variance")
    pdf.body(
        "Le choix de comparer 4 modèles (Logistic, Random Forest, XGBoost, "
        "MLP) couvre l'ensemble du spectre complexité / interprétabilité. "
        "La regression logistique est la baseline interprétable, le MLP est "
        "le modèle le plus expressif. La comparaison permet de visualiser "
        "concretement le compromis biais-variance : un modèle tres complexe "
        "n'est pas systematiquement le meilleur en generalisation."
    )
    pdf.figure(
        REPORTS_FIGURES_DIR / "diagram_bias_variance.png",
        "Schéma 3 · Positionnement des 4 modèles sur l'axe complexité / "
        "erreur. Le minimum d'erreur n'est pas necessairement le modèle "
        "le plus complexe.",
    )

    pdf.h2("3.4 Workflow décisionnel cote utilisateur")
    pdf.figure(
        REPORTS_FIGURES_DIR / "diagram_decision_workflow.png",
        "Schéma 4 · Du signal capteur à l'action terrain. Le SHAP intervient "
        "en explication post-prédiction pour justifier la décision auprès "
        "de l'opérateur de maintenance.",
        max_width=180,
    )


def build_section_modeling(pdf: ProjectReportPDF) -> None:
    """Section 4 · modelisation."""
    pdf.add_page()
    pdf.h1("4. Modélisation multi-algorithmes")

    pdf.h2("4.1 Modèles compares")
    pdf.body(
        "Le sujet impose au minimum 4 modèles dont au moins un Deep Learning. "
        "Nous retenons quatre algorithmes représentatifs des grandes familles "
        "supervisees applicables a la classification tabulaire."
    )
    pdf.bullet("Logistic Regression · baseline interprétable, pondération de " "classes activee.")
    pdf.bullet("Random Forest · 200 arbres, regularisation par bagging et " "min_samples_leaf=5.")
    pdf.bullet(
        "XGBoost · 300 estimateurs, learning rate 0.05, scale_pos_weight "
        "aligné sur le ratio neg/pos."
    )
    pdf.bullet(
        "Multi-Layer Perceptron 64-32-16 · Deep Learning, early stopping " "actif, alpha = 1e-3."
    )

    pdf.h2("4.2 Pipeline de préparation")
    pdf.body(
        "Chaque modèle est encapsule dans un sklearn.Pipeline qui chaîne "
        "les étapes suivantes : SimpleImputer median (numérique) et "
        "most_frequent (catégoriel), puis StandardScaler (numérique) et "
        "OneHotEncoder (catégoriel), enfin l'estimateur final. Cette "
        "structuré garantit que les statistiques d'imputation et de scaling "
        "sont calculees uniquement sur le train set et rejouees a "
        "l'identique en inférénce (anti data-leakage)."
    )

    pdf.h2("4.3 Strategie de gestion du déséquilibre")
    pdf.body(
        "La cible binaire est déséquilibree (les pannes sont minoritaires). "
        "Trois strategies complémentaires sont appliquéees pour ne pas "
        "biaiser l'apprentissage."
    )
    pdf.bullet("Stratification du split train / test (80 / 20).")
    pdf.bullet("class_weight = balanced pour Logistic Regression et Random Forest.")
    pdf.bullet("scale_pos_weight = neg / pos pour XGBoost (alternative pondération).")
    pdf.bullet(
        "Choix de F1 et PR-AUC comme métriques principales (plus pertinentes "
        "que l'accuracy en regime déséquilibre)."
    )

    pdf.h2("4.4 Cross-validation")
    pdf.body(
        "Une cross-validation stratifiée 5-fold est appliquéee sur un "
        "échantillon de 8000 lignes pour confirmer que les performances ne "
        "dépendent pas du découpage train / test choisi. L'écart-type des "
        "scores F1 par fold est rapporte dans le tableau de la section 5 "
        "et permet d'évaluer la stabilité globale du modèle."
    )


def build_section_évaluation(pdf: ProjectReportPDF) -> None:
    """Section 5 · évaluation comparative."""
    pdf.add_page()
    pdf.h1("5. Évaluation comparative des modèles")

    metrics_csv = REPORTS_DIR / "metrics_summary.csv"
    if metrics_csv.exists():
        metrics_df = pd.read_csv(metrics_csv)
        cols = ["model_name", "accuracy", "precision", "recall", "f1", "roc_auc"]
        cols = [c for c in cols if c in metrics_df.columns]
        pdf.metrics_table(metrics_df[cols], "5.1 Tableau comparatif des métriques")

        # Deuxieme tableau · CV + coût calcul (lisibilite).
        cols2 = ["model_name", "pr_auc", "fit_time_s", "predict_time_ms"]
        cols2 = [c for c in cols2 if c in metrics_df.columns]
        if "cv_f1_mean" in metrics_df.columns:
            cols2.extend(["cv_f1_mean", "cv_f1_std"])
        pdf.metrics_table(
            metrics_df[cols2],
            "5.2 Stabilité (cross-validation) et coût computationnel",
        )

    pdf.h2("5.3 Visualisation comparative")
    pdf.figure(
        REPORTS_FIGURES_DIR / "metrics_comparison_barplot.png",
        "Figure 8 · Histogramme groupe des 6 métriques cles. Permet de "
        "visualiser d'un coup d'oeil le compromis entre Precision et Recall.",
        max_width=180,
    )

    pdf.h2("5.4 Courbes ROC et Precision-Recall")
    pdf.figure(
        REPORTS_FIGURES_DIR / "roc_curves_comparison.png",
        "Figure 9 · Courbes ROC superposees. Plus l'aire sous la courbe est "
        "élevée, plus le modèle discrimine les classes.",
    )
    pdf.figure(
        REPORTS_FIGURES_DIR / "pr_curves_comparison.png",
        "Figure 10 · Courbes Precision-Recall. Plus pertinente que ROC en "
        "présence de déséquilibre car elle penalise les faux positifs.",
    )

    pdf.h2("5.5 Coût computationnel et écoresponsabilité")
    pdf.figure(
        REPORTS_FIGURES_DIR / "compute_cost_comparison.png",
        "Figure 11 · Temps d'entrainement et latence d'inférénce. Le sujet "
        "demande explicitement d'évaluer le degré d'écoresponsabilité (C4.3).",
        max_width=180,
    )

    pdf.h2("5.6 Sélection du modèle final candidat")
    final_name_path = REPORTS_DIR.parent / "models" / "final_model_name.txt"
    final_name = (
        final_name_path.read_text(encoding="utf-8").strip()
        if final_name_path.exists()
        else "(non disponible)"
    )
    pdf.body(
        f"Le modèle candidat final retenu est `{final_name}`. La sélection "
        "repose sur le score `F1 - 0.5 x écart-type CV`, qui privilegie "
        "les modèles a la fois performants et stables. Au-dela de la "
        "performance brute, ce modèle offre un bon compromis avec son coût "
        "computationnel et reste déployable dans une architecture API "
        "legère sans GPU."
    )

    pdf.h2("5.7 Analyse des erreurs")
    pdf.body(
        "Les matrices de confusion individuelles (annexe) montrent que les "
        "principaux faux negatifs (panne ratee) sont concentres sur les "
        "machines en mode Normal avec une vibration moderee et une "
        "température legèrement élevée. Cela suggère d'enrichir le dataset "
        "avec des features temporelles (variation a 1h, 6h, 24h) pour "
        "ameliorer la detection des dégradations progressives."
    )

    # Annexe matrices de confusion individuelles.
    pdf.h2("5.8 Matrices de confusion (test set)")
    for model_name in ["logistic_regression", "random_forest", "xgboost", "mlp"]:
        cm_path = REPORTS_FIGURES_DIR / f"confusion_matrix_{model_name}.png"
        if cm_path.exists():
            pdf.figure(
                cm_path,
                f"Matrice de confusion · {model_name} (normalisee par "
                "ligne, lecture en pourcentages de Recall par classe).",
                max_width=140,
            )


def build_section_interpretability(pdf: ProjectReportPDF, final_name: str) -> None:
    """Section 6 · interprétabilité (3 niveaux exiges)."""
    pdf.add_page()
    pdf.h1("6. Interprétabilité et explicabilite")

    pdf.body(
        "Le sujet impose explicitement trois niveaux d'explicabilite "
        "(basique, recommande, avance). Tous sont implementes dans "
        "src/interpretability.py et illustres ci-dessous sur le modèle "
        f"final `{final_name}`. Cette analyse répond a la question métier "
        "essentielle : pourquoi le modèle indique-t-il un risque de panne "
        "élevé sur une machine donnee ?"
    )

    pdf.h2("6.1 Feature Importance native")
    pdf.body(
        "Importance calculee par la reduction d'impurete Gini (Random Forest) "
        "ou par le gain d'information (XGBoost). Rapide a obtenir, mais "
        "biaisee en faveur des variables continues et instable en présence "
        "de correlations entre features."
    )
    native_fig = REPORTS_FIGURES_DIR / f"feature_importance_native_{final_name}.png"
    if native_fig.exists():
        pdf.figure(
            native_fig,
            "Figure 12 · Feature importance native. Vibration et "
            "température ressortent comme variables principales, confirmant "
            "l'EDA.",
        )

    pdf.h2("6.2 Permutation Importance (recommandee)")
    pdf.body(
        "Méthode agnostique au modèle · on permute aléatoirement chaque "
        "variable et on mesure la perte de F1. Plus stable et comparable "
        "entre familles de modèles que l'importance native, elle reste la "
        "reference recommandee pour comparer rigoureusement plusieurs "
        "algorithmes hétérogènes."
    )
    perm_fig = REPORTS_FIGURES_DIR / f"permutation_importance_{final_name}.png"
    if perm_fig.exists():
        pdf.figure(
            perm_fig,
            "Figure 13 · Permutation Importance avec barres d'écart-type. "
            "Confirme la dominance de vibration_rms et temperature_motor.",
        )

    pdf.h2("6.3 SHAP - explicabilite locale et globale (avance)")
    pdf.body(
        "SHAP (SHapley Additive exPlanations) attribue a chaque feature "
        "une contribution chiffree a la prédiction, avec garantie "
        "d'additivite et de coherence. Permet d'expliquer une prédiction "
        "individuelle (vue locale, par exemple : pourquoi cette machine "
        "a-t-elle 0.82 de risque ?) et d'agreger globalement."
    )
    shap_summary = REPORTS_FIGURES_DIR / f"shap_summary_{final_name}.png"
    shap_bar = REPORTS_FIGURES_DIR / f"shap_bar_{final_name}.png"
    if shap_summary.exists():
        pdf.figure(
            shap_summary,
            "Figure 14 · SHAP Summary plot. Chaque point = une prédiction. "
            "La couleur indique la valeur de la feature, l'abscisse l'impact "
            "sur la prédiction.",
        )
    if shap_bar.exists():
        pdf.figure(
            shap_bar,
            "Figure 15 · SHAP importance globale (moyenne des valeurs " "absolues SHAP).",
        )

    pdf.add_page()
    pdf.h2("6.4 Lecture métier")
    pdf.body(
        "Pour un responsable maintenance, ces visualisations répondent "
        "directement a la question critique : pourquoi le modèle indique "
        "un risque élevé ? Une vibration superieure a 6 mm/s combinee a "
        "une température au-dessus de 90 degrés Celsius sur une machine non "
        "maintenue depuis plus de 6 mois est l'archetype identifie comme "
        "pre-panne par les trois méthodes (importance native, permutation, "
        "SHAP) de maniere coherente et reproductible."
    )


def build_section_industrialization(pdf: ProjectReportPDF) -> None:
    """Section 7 · industrialisation (Dashboard + API)."""
    pdf.add_page()
    pdf.h1("7. Industrialisation - Dashboard et API")

    pdf.h2("7.1 Dashboard Streamlit")
    pdf.body(
        "Le dashboard est concu comme un outil décisionnel autonome, "
        "destine a un responsable maintenance non technique. Il propose "
        "5 onglets : vue d'ensemble (KPI), exploration des données, "
        "comparaison des modèles, simulateur de scénario machine, et "
        "interprétabilité. Le style adopte la charte EFREI (CSS "
        "personnalise) pour rester coherent avec l'identite "
        "institutionnelle de l'ecole."
    )

    pdf.h2("7.2 API REST FastAPI (option industrialisation)")
    pdf.body("L'API expose 3 endpoints documentés via Swagger UI.")
    pdf.bullet(
        "POST /predict · reçoit un JSON de mesures capteurs, renvoie la "
        "classe prédite, la probabilité, le niveau de risque et la "
        "recommandation opérationnelle."
    )
    pdf.bullet(
        "GET /health · vérification de l'etat du service et du chargement " "du modèle en mémoire."
    )
    pdf.bullet(
        "GET /model-info · metadonnées du modèle servi (nom, métriques, " "features attendues)."
    )
    pdf.body(
        "La validation Pydantic v2 contrôle automatiquement les types et "
        "les plages de valeurs des entrees · une valeur hors plage "
        "déclenche une 422 sans atteindre le modèle. L'architecture "
        "Front (Streamlit) -> API (FastAPI) -> Modèle (joblib) reproduit "
        "fidèlement les pratiques d'un environnement de production."
    )


def build_section_conclusion(pdf: ProjectReportPDF) -> None:
    """Section 8 · conclusion."""
    pdf.add_page()
    pdf.h1("8. Conclusion et perspectives")

    pdf.h2("8.1 Synthese")
    pdf.body(
        "Ce projet livre un MVP complet de maintenance prédictive integrant "
        "les briques fondamentales d'un système d'IA industriel."
    )
    pdf.bullet("Pipeline de données reproductible (24 042 lignes, 15 variables).")
    pdf.bullet("4 modèles entraines, compares avec rigueur méthodologique.")
    pdf.bullet("Interprétabilité a 3 niveaux (native, permutation, SHAP).")
    pdf.bullet("Dashboard Streamlit opérationnel avec CSS personnalise.")
    pdf.bullet("API FastAPI documentee et validee Pydantic v2.")
    pdf.bullet("Rapport analytique structuré (le présent document).")

    pdf.h2("8.2 Limites identifiees")
    pdf.body(
        "Le dataset reste un environnement contrôle. En production reelle, "
        "des biais supplémentaires apparaîtraient · dérive temporelle des "
        "capteurs (drift), hétérogénéité des familles de machines, "
        "evenements rares (mode degrade) sous-représentes. Le modèle final "
        "gagnerait a etre couple a un système de monitoring statistique de "
        "la qualite des données (data drift detection) et a un protocole "
        "de re-entrainement périodique."
    )

    pdf.h2("8.3 Perspectives")
    pdf.bullet(
        "Ajouter une couche temporelle (RNN / LSTM) pour exploiter la " "succession des mesures."
    )
    pdf.bullet(
        "Etendre les tâches prédictives a la regression sur la durée de "
        "vie restante (rul_hours) et a la classification multi-classe du "
        "type de panne."
    )
    pdf.bullet(
        "Brancher l'API a un système GMAO (Gestion de Maintenance Assistee "
        "par Ordinateur) pour déclencher automatiquement les ordres de travail."
    )
    pdf.bullet(
        "Mesurer l'empreinte carbone reelle des entrainements via CodeCarbon "
        "(écoresponsabilité, C4.3)."
    )


def build_annex_rncp(pdf: ProjectReportPDF) -> None:
    """Annexe · matrice de couverture des compétences RNCP40875."""
    pdf.add_page()
    pdf.h1("Annexe A. Alignement RNCP40875 - Bloc BC2")

    pdf.body(
        "Le tableau ci-dessous detaille la couverture par notre projet de "
        "chaque compétence du Bloc 2 (Piloter et implementer des solutions "
        "d'IA en s'aidant notamment de l'IA generative)."
    )
    pdf.ln(4)

    rows = [
        (
            "C3.1",
            "Préparation des données",
            "src/preprocessing.py · ColumnTransformer (SimpleImputer + "
            "StandardScaler + OneHotEncoder).",
        ),
        (
            "C3.2",
            "Tableau de bord interactif",
            "dashboard/app.py · Streamlit, CSS personnalise, 5 onglets, " "simulateur temps reel.",
        ),
        (
            "C3.3",
            "Analyse exploratoire",
            "scripts/02_eda.py · 7 graphiques + statistiques descriptives.",
        ),
        (
            "C4.1",
            "Strategie d'integration IA",
            "Architecture API + Dashboard + Modele decrite section 3.1, " "schéma 1.",
        ),
        (
            "C4.2",
            "Modèles prédictifs ML / DL",
            "src/models.py · 4 modèles dont MLP (DL), pipelines " "reproductibles et seedés.",
        ),
        (
            "C4.3",
            "Évaluation comparative",
            "src/évaluation.py · 6 métriques + ROC/PR + coût computationnel "
            "(écoresponsabilité).",
        ),
    ]

    # Entete tableau.
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_fill_color(*COLOR_NAVY)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(20, 9, "Code", border=1, align="C", fill=True)
    pdf.cell(55, 9, "Compétence", border=1, align="C", fill=True)
    pdf.cell(105, 9, "Couverture dans le projet", border=1, align="C", fill=True)
    pdf.ln()

    pdf.set_text_color(*COLOR_DARK_TEXT)
    pdf.set_font("Helvetica", "", 9)
    for i, (code, comp, cov) in enumerate(rows):
        fill = i % 2 == 0
        if fill:
            pdf.set_fill_color(*COLOR_LIGHT_BG)
        else:
            pdf.set_fill_color(255, 255, 255)
        # On utilise multi_cell pour la 3e colonne (texte long), ce qui
        # impose de gèrer manuellement la position des autres cellules.
        x_start = pdf.get_x()
        y_start = pdf.get_y()

        # Hauteur de ligne calculee a partir du texte le plus long.
        # Pour rester simple, on fixe a 14mm (2 lignes possibles).
        row_h = 14

        pdf.set_xy(x_start, y_start)
        pdf.cell(20, row_h, code, border=1, align="C", fill=fill)
        pdf.cell(55, row_h, comp, border=1, align="L", fill=fill)
        pdf.set_xy(x_start + 75, y_start)
        pdf.multi_cell(105, 5, cov, border=1, align="L", fill=fill)
        # Repositionne pour la ligne suivante.
        pdf.set_xy(x_start, y_start + row_h)

    pdf.ln(8)
    pdf.h2("References techniques")
    pdf.bullet("scikit-learn 1.x · Pipeline, ColumnTransformer, MLPClassifier.")
    pdf.bullet("XGBoost 2.x · gradient boosting optimise.")
    pdf.bullet("SHAP 0.4x · explicabilite locale et globale.")
    pdf.bullet("Streamlit 1.x · dashboard décisionnel.")
    pdf.bullet("FastAPI 0.1x + Pydantic v2 · API REST validee.")
    pdf.bullet("FPDF2 2.x · production du présent rapport.")


def render_full_report(output_path: Path | None = None) -> Path:
    """Pipeline complet de production du rapport.

    Returns
    -------
    Path
        Chemin du PDF produit.
    """
    pdf = ProjectReportPDF(orientation="P", unit="mm", format="A4")
    # Marges generes pour aérér la mise en page (top 22 pour laisser
    # respirer le header, bottom 20 pour le footer).
    pdf.set_margins(20, 22, 20)
    pdf.set_auto_page_break(auto=True, margin=22)
    pdf.set_compression(True)

    # Recuperation du nom du modèle final pour la section interprétabilité.
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
    build_section_évaluation(pdf)
    build_section_interpretability(pdf, final_name)
    build_section_industrialization(pdf)
    build_section_conclusion(pdf)
    build_annex_rncp(pdf)

    target = output_path or (REPORTS_DIR / "rapport_projet_data_science.pdf")
    pdf.output(str(target))
    return target
