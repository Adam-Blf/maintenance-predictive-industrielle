# -*- coding: utf-8 -*-
"""Script 06 · génération du rapport PDF au format EFREI Mastère DE 2025-26
Sarah MALAEB - Guide de Rapport Projet Data Science.

Structure : 19 sections, 22-32 pages, charte EFREI, fpdf2 uniquement.

Usage :
    python scripts/06_build_report.py

Sortie :
    reports/06/rapport_projet_data_science.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

# Racine du projet
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bootstrap import ensure_dependencies  # noqa: E402

ensure_dependencies(verbose=False)

# ---------------------------------------------------------------------------
# Imports fpdf2
# ---------------------------------------------------------------------------
from fpdf import FPDF
from PIL import Image as PILImage

# ---------------------------------------------------------------------------
# Constantes et chemins
# ---------------------------------------------------------------------------
OUTPUT_DIR = ROOT / "reports" / "06"
OUTPUT_PDF = OUTPUT_DIR / "rapport_projet_data_science.pdf"

ASSETS = ROOT / "assets"
LOGO_PATH = ASSETS / "logo_efrei.png"

R02 = ROOT / "reports" / "02"
R03 = ROOT / "reports" / "03"
R04 = ROOT / "reports" / "04"
R05 = ROOT / "reports" / "05"
R07 = ROOT / "reports" / "07"
R08 = ROOT / "reports" / "08"
R10 = ROOT / "reports" / "10"

# Couleurs charte EFREI (RGB tuples)
NAVY = (13, 71, 161)
BLUE = (30, 136, 229)
LIGHT_GREY = (245, 247, 250)
BORDER_GREY = (200, 205, 215)
BLACK = (33, 33, 33)
GREEN_HIGHLIGHT = (220, 252, 231)
WHITE = (255, 255, 255)
CAPTION_GREY = (110, 110, 110)


# ---------------------------------------------------------------------------
# Helpers texte
# ---------------------------------------------------------------------------

def _safe(text: str) -> str:
    """Substitue les caractères hors latin-1 (emojis, smart quotes) par
    leurs équivalents ASCII. Les accents français (é à ç ô î ï û ù â ê ë)
    restent intacts car latin-1 les supporte nativement."""
    return (
        text
        # Emojis dashboard (U+1F3ED, U+1F6A8, U+1F4B6, U+1F527, U+1F52C)
        .replace("🏭", "[Parc]")
        .replace("🚨", "[Alerte]")
        .replace("💶", "[Impact]")
        .replace("🔧", "[Diag]")
        .replace("🔬", "[Tech]")
        # gear, check, cross
        .replace("⚙️", "")
        .replace("⚙", "")
        .replace("✅", "OK")
        .replace("❌", "X")
        # tirets longs et ponctuation type smart
        .replace("—", "·")  # em dash
        .replace("–", "-")        # en dash
        .replace("‘", "'").replace("’", "'")
        .replace("“", '"').replace("”", '"')
        .replace("…", "...")
        .replace(" ", " ")
        # encode latin-1 safe : les accents français sont préservés
        .encode("latin-1", errors="replace").decode("latin-1")
    )


# ---------------------------------------------------------------------------
# Classe FPDF custom
# ---------------------------------------------------------------------------

class EFREIPDF(FPDF):
    def __init__(self) -> None:
        super().__init__("P", "mm", "A4")
        self.is_cover = False
        self.set_margins(22, 22, 22)
        self.set_auto_page_break(auto=True, margin=28)
        self.alias_nb_pages()

    def header(self) -> None:
        if self.is_cover:
            return
        self.set_y(10)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*CAPTION_GREY)
        self.cell(80, 4, "Maintenance Prédictive Industrielle", border=0, align="L")
        self.set_x(self.w - 22 - 80)
        self.cell(80, 4, "EFREI Mastère DE&IA · 2025-2026", border=0, align="R")
        self.set_draw_color(*NAVY)
        self.set_line_width(0.3)
        self.line(22, 16, self.w - 22, 16)
        self.set_y(20)

    def footer(self) -> None:
        if self.is_cover:
            return
        self.set_y(-18)
        self.set_draw_color(*BORDER_GREY)
        self.set_line_width(0.2)
        self.line(22, self.get_y(), self.w - 22, self.get_y())
        self.ln(2)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*CAPTION_GREY)
        self.cell(80, 4, "Adam BELOUCIF · Emilien MORICE", border=0, align="L")
        self.cell(0, 4, f"Page {self.page_no()} / {{nb}}", border=0, align="R")


# ---------------------------------------------------------------------------
# Helpers mise en page
# ---------------------------------------------------------------------------

def _ensure_space(pdf: EFREIPDF, mm_needed: float) -> None:
    """Saut de page si moins de mm_needed reste avant la zone footer."""
    if pdf.get_y() + mm_needed > pdf.h - 28:
        pdf.add_page()


def h1(pdf: EFREIPDF, num: int, text: str) -> None:
    """Chaque H1 commence une nouvelle page."""
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(*NAVY)
    pdf.ln(2)
    pdf.cell(
        0, 9, _safe(f"{num}. {text}"),
        new_x="LMARGIN", new_y="NEXT", align="L",
    )
    pdf.set_draw_color(*NAVY)
    pdf.set_line_width(0.4)
    pdf.line(22, pdf.get_y(), pdf.w - 22, pdf.get_y())
    pdf.ln(4)


def h2(pdf: EFREIPDF, num: str, text: str) -> None:
    _ensure_space(pdf, 22)
    pdf.set_font("Helvetica", "B", 12.5)
    pdf.set_text_color(*BLUE)
    pdf.ln(3)
    pdf.cell(
        0, 7, _safe(f"{num} {text}"),
        new_x="LMARGIN", new_y="NEXT", align="L",
    )
    pdf.ln(1.5)


def h3(pdf: EFREIPDF, text: str) -> None:
    _ensure_space(pdf, 18)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(*BLACK)
    pdf.ln(2)
    pdf.cell(
        0, 6, _safe(text),
        new_x="LMARGIN", new_y="NEXT", align="L",
    )
    pdf.ln(1)


def p(pdf: EFREIPDF, text: str, justified: bool = True) -> None:
    pdf.set_font("Helvetica", "", 10.5)
    pdf.set_text_color(*BLACK)
    pdf.multi_cell(
        0, 5.2, _safe(text),
        align="J" if justified else "L",
        new_x="LMARGIN", new_y="NEXT",
    )
    pdf.ln(2)


def bullet(pdf: EFREIPDF, text: str) -> None:
    pdf.set_font("Helvetica", "", 10.5)
    pdf.set_text_color(*BLACK)
    pdf.set_x(26)
    pdf.multi_cell(
        162, 5.2, _safe("·  " + text),
        align="L",
        new_x="LMARGIN", new_y="NEXT",
    )
    pdf.ln(0.5)


def code_block(pdf: EFREIPDF, text: str) -> None:
    """Bloc code style monospace avec fond gris."""
    lines = _safe(text).split("\n")
    needed = len(lines) * 4.5 + 6
    _ensure_space(pdf, needed)
    pdf.set_fill_color(*LIGHT_GREY)
    pdf.set_draw_color(*BORDER_GREY)
    pdf.set_line_width(0.2)
    x0 = pdf.get_x()
    y0 = pdf.get_y()
    pdf.rect(22, y0, 166, needed - 4, "FD")
    pdf.set_y(y0 + 2)
    pdf.set_font("Courier", "", 8.5)
    pdf.set_text_color(*BLACK)
    for line in lines:
        pdf.set_x(25)
        pdf.cell(160, 4.5, line, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)


def add_figure(
    pdf: EFREIPDF,
    img_path: Path,
    caption: str,
    max_width_mm: float = 120,
) -> None:
    if not img_path.exists():
        print(f"  [SKIP] figure absente · {img_path.name}")
        return
    try:
        iw, ih = PILImage.open(img_path).size
    except Exception:
        iw, ih = 1, 1
    ratio = ih / iw if iw > 0 else 0.62
    w = min(max_width_mm, 166)
    h = w * ratio
    needed = h + 10
    _ensure_space(pdf, needed)
    x = (pdf.w - w) / 2
    y = pdf.get_y()
    pdf.image(str(img_path), x=x, y=y, w=w, h=h)
    pdf.set_y(y + h + 1)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(*CAPTION_GREY)
    pdf.multi_cell(
        0, 4, _safe(caption),
        align="C", new_x="LMARGIN", new_y="NEXT",
    )
    pdf.ln(4)


# Compteur global figures
_fig_counter = [0]


def _next_fig(caption: str) -> str:
    _fig_counter[0] += 1
    return f"Figure {_fig_counter[0]} - {caption}"


def add_figure_captioned(
    pdf: EFREIPDF,
    img_path: Path,
    caption: str,
    max_width_mm: float = 120,
) -> None:
    add_figure(pdf, img_path, _next_fig(caption), max_width_mm)


def make_table(
    pdf: EFREIPDF,
    headers: list[str],
    rows: list[list[str]],
    col_widths_mm: list[float],
    highlight_row_idx: int | None = None,
    header_row_height: float = 7,
    body_row_height: float = 6,
) -> None:
    """Tableau avec header navy, lignes alternées, highlight optionnel."""
    total_w = sum(col_widths_mm)
    needed = header_row_height + len(rows) * body_row_height + 6
    _ensure_space(pdf, needed)

    x_start = (pdf.w - total_w) / 2

    # Header
    pdf.set_x(x_start)
    pdf.set_fill_color(*NAVY)
    pdf.set_text_color(*WHITE)
    pdf.set_font("Helvetica", "B", 9.5)
    pdf.set_draw_color(*BORDER_GREY)
    pdf.set_line_width(0.2)
    for w, head in zip(col_widths_mm, headers):
        pdf.cell(w, header_row_height, _safe(str(head)), border=1, align="C", fill=True)
    pdf.ln(header_row_height)

    # Body
    for i, row in enumerate(rows):
        pdf.set_x(x_start)
        if highlight_row_idx is not None and i == highlight_row_idx:
            pdf.set_fill_color(*GREEN_HIGHLIGHT)
            pdf.set_font("Helvetica", "B", 9)
        elif i % 2 == 0:
            pdf.set_fill_color(*WHITE)
            pdf.set_font("Helvetica", "", 9)
        else:
            pdf.set_fill_color(*LIGHT_GREY)
            pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(*BLACK)
        for w, cell_value in zip(col_widths_mm, row):
            pdf.cell(w, body_row_height, _safe(str(cell_value)), border=1, align="C", fill=True)
        pdf.ln(body_row_height)

    pdf.ln(3)


# ---------------------------------------------------------------------------
# Compteurs de sections
# ---------------------------------------------------------------------------
_h1_counter = [0]
_h2_counter = [0]


def _next_h1() -> int:
    _h1_counter[0] += 1
    _h2_counter[0] = 0
    return _h1_counter[0]


def _next_h2() -> str:
    _h2_counter[0] += 1
    return f"{_h1_counter[0]}.{_h2_counter[0]}"


# ---------------------------------------------------------------------------
# Page de garde
# ---------------------------------------------------------------------------

def section_1_cover(pdf: EFREIPDF) -> None:
    """Page de garde - is_cover=True, pas de header/footer."""
    pdf.is_cover = True
    pdf.add_page()

    # Logo EFREI
    if LOGO_PATH.exists():
        try:
            iw, ih = PILImage.open(LOGO_PATH).size
            logo_w = 60.0
            logo_h = logo_w * (ih / iw)
        except Exception:
            logo_w, logo_h = 60.0, 30.0
        pdf.image(str(LOGO_PATH), x=75, y=50, w=logo_w, h=logo_h)
        pdf.set_y(50 + logo_h + 5)
    else:
        print(f"  [SKIP] Logo EFREI manquant : {LOGO_PATH}")
        pdf.set_y(85)

    # Titre principal
    pdf.set_y(95)
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(*NAVY)
    pdf.cell(0, 12, "Système Intelligent Multi-Modèles", new_x="LMARGIN", new_y="NEXT", align="C")

    # Sous-titre
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(*BLUE)
    pdf.cell(0, 10, "Maintenance Prédictive Industrielle", new_x="LMARGIN", new_y="NEXT", align="C")

    # Ligne décorative navy
    pdf.set_draw_color(*NAVY)
    pdf.set_line_width(0.5)
    pdf.line(70, 124, 140, 124)

    # Bloc info à partir de y=140
    pdf.set_y(140)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(*BLACK)

    for line in [
        "Projet Data Science",
        "M1 Mastère Data Engineering & IA · EFREI Paris Panthéon-Assas",
        "Année 2025-2026",
    ]:
        pdf.cell(0, 7, line, new_x="LMARGIN", new_y="NEXT", align="C")

    pdf.ln(4)

    for line in [
        "Adam BELOUCIF · n° 20220055",
        "Emilien MORICE",
        "Tutrice / Formatrice : Sarah MALAEB",
    ]:
        pdf.cell(0, 7, line, new_x="LMARGIN", new_y="NEXT", align="C")

    # Date
    pdf.set_y(210)
    pdf.set_font("Helvetica", "I", 11)
    pdf.set_text_color(*BLACK)
    pdf.cell(0, 7, "Avril 2026", new_x="LMARGIN", new_y="NEXT", align="C")

    # Bloc RNCP
    pdf.set_y(255)
    pdf.set_font("Helvetica", "I", 9.5)
    pdf.set_text_color(*NAVY)
    pdf.cell(0, 5, "Projet certifiant RNCP40875 · Expert en Ingénierie de Données", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.cell(0, 5, "Bloc 2 · Pilotage et implémentation de solutions IA", new_x="LMARGIN", new_y="NEXT", align="C")

    # Désactiver le mode cover avant la prochaine page
    pdf.is_cover = False


# ---------------------------------------------------------------------------
# Table des matières
# ---------------------------------------------------------------------------

def section_toc(pdf: EFREIPDF) -> None:
    """Table des matières (page 2)."""
    pdf.add_page()

    # Titre TOC (pas numéroté)
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(*NAVY)
    pdf.ln(2)
    pdf.cell(0, 9, "Table des matières", new_x="LMARGIN", new_y="NEXT", align="L")
    pdf.set_draw_color(*NAVY)
    pdf.set_line_width(0.4)
    pdf.line(22, pdf.get_y(), pdf.w - 22, pdf.get_y())
    pdf.ln(6)

    toc_entries = [
        ("1.", "Page de garde"),
        ("2.", "Résumé exécutif"),
        ("3.", "Introduction et Contexte"),
        ("4.", "Analyse du besoin utilisateur"),
        ("5.", "Méthodologie de travail et gestion de projet"),
        ("6.", "Référentiel de données"),
        ("7.", "Analyse exploratoire et visualisation"),
        ("8.", "Préparation et transformation des données"),
        ("9.", "Pipeline IA et architecture"),
        ("10.", "Implémentation technique"),
        ("11.", "Évaluation comparative des modèles"),
        ("12.", "Interprétabilité et analyse métier"),
        ("13.", "Interface utilisateur et prototype"),
        ("14.", "API REST"),
        ("15.", "Résultats et tests de démonstration"),
        ("16.", "Gouvernance, responsabilité et limites"),
        ("17.", "Limites et pistes d'amélioration"),
        ("18.", "Conclusion"),
        ("19.", "Annexes"),
    ]

    for num, title in toc_entries:
        _ensure_space(pdf, 8)
        pdf.set_font("Helvetica", "B", 10.5)
        pdf.set_text_color(*NAVY)
        pdf.set_x(22)
        pdf.cell(14, 6, num, border=0, align="L")
        pdf.set_font("Helvetica", "", 10.5)
        pdf.set_text_color(*BLACK)
        pdf.cell(0, 6, title, border=0, align="L", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(0.5)


# ---------------------------------------------------------------------------
# Sections 2-19
# ---------------------------------------------------------------------------

def section_2_executive_summary(pdf: EFREIPDF) -> None:
    n = _next_h1()
    h1(pdf, n, "Résumé exécutif")

    p(pdf, (
        "Ce projet répond à une problématique opérationnelle concrète : comment anticiper les "
        "défaillances de machines industrielles pour éliminer les arrêts non planifiés, source "
        "majeure de pertes économiques en milieu de production ? En utilisant un dataset public "
        "Kaggle (CC0 v3.0, 24 042 observations, 15 colonnes) simulant des capteurs IoT "
        "industriels, nous avons construit un système multi-tâches couvrant trois types de "
        "prédiction complémentaires."
    ))

    p(pdf, (
        "La tâche principale, classification binaire, prédit si une machine tombera en panne "
        "dans les 24 heures (cible failure_within_24h). La tâche secondaire, classification "
        "multiclasse à 5 catégories, identifie le type de défaillance probable parmi : "
        "bearing, motor_overheat, hydraulic, electrical, none. La tâche tertiaire, régression, "
        "estime la Remaining Useful Life (RUL) en heures."
    ))

    p(pdf, (
        "Douze modèles ont été entraînés (4 par tâche : Logistic Regression, Random Forest, "
        "XGBoost, MLP). Le modèle final retenu pour la classification binaire est XGBoost "
        "(F1 = 0.886, ROC-AUC = 0.995, PR-AUC = 0.974, stabilité CV 5-fold : 0.886 +/- 0.011). "
        "Pour la régression RUL, le Random Forest obtient le meilleur compromis "
        "(MAE = 9.57 h, R2 = 0.673)."
    ))

    p(pdf, (
        "Un seuil de décision cost-sensitive a été optimisé à 0.23 (contre 0.50 par défaut), "
        "permettant une économie estimée à 12 000 EUR par cycle de scoring en réduisant les "
        "faux négatifs coûteux. L'ensemble du système est délivré sous la forme d'un dashboard "
        "Streamlit 5 onglets orienté métier et d'une API REST FastAPI avec 4 endpoints, "
        "tous documentés et testés (23 tests pytest)."
    ))


def section_3_introduction(pdf: EFREIPDF) -> None:
    n = _next_h1()
    h1(pdf, n, "Introduction et Contexte")

    p(pdf, (
        "La maintenance des équipements industriels représente un enjeu économique et "
        "opérationnel critique. Selon les estimations sectorielles, une heure d'arrêt non "
        "planifié coûte entre 5 000 EUR et 50 000 EUR selon le type d'industrie "
        "(automobile, chimie, énergie). Face à ce constat, trois approches coexistent :"
    ))

    h2(pdf, f"{n}.1", "Trois stratégies de maintenance")
    make_table(
        pdf,
        ["Stratégie", "Principe", "Limites"],
        [
            ["Corrective", "Réparer après la panne", "Coûts élevés, arrêts longs"],
            ["Préventive", "Remplacer à intervalles fixes", "Remplacement prématuré, gaspillage"],
            ["Prédictive", "Intervenir si les capteurs le signalent", "Nécessite données + ML"],
        ],
        [38, 65, 63],
    )

    p(pdf, (
        "La maintenance prédictive s'appuie sur l'apprentissage automatique supervisé pour "
        "transformer les signaux capteurs (vibrations, température, courant, pression, RPM) "
        "en décisions d'intervention. Cette approche nécessite trois composantes : des données "
        "historiques labellisées, un pipeline de prétraitement robuste, et des modèles "
        "capables de généraliser."
    ))

    h2(pdf, f"{n}.2", "Articulation des trois tâches")
    p(pdf, (
        "Le projet découpe le problème en trois tâches complémentaires, chacune répondant "
        "à une question métier distincte :"
    ))
    for item in [
        "Binaire (failure_within_24h) : 'Cette machine va-t-elle tomber en panne dans "
        "les 24 heures ?' Répond à l'urgence d'intervention.",
        "Multiclasse (failure_type) : 'Quel type de panne est probable ?' Oriente le "
        "technicien vers la bonne pièce de rechange.",
        "Régression (rul_hours) : 'Combien d'heures de fonctionnement reste-t-il ?' "
        "Permet la planification de maintenance sur le long terme.",
    ]:
        bullet(pdf, item)


def section_4_analyse_besoin(pdf: EFREIPDF) -> None:
    n = _next_h1()
    h1(pdf, n, "Analyse du besoin utilisateur")

    h2(pdf, f"{n}.1", "Persona cible")
    p(pdf, (
        "Le persona principal est le responsable maintenance d'une usine industrielle : "
        "profil opérationnel, non data-scientist, qui supervise un parc de machines "
        "hétérogènes (CNC, pompes, compresseurs, bras robotiques). Il a besoin de "
        "prioriser ses équipes d'intervention, d'éviter les arrêts non planifiés et "
        "de justifier ses décisions à sa direction."
    ))

    h2(pdf, f"{n}.2", "Objectifs fonctionnels")
    for item in [
        "Prioriser les interventions : identifier les machines à risque élevé avant "
        "qu'elles ne tombent en panne.",
        "Éviter les arrêts non planifiés : anticiper via des alertes 24h avant la défaillance.",
        "Optimiser les ressources : ne pas sur-intervenir sur des machines saines "
        "(coûteux et perturbateur).",
        "Planification à moyen terme : disposer d'une estimation de RUL pour "
        "programmer les maintenances.",
    ]:
        bullet(pdf, item)

    h2(pdf, f"{n}.3", "Scénarios d'usage")
    for item in [
        "Matin : le responsable ouvre le dashboard, consulte l'état du parc "
        "(onglet 'État du parc'), identifie les machines critiques.",
        "Mi-journée : il envoie son équipe selon le 'Plan d'intervention' classé par "
        "niveau d'urgence.",
        "Diagnostic en atelier : le technicien saisit les valeurs capteurs en temps "
        "réel dans l'onglet 'Diagnostic machine' pour une recommandation immédiate.",
    ]:
        bullet(pdf, item)

    h2(pdf, f"{n}.4", "Contraintes et indicateurs")
    make_table(
        pdf,
        ["Contrainte", "Valeur cible", "Réalisation projet"],
        [
            ["Latence inférence", "< 500 ms", "< 10 us/sample (XGBoost)"],
            ["Recall pannes (binaire)", ">= 85%", "95.8% (XGBoost)"],
            ["Précision", "Acceptable", "82.4% (XGBoost)"],
            ["Interprétabilité", "Exigée", "SHAP + permutation + native FI"],
            ["Dashboard", "Orienté métier", "5 onglets, vocabulaire opérationnel"],
            ["API", "REST documentée", "FastAPI + Swagger auto /docs"],
        ],
        [45, 35, 86],
    )


def section_5_methodologie(pdf: EFREIPDF) -> None:
    n = _next_h1()
    h1(pdf, n, "Méthodologie de travail et gestion de projet")

    h2(pdf, f"{n}.1", "Approche hybride Agile / Kanban")
    p(pdf, (
        "Le binôme a adopté une organisation hybride inspirée d'Agile Kanban : sprints "
        "informels d'une semaine, revue hebdomadaire des livrables, tableau Notion pour "
        "le suivi des tickets. La communication s'est faite via Teams et les revues de "
        "code via pull requests sur le dépôt GitHub Adam-Blf."
    ))

    h2(pdf, f"{n}.2", "Répartition des rôles")
    make_table(
        pdf,
        ["Domaine", "Adam BELOUCIF", "Emilien MORICE"],
        [
            ["Data pipeline", "Pipeline preprocessing, anti-leakage", "Relecture EDA"],
            ["Modélisation", "Architecture src/, config, bootstrap", "EDA + modèles + tuning"],
            ["Interprétabilité", "Intégration rapport", "SHAP, permutation importance"],
            ["Dashboard", "App Streamlit 5 onglets, CSS custom", "Wireframes"],
            ["API", "FastAPI + Pydantic v2", "Tests intégration"],
            ["Tests", "pytest 23 tests (5 fichiers)", "Validation métier"],
        ],
        [40, 63, 63],
    )

    h2(pdf, f"{n}.3", "Grandes étapes du projet")
    make_table(
        pdf,
        ["Sprint", "Activités principales", "Livrables"],
        [
            ["S1 - Cadrage", "Sujet, dataset Kaggle, arborescence repo", "README, FACTS.md"],
            ["S2 - EDA", "scripts/02_eda.py, 8 figures, 2 CSV", "reports/02/"],
            ["S3 - Preprocessing", "ColumnTransformer, split 80/20", "data/processed/"],
            ["S4 - Train binaire", "4 modèles, CV 5-fold, sélection XGBoost", "models/*.joblib, reports/03/"],
            ["S5 - Interprétabilité", "SHAP, permutation, native FI", "reports/04/"],
            ["S6 - Multi + Rég", "5 classes failure_type, RUL régression", "reports/07-08/"],
            ["S7 - Tuning + Calibration", "Optuna 20 trials, seuil 0.23", "reports/09-10/, models/optimal_threshold.json"],
            ["S8 - Dashboard + API", "Streamlit 5 onglets, FastAPI", "dashboard/app.py, api/main.py"],
            ["S9 - Rapport + PPT", "PDF fpdf2 19 sections, slides", "reports/06/, reports/11/"],
        ],
        [30, 75, 61],
    )

    h2(pdf, f"{n}.4", "Risques rencontrés et solutions")
    for item in [
        "Déséquilibre des classes (~85.2% sains / ~14.8% pannes) : "
        "class_weight='balanced' pour LogReg/RF, scale_pos_weight pour XGBoost. "
        "Pas de SMOTE : moins efficace pour gradient boosting.",
        "NaN volontaires (2-4% sur 5 capteurs) : SimpleImputer median, "
        "imputation dans le pipeline anti-leakage (fit sur train uniquement).",
        "MLP overfitting : early_stopping=True, n_iter_no_change=10, "
        "alpha=1e-3. Score CV F1=0.795 contre 0.886 pour XGBoost.",
        "Anti-data-leakage : ColumnTransformer encapsulé dans sklearn.Pipeline "
        "avec l'estimateur, garantit que le préprocesseur n'est jamais fitté "
        "sur le test set (ADR 0003).",
    ]:
        bullet(pdf, item)


def section_6_referentiel_donnees(pdf: EFREIPDF) -> None:
    n = _next_h1()
    h1(pdf, n, "Référentiel de données")

    h2(pdf, f"{n}.1", "Source et licence")
    p(pdf, (
        "Dataset : Kaggle CC0 (domaine public) - tatheerabbas/industrial-machine-predictive-"
        "maintenance v3.0. URL : https://www.kaggle.com/datasets/tatheerabbas/"
        "industrial-machine-predictive-maintenance. "
        "Fichier local : data/raw/predictive_maintenance_v3.csv (~2.8 MB). "
        "Politique projet : CSV Kaggle officiel uniquement, aucun fallback synthétique "
        "(src/data_loader.py lève FileNotFoundError si absent)."
    ))

    h2(pdf, f"{n}.2", "Schéma des 15 colonnes")
    make_table(
        pdf,
        ["Colonne", "Type", "Unité / Domaine", "Rôle"],
        [
            ["timestamp", "str datetime", "YYYY-MM-DD HH:MM:SS", "ID temporel"],
            ["machine_id", "int", "1..N", "ID machine"],
            ["machine_type", "cat", "CNC, Pump, Compressor, Robotic Arm", "Feature cat."],
            ["vibration_rms", "float", "mm/s [0.35-10.0]", "Feature numérique"],
            ["temperature_motor", "float", "degC [28-95]", "Feature numérique"],
            ["current_phase_avg", "float", "A [2.2-35.0]", "Feature numérique"],
            ["pressure_level", "float", "bar [10.1-206.5]", "Feature numérique"],
            ["rpm", "float", "tr/min [124-4099]", "Feature numérique"],
            ["operating_mode", "cat", "normal, idle, peak", "Feature cat."],
            ["hours_since_maintenance", "float", "h [0-575.6]", "Feature numérique"],
            ["ambient_temp", "float", "degC [8-18]", "Feature numérique"],
            ["rul_hours", "float", "h [0.5-99.0]", "Cible régression"],
            ["failure_within_24h", "int 0/1", "binaire", "Cible binaire principale"],
            ["failure_type", "cat 5 classes", "none, bearing, motor_overheat, hydraulic, electrical", "Cible multiclasse"],
            ["estimated_repair_cost", "float", "EUR", "Cible régression secondaire"],
        ],
        [45, 28, 55, 38],
    )

    h2(pdf, f"{n}.3", "Volumétrie et qualité")
    make_table(
        pdf,
        ["Indicateur", "Valeur"],
        [
            ["Nombre de lignes", "24 042"],
            ["Nombre de colonnes", "15"],
            ["Déséquilibre cible binaire", "~14.8% pannes / ~85.2% sains"],
            ["NaN vibration_rms", "1 000 (4.16%)"],
            ["NaN pressure_level", "924 (3.84%)"],
            ["NaN temperature_motor", "834 (3.47%)"],
            ["NaN current_phase_avg", "731 (3.04%)"],
            ["NaN rpm", "533 (2.22%)"],
            ["Autres colonnes", "0 NaN"],
        ],
        [70, 96],
    )

    h2(pdf, f"{n}.4", "Limites du dataset")
    for item in [
        "Données simulées (pas un vrai flux IoT production) : pas de composante "
        "temporelle longue exploitable.",
        "Seulement 4 types de machines : les modèles ne généralisent pas à d'autres "
        "gammes d'équipements sans ré-entraînement.",
        "NaN volontaires : simulent des pannes capteurs, pas des données réellement "
        "manquantes au sens aléatoire.",
    ]:
        bullet(pdf, item)


def section_7_eda(pdf: EFREIPDF) -> None:
    n = _next_h1()
    h1(pdf, n, "Analyse exploratoire et visualisation")

    p(pdf, (
        "L'analyse exploratoire (script 02_eda.py) a produit 8 figures et 2 CSV "
        "dans reports/02/. Elle a permis de valider la qualité du dataset, de "
        "comprendre la distribution des cibles et d'identifier les capteurs les plus "
        "discriminants avant toute modélisation."
    ))

    h2(pdf, f"{n}.1", "Distribution de la cible et des types de panne")
    add_figure_captioned(
        pdf,
        R02 / "eda_target_distribution.png",
        "Distribution de la cible binaire failure_within_24h",
        max_width_mm=100,
    )
    add_figure_captioned(
        pdf,
        R02 / "eda_failure_type_distribution.png",
        "Répartition des 4 types de panne (sur machines en panne uniquement)",
        max_width_mm=100,
    )
    p(pdf, (
        "Le déséquilibre de classe (~14.8% pannes) justifie le recours au F1-score et "
        "à la PR-AUC plutôt qu'à l'accuracy seule. Les 4 types de panne ont des "
        "supports différents, ce qui rend la macro-F1 un indicateur pertinent pour "
        "la tâche multiclasse."
    ))

    h2(pdf, f"{n}.2", "Distributions des capteurs et discriminance")
    add_figure_captioned(
        pdf,
        R02 / "eda_sensor_distributions.png",
        "Distributions des 7 capteurs numériques (histogrammes + KDE)",
        max_width_mm=140,
    )
    add_figure_captioned(
        pdf,
        R02 / "eda_boxplots_by_class.png",
        "Boxplots des capteurs par classe OK / Panne (capteurs discriminants mis en évidence)",
        max_width_mm=140,
    )

    h2(pdf, f"{n}.3", "Corrélations et zone de risque")
    add_figure_captioned(
        pdf,
        R02 / "eda_correlation_heatmap.png",
        "Matrice de corrélation triangulaire (7 capteurs + cible binaire)",
        max_width_mm=110,
    )
    add_figure_captioned(
        pdf,
        R02 / "eda_scatter_vib_temp.png",
        "Scatter vibration x température - zone haute droite dominée par les pannes",
        max_width_mm=110,
    )

    h2(pdf, f"{n}.4", "Mode opératoire et valeurs manquantes")
    add_figure_captioned(
        pdf,
        R02 / "eda_operating_mode.png",
        "Comptage par mode opératoire et taux de panne (%) : le mode peak sur-risque",
        max_width_mm=120,
    )
    add_figure_captioned(
        pdf,
        R02 / "eda_missing_values.png",
        "Pourcentage de valeurs manquantes par colonne (5 capteurs, 2-4%)",
        max_width_mm=100,
    )

    h2(pdf, f"{n}.5", "Hypothèses métier dégagées")
    for item in [
        "vibration_rms et temperature_motor sont les 2 capteurs les plus discriminants "
        "(boxplots + scatter confirment une zone 'haute droite' quasi exclusivement pannes).",
        "Le mode opératoire peak concentre le taux de panne le plus élevé : "
        "les machines en surcharge sont à surveiller en priorité.",
        "Stats descriptives clés : vibration_rms moy = 1.62 mm/s, "
        "temperature_motor moy = 51.4 degC, rpm moy = 1145 tr/min, "
        "hours_since_maintenance moy = 172.6 h.",
        "Absence de corrélation critique entre capteurs : pas de redondance "
        "à supprimer, les 7 features numériques sont toutes informatives.",
    ]:
        bullet(pdf, item)


def section_8_preprocessing(pdf: EFREIPDF) -> None:
    n = _next_h1()
    h1(pdf, n, "Préparation, transformation des données et visualisation du prétraitement")

    h2(pdf, f"{n}.1", "Architecture anti-data-leakage")
    p(pdf, (
        "L'ensemble du prétraitement est encapsulé dans un sklearn.Pipeline "
        "(ColumnTransformer + estimateur), garantissant que le préprocesseur ne voit "
        "jamais le test set lors de l'ajustement. Cette architecture répond à l'ADR 0003 "
        "(docs/adr/0003-anti-data-leakage.md) et est le choix fondateur du pipeline."
    ))

    h2(pdf, f"{n}.2", "Branches du ColumnTransformer")
    make_table(
        pdf,
        ["Branche", "Features", "Étapes", "Justification"],
        [
            ["Numérique", "7 capteurs", "SimpleImputer(median) -> StandardScaler",
             "Median robuste aux outliers ; imputation avant scaling obligatoire"],
            ["Catégorielle", "machine_type, operating_mode",
             "SimpleImputer(most_frequent) -> OneHotEncoder(handle_unknown=ignore)",
             "handle_unknown=ignore protège l'inférence si modalité inédite"],
            ["Exclues (drop)", "timestamp, machine_id, cibles", "remainder='drop'",
             "Garantit que les cibles ne fuitent jamais comme features"],
        ],
        [28, 32, 55, 51],
    )

    h2(pdf, f"{n}.3", "Split et validation croisée")
    for item in [
        "Split : train_test_split(test_size=0.20, stratify=y, random_state=42) - "
        "stratification sur failure_within_24h pour préserver le ratio 25/75 dans "
        "les deux sous-ensembles.",
        "CV : StratifiedKFold(n_splits=5, shuffle=True, random_state=42) - "
        "le pipeline complet est réajusté à chaque fold (anti-leakage total).",
        "Sous-échantillonnage CV : 8 000 lignes pour limiter le temps de calcul "
        "tout en restant représentatif.",
    ]:
        bullet(pdf, item)

    h2(pdf, f"{n}.4", "Gestion du déséquilibre de classes")
    make_table(
        pdf,
        ["Modèle", "Stratégie", "Justification"],
        [
            ["LogisticRegression", "class_weight='balanced'",
             "Pondère les classes inversement à leur fréquence"],
            ["RandomForest", "class_weight='balanced'",
             "Idem, supporte nativement"],
            ["XGBoost", "scale_pos_weight = n_neg / n_pos",
             "Plus efficace pour gradient boosting que SMOTE"],
            ["MLP", "Pas de pondération",
             "early_stopping + alpha compensent partiellement"],
        ],
        [35, 50, 81],
    )

    p(pdf, (
        "Le SMOTE a été exclu car les méthodes de pondération intégrées sont plus "
        "efficaces pour les modèles basés arbres/gradient boosting et éliminent le "
        "risque de surapprentissage sur des exemples synthétiques."
    ))


def section_9_pipeline(pdf: EFREIPDF) -> None:
    n = _next_h1()
    h1(pdf, n, "Pipeline IA et architecture")

    h2(pdf, f"{n}.1", "Architecture globale")
    p(pdf, (
        "Le projet suit une architecture en 6 étapes numérotées (scripts 02 à 10) "
        "plus la phase de serving. Chaque étape produit des artefacts persistés "
        "(figures PNG, CSV métriques, modèles joblib) consommés par les étapes suivantes."
    ))
    add_figure_captioned(
        pdf,
        R05 / "diagram_architecture.png",
        "Architecture du système : data -> features -> models -> serving",
        max_width_mm=140,
    )

    h2(pdf, f"{n}.2", "Pipeline ML détaillé")
    add_figure_captioned(
        pdf,
        R05 / "diagram_ml_pipeline.png",
        "Pipeline ML : preprocessing -> entraînement -> évaluation -> calibration",
        max_width_mm=140,
    )

    h2(pdf, f"{n}.3", "Description des 6 étapes")
    make_table(
        pdf,
        ["Étape", "Script", "Entrées", "Sorties"],
        [
            ["02 - EDA", "02_eda.py",
             "data/raw/predictive_maintenance_v3.csv",
             "8 PNG + 2 CSV dans reports/02/"],
            ["03 - Train binaire", "03_train_models.py", "CSV raw",
             "5 modèles joblib, 6 figures, métriques CSV/JSON dans reports/03/"],
            ["04 - Interprétabilité", "04_interpret.py", "final_model.joblib",
             "8 figures SHAP/permutation dans reports/04/"],
            ["05 - Diagrammes", "05_generate_diagrams.py", "(aucun)",
             "4 PNG pédagogiques dans reports/05/"],
            ["07-08-09-10 - Bonus", "7 scripts", "CSV raw, modèles",
             "Multiclasse, régression, tuning, calibration dans reports/07-10/"],
            ["Serving", "dashboard/app.py + api/main.py", "final_model.joblib",
             "Interface Streamlit + API FastAPI"],
        ],
        [30, 40, 40, 56],
    )


def section_10_implementation(pdf: EFREIPDF) -> None:
    n = _next_h1()
    h1(pdf, n, "Implémentation technique")

    h2(pdf, f"{n}.1", "Stack technique")
    make_table(
        pdf,
        ["Domaine", "Outil / Librairie", "Version"],
        [
            ["Langage", "Python", "3.12"],
            ["Data", "NumPy, Pandas, SciPy", ">= 1.26 / 2.1 / 1.11"],
            ["ML", "scikit-learn, XGBoost, joblib", ">= 1.4 / 2.0 / 1.3"],
            ["Deep Learning", "MLPClassifier (scikit-learn)", "Inclus dans scikit-learn"],
            ["Tuning", "Optuna (TPE sampler)", ">= 3.5"],
            ["Interprétabilité", "SHAP (TreeExplainer)", ">= 0.44"],
            ["Écoresponsabilité", "CodeCarbon", ">= 2.3 (RNCP C4.3)"],
            ["Visualisation", "Matplotlib, Seaborn, Plotly", ">= 3.8 / 0.13 / 5.18"],
            ["Dashboard", "Streamlit", ">= 1.32"],
            ["API", "FastAPI + Pydantic v2 + Uvicorn", ">= 0.110 / 2.6 / 0.27"],
            ["PDF / PPTX", "fpdf2, python-pptx, Pillow", ">= 2.7 / 0.6.21 / 10.0"],
            ["Tests", "pytest, pytest-cov", ">= 8.0 / 4.1"],
        ],
        [40, 60, 66],
    )

    h2(pdf, f"{n}.2", "Justification du MLP scikit-learn comme composante Deep Learning")
    p(pdf, (
        "Le module MLPClassifier/MLPRegressor de scikit-learn (architecture "
        "64-32-16 neurones, activation relu, optimizer adam, early_stopping=True) "
        "constitue la composante Deep Learning du projet. Ce choix est documenté "
        "dans l'ADR 0001 (docs/adr/0001-stack-technique.md) : avec 24 042 lignes "
        "de données tabulaires, TensorFlow et PyTorch sont sur-dimensionnés "
        "(surapprentissage plus probable, complexité de déploiement accrue, "
        "temps d'itération plus long). Le MLP scikit-learn s'intègre directement "
        "dans le sklearn.Pipeline, garantissant l'anti-data-leakage."
    ))

    h2(pdf, f"{n}.3", "Arborescence du dépôt")
    tree_text = (
        "maintenance-predictive-industrielle/\n"
        "|- README.md\n"
        "|- LICENSE (MIT)\n"
        "|- requirements.txt\n"
        "|- api/main.py              (FastAPI - /predict, /health, /model-info)\n"
        "|- dashboard/app.py         (Streamlit - 5 onglets, ~1280 lignes)\n"
        "|- data/\n"
        "|  |- raw/predictive_maintenance_v3.csv  (24 042 x 15)\n"
        "|  `- processed/{X_train, X_test, y_train, y_test}.csv\n"
        "|- src/                     (13 modules Python)\n"
        "|  |- bootstrap.py          (auto-install deps)\n"
        "|  |- config.py             (paths, hyperparams, palette)\n"
        "|  |- data_loader.py        (load + validate schema)\n"
        "|  |- preprocessing.py      (ColumnTransformer)\n"
        "|  |- models.py             (4 builders binaire)\n"
        "|  |- models_multiclass.py  (4 builders multiclasse)\n"
        "|  |- models_regression.py  (4 builders regression)\n"
        "|  |- evaluation.py         (metrics + plots)\n"
        "|  |- interpretability.py   (SHAP + permutation + FI)\n"
        "|  |- calibration.py        (reliability + cost curve)\n"
        "|  |- tuning.py             (Optuna TPE)\n"
        "|  `- diagrams.py           (schemas pedagogiques)\n"
        "|- scripts/                 (10 scripts numerotes)\n"
        "|- models/                  (16 .joblib + JSON metadata)\n"
        "|- reports/                 (02/ 03/ 04/ 05/ 06/ 07/ 08/ 09/ 10/ 11/)\n"
        "|- tests/                   (5 fichiers, 23 tests pytest)\n"
        "|- docs/adr/                (0001-0004 Architecture Decision Records)\n"
        "`- assets/                  (logo_efrei.png x3 variantes)\n"
    )
    code_block(pdf, tree_text)


def section_11_evaluation(pdf: EFREIPDF) -> None:
    n = _next_h1()
    h1(pdf, n, "Évaluation comparative des modèles")

    h2(pdf, f"{n}.1", "Classification binaire (failure_within_24h)")
    p(pdf, (
        "Test set : 20% du dataset (4 809 observations, 712 pannes). "
        "Métrique principale : F1-score (compromis précision/recall adapté au "
        "déséquilibre). Métrique secondaire : ROC-AUC et PR-AUC."
    ))
    make_table(
        pdf,
        ["Modèle", "Accuracy", "Précision", "Recall", "F1", "ROC-AUC", "PR-AUC", "CV F1"],
        [
            ["Logistic Regression", "0.910", "0.641", "0.895", "0.747", "0.959", "0.838", "0.750 +/- 0.008"],
            ["Random Forest", "0.955", "0.791", "0.949", "0.863", "0.992", "0.954", "0.844 +/- 0.009"],
            ["XGBoost (FINAL)", "0.963", "0.824", "0.958", "0.886", "0.995", "0.974", "0.886 +/- 0.011"],
            ["MLP", "0.952", "0.842", "0.830", "0.836", "0.984", "0.909", "0.795 +/- 0.010"],
        ],
        [35, 18, 20, 18, 16, 20, 18, 21],
        highlight_row_idx=2,
    )
    add_figure_captioned(
        pdf,
        R03 / "roc_curves_comparison.png",
        "Courbes ROC superposées des 4 modèles binaires (TPR vs FPR)",
        max_width_mm=110,
    )
    add_figure_captioned(
        pdf,
        R03 / "pr_curves_comparison.png",
        "Courbes Précision-Recall superposées - XGBoost domine sur tout le spectre",
        max_width_mm=110,
    )
    add_figure_captioned(
        pdf,
        R03 / "confusion_matrix_xgboost.png",
        "Matrice de confusion XGBoost (modèle final binaire)",
        max_width_mm=90,
    )
    p(pdf, (
        "Justification du choix XGBoost : meilleur F1 test (0.886), meilleur ROC-AUC "
        "(0.995), meilleur PR-AUC (0.974), CV la plus stable (0.886 +/- 0.011), "
        "et latence inférence < 10 us/sample - compatible avec une API temps réel. "
        "Sélection via score composite F1_test - 0.5 * sigma(F1_CV)."
    ))

    h2(pdf, f"{n}.2", "Classification multiclasse (failure_type - 5 classes)")
    make_table(
        pdf,
        ["Modèle", "Accuracy", "Macro-F1", "Weighted-F1", "Fit time (s)"],
        [
            ["Logistic Regression", "0.843", "0.675", "0.870", "0.249"],
            ["Random Forest", "0.964", "0.899", "0.965", "0.958"],
            ["XGBoost (FINAL)", "0.977", "0.931", "0.977", "1.800"],
            ["MLP", "0.944", "0.812", "0.943", "2.784"],
        ],
        [40, 25, 25, 30, 46],
        highlight_row_idx=2,
    )
    p(pdf, (
        "Détail par classe XGBoost : bearing (F1=0.918, n=223), electrical (F1=0.906, n=131), "
        "hydraulic (F1=0.923, n=146), motor_overheat (F1=0.922, n=212), none (F1=0.987, n=4097)."
    ))
    add_figure_captioned(
        pdf,
        R07 / "multiclass_confusion_matrix.png",
        "Matrice de confusion 5x5 normalisée XGBoost (multiclasse)",
        max_width_mm=110,
    )

    h2(pdf, f"{n}.3", "Régression RUL (rul_hours)")
    make_table(
        pdf,
        ["Modèle", "MAE (h)", "RMSE (h)", "R2", "Fit time (s)"],
        [
            ["Ridge", "20.47", "24.39", "0.140", "0.023"],
            ["Random Forest (FINAL)", "9.57", "15.04", "0.673", "3.847"],
            ["XGBoost", "10.72", "15.43", "0.656", "0.545"],
            ["MLP", "11.89", "16.66", "0.599", "6.797"],
        ],
        [45, 25, 25, 20, 51],
        highlight_row_idx=1,
    )
    add_figure_captioned(
        pdf,
        R08 / "regression_pred_vs_true.png",
        "Prédictions vs vraies valeurs RUL - Random Forest (diagonale = prédiction parfaite)",
        max_width_mm=110,
    )
    p(pdf, (
        "Justification du choix Random Forest pour la régression : meilleur R2 (0.673) "
        "et meilleur MAE (9.57 h), supérieur à XGBoost et MLP sur les deux indicateurs."
    ))


def section_12_interpretabilite(pdf: EFREIPDF) -> None:
    n = _next_h1()
    h1(pdf, n, "Interprétabilité et analyse métier")

    p(pdf, (
        "L'interprétabilité est déployée sur trois niveaux complémentaires, couvrant "
        "les critères RNCP C3.1 / C3.2 / C3.3 (cf. docs/CONFORMITE_RNCP.md). "
        "Chaque niveau répond à une question différente : 'quelles features utilise "
        "le modèle ?', 'lesquelles sont vraiment importantes si on les perturbe ?', "
        "et 'pourquoi cette prédiction sur cette observation ?'"
    ))

    h2(pdf, f"{n}.1", "Niveau 1 : Feature importance native XGBoost")
    add_figure_captioned(
        pdf,
        R04 / "feature_importance_native_xgboost.png",
        "Feature importance native XGBoost (gain moyen par feature dans les arbres)",
        max_width_mm=120,
    )

    h2(pdf, f"{n}.2", "Niveau 2 : Permutation importance")
    add_figure_captioned(
        pdf,
        R04 / "permutation_importance_xgboost.png",
        "Permutation importance XGBoost (perte de F1 si la feature est bruitée)",
        max_width_mm=120,
    )

    h2(pdf, f"{n}.3", "Niveau 3 : SHAP (TreeExplainer, max_samples=400)")
    add_figure_captioned(
        pdf,
        R04 / "shap_summary_xgboost.png",
        "SHAP dot plot XGBoost - contribution par observation aux prédictions",
        max_width_mm=140,
    )

    h2(pdf, f"{n}.4", "Traduction métier")
    p(pdf, (
        "Les trois niveaux convergent vers la même conclusion : vibration_rms et "
        "temperature_motor expliquent environ 70% de la décision de classement. "
        "Cette convergence produit une règle métier interprétable directement "
        "actionnable par les techniciens :"
    ))
    make_table(
        pdf,
        ["Condition capteurs", "Niveau de risque", "Action recommandée"],
        [
            ["vibration_rms > seuil ET temperature_motor > seuil", "HIGH",
             "Intervention préventive dans les 12-24h"],
            ["Un seul capteur au-dessus du seuil", "MODERATE",
             "Contrôle visuel sous 48h"],
            ["Tous les capteurs sous seuil", "LOW",
             "Surveillance continue, aucune action"],
        ],
        [55, 30, 81],
    )
    p(pdf, (
        "Le mode opératoire 'peak' amplifie le risque indépendamment des valeurs "
        "capteurs absolues : les machines en surcharge doivent être surveillées "
        "même si vibration et température restent modérées."
    ))


def section_13_dashboard(pdf: EFREIPDF) -> None:
    n = _next_h1()
    h1(pdf, n, "Interface utilisateur et prototype")

    p(pdf, (
        "Le dashboard est développé avec Streamlit (>= 1.32) et une couche CSS "
        "custom inspirée de la charte EFREI (Plus Jakarta Sans, palette bleu/vert/"
        "orange/rouge, design tokens type Linear/Vercel). Il est exclusivement "
        "orienté vers les besoins métier du responsable maintenance, avec un "
        "vocabulaire opérationnel (pas de F1/ROC/SHAP sur les onglets principaux)."
    ))

    h2(pdf, f"{n}.1", "5 onglets du dashboard")
    make_table(
        pdf,
        ["Onglet", "Audience", "Contenu principal"],
        [
            ["État du parc", "Responsable maintenance",
             "Vue temps réel des 20 machines, top alertes, KPIs parc"],
            ["Plan d'intervention", "Chef d'équipe",
             "Table d'actions priorisées par niveau d'urgence (high/moderate/low)"],
            ["Impact économique", "Direction",
             "ROI, coûts évités, comparaison avec/sans IA, économie 12k EUR seuil"],
            ["Diagnostic machine", "Technicien",
             "Saisie capteurs -> prédiction temps réel -> recommandation actionnable"],
            ["Détails techniques", "DSI / Jury",
             "3 sous-onglets : Données (EDA), Modèles (métriques), Interprétabilité (SHAP)"],
        ],
        [35, 35, 96],
    )

    h2(pdf, f"{n}.2", "Distinction EDA vs visualisations métier")
    p(pdf, (
        "Le dashboard opère une distinction claire entre les visualisations scientifiques "
        "(matrices de confusion, courbes ROC, SHAP dot plots - reléguées dans l'onglet "
        "'Détails techniques' pour le jury et la DSI) et les visualisations métier "
        "(jauges d'alerte, plans d'intervention, calculs d'économie - accessibles "
        "directement depuis les onglets principaux)."
    ))

    h2(pdf, f"{n}.3", "Lancement")
    code_block(
        pdf,
        "# Depuis la racine du projet :\n"
        "streamlit run dashboard/app.py\n\n"
        "# L'application s'ouvre sur http://localhost:8501\n"
        "# Le modele est charge en lazy-load au premier appel\n",
    )
    p(pdf, (
        "Note : les captures d'écran live du dashboard ne sont pas incluses dans "
        "ce rapport PDF car Streamlit nécessite un serveur actif pour le rendu. "
        "Le code source complet est disponible dans dashboard/app.py (~1280 lignes)."
    ))


def section_14_api(pdf: EFREIPDF) -> None:
    n = _next_h1()
    h1(pdf, n, "API REST")

    p(pdf, (
        "L'API est développée avec FastAPI 0.110, Pydantic v2 et Uvicorn. "
        "Elle expose le modèle final XGBoost via 4 endpoints REST documentés "
        "automatiquement sous /docs (Swagger UI)."
    ))

    h2(pdf, f"{n}.1", "Endpoints")
    make_table(
        pdf,
        ["Méthode", "Route", "Description"],
        [
            ["GET", "/", "Message d'accueil + liens vers /docs et /health"],
            ["GET", "/health", "Status (healthy/degraded), model_loaded, api_version, timestamp_utc"],
            ["GET", "/model-info", "model_name, métriques, features_required, operating_modes"],
            ["POST", "/predict", "Reçoit SensorReading (9 champs), renvoie PredictionResponse"],
        ],
        [20, 30, 116],
    )

    h2(pdf, f"{n}.2", "Schéma Pydantic et validation")
    p(pdf, (
        "Pydantic v2 valide automatiquement les bornes de chaque capteur (ge/le). "
        "Toute valeur hors plage retourne un code HTTP 422 avec détail de l'erreur. "
        "Exemple : vibration_rms < 0 ou operating_mode hors {normal, idle, peak} "
        "déclenchent une réponse 422 sans que le modèle soit chargé."
    ))

    h2(pdf, f"{n}.3", "Exemple JSON - entrée")
    code_block(
        pdf,
        '{\n'
        '  "vibration_rms": 1.2,\n'
        '  "temperature_motor": 48.5,\n'
        '  "current_phase_avg": 12.3,\n'
        '  "pressure_level": 45.0,\n'
        '  "rpm": 1100,\n'
        '  "operating_mode": "normal",\n'
        '  "machine_type": "CNC",\n'
        '  "hours_since_maintenance": 80.0,\n'
        '  "ambient_temp": 22.0\n'
        '}',
    )

    h2(pdf, f"{n}.4", "Exemple JSON - sortie (cas faible risque)")
    code_block(
        pdf,
        '{\n'
        '  "failure_within_24h": 0,\n'
        '  "probability": 0.07,\n'
        '  "risk_level": "low",\n'
        '  "recommendation": "Aucune action requise. Surveillance continue.",\n'
        '  "model_name": "xgboost",\n'
        '  "timestamp_utc": "2026-04-29T08:00:00Z"\n'
        '}',
    )

    h2(pdf, f"{n}.5", "Lancement")
    code_block(
        pdf,
        "uvicorn api.main:app --reload --host 127.0.0.1 --port 8000\n"
        "# Swagger UI disponible sur http://127.0.0.1:8000/docs\n",
    )


def section_15_resultats(pdf: EFREIPDF) -> None:
    n = _next_h1()
    h1(pdf, n, "Résultats et tests de démonstration")

    h2(pdf, f"{n}.1", "Scénarios de démonstration")
    make_table(
        pdf,
        ["Paramètre", "Cas faible risque", "Cas haut risque"],
        [
            ["vibration_rms", "0.6 mm/s", "4.5 mm/s"],
            ["temperature_motor", "42 degC", "82 degC"],
            ["rpm", "1100 tr/min", "2900 tr/min"],
            ["operating_mode", "normal", "peak"],
            ["machine_type", "CNC", "Compressor"],
            ["hours_since_maintenance", "30 h", "520 h"],
            ["Probabilité prédite", "~0.07", "~0.89"],
            ["Risk level", "low", "high"],
            ["Recommandation", "Surveillance continue", "Intervention dans les 12-24h"],
        ],
        [50, 58, 58],
    )

    h2(pdf, f"{n}.2", "Suite de tests pytest")
    make_table(
        pdf,
        ["Fichier de test", "Nb tests", "Ce qui est vérifié"],
        [
            ["test_smoke.py", "5",
             "Imports, chemins config, schéma 15 colonnes, ColumnTransformer, MODEL_CATALOG"],
            ["test_preprocessing.py", "3",
             "fit_transform produit 2D, imputation élimine 100% NaN, >= 13 feature names"],
            ["test_models.py", "8",
             "4 modèles binaires fit+predict_proba, multiclasse classes valides, "
             "régression floats, KeyError sur nom invalide"],
            ["test_evaluation.py", "4",
             "Classifieur parfait -> métriques=1.0, zeroRecall=0, to_dict clés complètes"],
            ["test_api.py", "3",
             "/health 200, /predict valide 200, valeur hors plage 422"],
        ],
        [45, 20, 101],
    )

    h2(pdf, f"{n}.3", "Optimisation cost-sensitive et calibration")
    add_figure_captioned(
        pdf,
        R10 / "cost_threshold_xgboost.png",
        "Courbe coût(seuil) XGBoost - seuil optimal 0.23, économie 12k EUR/cycle",
        max_width_mm=110,
    )
    add_figure_captioned(
        pdf,
        R10 / "reliability_diagram_xgboost.png",
        "Reliability diagram XGBoost - calibration probabiliste du modèle final",
        max_width_mm=110,
    )
    p(pdf, (
        "Paramètres cost-sensitive : FN = 1 000 EUR (panne ratée), FP = 100 EUR "
        "(alerte inutile). Au seuil 0.5 : FN=30, FP=146 -> coût = 44 600 EUR. "
        "Au seuil 0.23 : FN=7, FP=256 -> coût = 32 600 EUR. "
        "Économie : 12 000 EUR par cycle de scoring."
    ))


def section_16_gouvernance(pdf: EFREIPDF) -> None:
    n = _next_h1()
    h1(pdf, n, "Gouvernance, responsabilité et limites")

    h2(pdf, f"{n}.1", "Qualité et traçabilité des données")
    for item in [
        "Dataset versionné (v3.0 Kaggle CC0) avec hash SHA256 vérifié à chaque "
        "chargement via pandera (src/data_loader.py).",
        "RANDOM_STATE = 42 propagé dans tous les scripts via src/config.py : "
        "garantit la reproductibilité exacte des résultats.",
        "Modèles sérialisés en joblib avec métadonnées JSON (model_name, "
        "metrics, features_required) dans models/.",
        "Seuil optimal persisté dans models/optimal_threshold.json pour "
        "utilisation cohérente par le dashboard et l'API.",
    ]:
        bullet(pdf, item)

    h2(pdf, f"{n}.2", "Limites du dataset")
    for item in [
        "Données simulées (pas un vrai flux capteurs IoT) : les distributions "
        "sont construites, pas observées en production.",
        "Seulement 4 types de machines (CNC, Pump, Compressor, Robotic Arm) : "
        "risque de biais si déployé sur d'autres gammes.",
        "Pas de dimension temporelle longue exploitable : les événements ne sont "
        "pas sériellement corrélés.",
    ]:
        bullet(pdf, item)

    h2(pdf, f"{n}.3", "Risques et mesures de mitigation")
    make_table(
        pdf,
        ["Risque", "Impact", "Mitigation en place"],
        [
            ["Surapprentissage", "Modèle ne généralise pas",
             "CV 5-fold stratifiée, early_stopping MLP, test set isolé"],
            ["Dérive des données", "Performances dégradées en prod",
             "Limite - non monitoré (recommandation : Evidently AI)"],
            ["Biais de sélection", "4 types machines uniquement",
             "Documenté dans CONFORMITE_RNCP.md"],
            ["API CORS permissive", "Accès non autorisé",
             "allow_origins=['*'] à restreindre en prod (commenté dans le code)"],
            ["Dashboard sans auth", "Accès local uniquement",
             "Acceptable pour démo, OAuth2 recommandé en prod"],
        ],
        [35, 40, 91],
    )

    h2(pdf, f"{n}.4", "Recommandations pour la production")
    for item in [
        "Monitoring de la dérive (data drift et concept drift) : Evidently AI "
        "ou Great Expectations.",
        "Réentraînement trimestriel avec injection des nouvelles données capteurs.",
        "Validation humaine systématique pour les cas high-risk avant intervention.",
        "Documentation des décisions techniques via les 4 ADR existants "
        "(docs/adr/0001-0004) et extension au fil des évolutions.",
    ]:
        bullet(pdf, item)


def section_17_limites(pdf: EFREIPDF) -> None:
    n = _next_h1()
    h1(pdf, n, "Limites et pistes d'amélioration")

    h2(pdf, f"{n}.1", "Ce qui a fonctionné")
    for item in [
        "Pipeline anti-data-leakage complet : ColumnTransformer + Pipeline "
        "sklearn garantit l'intégrité de l'évaluation.",
        "Couverture des 3 tâches (binaire + multiclasse + régression) sur "
        "un seul dataset : pédagogique et métier.",
        "12 modèles entraînés et évalués de manière reproductible (RANDOM_STATE=42).",
        "Dashboard métier 5 onglets et API 4 endpoints déployés et testés.",
        "Interprétabilité sur 3 niveaux (native FI + permutation + SHAP).",
    ]:
        bullet(pdf, item)

    h2(pdf, f"{n}.2", "Difficultés rencontrées")
    for item in [
        "Déséquilibre des classes (25/75) : nécessite une stratégie spécifique "
        "par modèle, SMOTE écarté, class_weight retenu.",
        "MLP overfitting : malgré early_stopping et régularisation L2, "
        "le MLP reste inférieur à XGBoost (F1 0.836 vs 0.886).",
        "Optuna documentaire : les best_params trouvés (XGBoost : n_est=500, "
        "lr=0.084, max_depth=10) ne sont pas réinjectés dans src/models.py "
        "pour ne pas invalider les résultats déjà calculés.",
    ]:
        bullet(pdf, item)

    h2(pdf, f"{n}.3", "Pistes d'amélioration prioritaires")
    make_table(
        pdf,
        ["Priorité", "Amélioration", "Bénéfice attendu"],
        [
            ["1 - Haute", "Réinjecter les best_params Optuna dans models.py",
             "Gain estimé +0.01-0.02 F1 sur XGBoost"],
            ["2 - Haute", "Monitoring drift (Evidently AI)",
             "Détection précoce de la dégradation en production"],
            ["3 - Moyenne", "Déploiement cloud (Azure ML / AWS SageMaker)",
             "Scalabilité, haute disponibilité, CI/CD ML"],
            ["4 - Moyenne", "Base PostgreSQL pour historiser les prédictions",
             "Audit, traçabilité, ré-entraînement incrémental"],
            ["5 - Basse", "Authentification API (OAuth2 / JWT)",
             "Sécurisation pour déploiement multi-utilisateurs"],
            ["6 - Basse", "Monitoring CO2 (CodeCarbon reports)",
             "Écoresponsabilité documentée (RNCP C4.3)"],
        ],
        [25, 60, 81],
    )


def section_18_conclusion(pdf: EFREIPDF) -> None:
    n = _next_h1()
    h1(pdf, n, "Conclusion")

    p(pdf, (
        "Ce projet a transformé un dataset de capteurs industriels (24 042 observations, "
        "15 colonnes, Kaggle CC0) en un outil décisionnel complet pour la maintenance "
        "prédictive. En suivant une démarche structurée et reproductible - de l'analyse "
        "exploratoire jusqu'au déploiement - nous avons couvert l'intégralité du cycle "
        "de vie d'un projet de Data Science supervisé."
    ))

    p(pdf, (
        "Les chiffres clés du livrable : 12 modèles entraînés sur 3 tâches "
        "complémentaires, modèle final XGBoost binaire (F1=0.886, ROC-AUC=0.995), "
        "Random Forest pour la régression RUL (MAE=9.57 h, R2=0.673), seuil "
        "cost-sensitive à 0.23 générant une économie estimée à 12 000 EUR par cycle, "
        "dashboard Streamlit 5 onglets orienté métier, API FastAPI 4 endpoints, "
        "et 23 tests pytest couvrant les modules critiques."
    ))

    p(pdf, (
        "Sur le plan pédagogique, le projet couvre le Bloc 2 du RNCP40875 "
        "(Expert en Ingénierie de Données) en combinant : collecte et validation "
        "de données, preprocessing robuste (anti-leakage), modélisation supervisée "
        "multi-tâches, interprétabilité à 3 niveaux, évaluation comparative rigoureuse, "
        "et mise en production (API + dashboard). L'architecture modulaire du dépôt "
        "(13 modules src/, 10 scripts numérotés, 4 ADR, 5 fichiers de tests) "
        "garantit la maintenabilité et la défendabilité devant le jury."
    ))

    p(pdf, (
        "Les limites identifiées - données simulées, tuning Optuna documentaire, "
        "absence de monitoring de dérive - sont documentées et accompagnées de "
        "pistes d'amélioration concrètes pour une mise en production réelle. "
        "Ce projet constitue une base solide, extensible et transférable à un "
        "contexte industriel réel."
    ))


def section_19_annexes(pdf: EFREIPDF) -> None:
    n = _next_h1()
    h1(pdf, n, "Annexes")

    h2(pdf, f"{n}.1", "Lien GitHub")
    p(pdf, "Dépôt public : https://github.com/Adam-Blf/maintenance-predictive-industrielle")

    h2(pdf, f"{n}.2", "Requirements principaux")
    code_block(
        pdf,
        "numpy>=1.26\n"
        "pandas>=2.1\n"
        "scikit-learn>=1.4\n"
        "xgboost>=2.0\n"
        "joblib>=1.3\n"
        "optuna>=3.5\n"
        "shap>=0.44\n"
        "codecarbon>=2.3\n"
        "matplotlib>=3.8\n"
        "seaborn>=0.13\n"
        "plotly>=5.18\n"
        "streamlit>=1.32\n"
        "fastapi>=0.110\n"
        "pydantic>=2.6\n"
        "uvicorn[standard]>=0.27\n"
        "fpdf2>=2.7\n"
        "Pillow>=10.0\n"
        "python-pptx>=0.6.21\n"
        "pandera>=0.18\n"
        "pytest>=8.0\n"
        "pytest-cov>=4.1\n",
    )

    h2(pdf, f"{n}.3", "Suite pytest - 23 tests")
    make_table(
        pdf,
        ["Fichier", "Tests", "Périmètre"],
        [
            ["test_smoke.py", "5", "Imports, schéma, config, ColumnTransformer, MODEL_CATALOG"],
            ["test_preprocessing.py", "3", "fit_transform, imputation NaN, feature names"],
            ["test_models.py", "8", "Binaire (4), multiclasse (2), régression (1), KeyError (1)"],
            ["test_evaluation.py", "4", "Métriques parfaites, zero recall, to_dict"],
            ["test_api.py", "3", "/health, /predict valide, validation 422"],
        ],
        [45, 20, 101],
    )
    p(pdf, "Exécution : pytest tests/ -v --tb=short (durée ~7 secondes)")

    h2(pdf, f"{n}.4", "Architecture Decision Records")
    make_table(
        pdf,
        ["ADR", "Titre", "Décision principale"],
        [
            ["0001", "Stack technique",
             "Python 3.12, scikit-learn, XGBoost, MLP sklearn (pas PyTorch/TF)"],
            ["0002", "Source données",
             "Kaggle CC0 uniquement, aucun fallback synthétique"],
            ["0003", "Anti-data-leakage",
             "ColumnTransformer dans sklearn.Pipeline, fit exclusivement sur train"],
            ["0004", "Stratégie déploiement",
             "Streamlit local + FastAPI local, cloud en roadmap"],
        ],
        [15, 50, 101],
    )

    h2(pdf, f"{n}.5", "Glossaire ML")
    make_table(
        pdf,
        ["Terme", "Définition"],
        [
            ["F1-score",
             "Moyenne harmonique de la précision et du recall - adapté aux classes déséquilibrées"],
            ["ROC-AUC",
             "Aire sous la courbe ROC (TPR vs FPR) - robuste au déséquilibre"],
            ["PR-AUC",
             "Aire sous la courbe Précision-Recall - plus informative que ROC-AUC si "
             "la classe positive est rare"],
            ["Brier score",
             "Erreur quadratique moyenne entre probabilités prédites et labels réels - "
             "mesure la calibration"],
            ["SHAP",
             "SHapley Additive exPlanations - attribue à chaque feature sa contribution "
             "marginale à la prédiction"],
            ["scale_pos_weight",
             "Hyperparamètre XGBoost = n_négatifs / n_positifs - compense le déséquilibre "
             "de classes"],
            ["RUL",
             "Remaining Useful Life - durée de vie résiduelle estimée avant défaillance"],
            ["ColumnTransformer",
             "Objet scikit-learn appliquant des transformations différentes à des sous-ensembles "
             "de colonnes - clé de l'anti-leakage"],
        ],
        [40, 126],
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Construit le rapport PDF complet 19 sections avec fpdf2."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[06_build_report] Construction du rapport PDF (fpdf2)...")
    print(f"  Sortie : {OUTPUT_PDF}")

    pdf = EFREIPDF()

    # Page 1 : couverture
    section_1_cover(pdf)

    # Page 2 : table des matières
    section_toc(pdf)

    # Sections 2-19
    section_2_executive_summary(pdf)
    section_3_introduction(pdf)
    section_4_analyse_besoin(pdf)
    section_5_methodologie(pdf)
    section_6_referentiel_donnees(pdf)
    section_7_eda(pdf)
    section_8_preprocessing(pdf)
    section_9_pipeline(pdf)
    section_10_implementation(pdf)
    section_11_evaluation(pdf)
    section_12_interpretabilite(pdf)
    section_13_dashboard(pdf)
    section_14_api(pdf)
    section_15_resultats(pdf)
    section_16_gouvernance(pdf)
    section_17_limites(pdf)
    section_18_conclusion(pdf)
    section_19_annexes(pdf)

    pdf.output(str(OUTPUT_PDF))

    if OUTPUT_PDF.exists():
        size_kb = OUTPUT_PDF.stat().st_size / 1024
        nb_pages = pdf.page
        print(f"[06_build_report] PDF généré avec succès.")
        print(f"  Pages      : {nb_pages}")
        print(f"  Taille     : {size_kb:.0f} Ko")
        if size_kb < 800:
            print(f"  [WARN] Taille < 800 Ko - vérifier les figures manquantes.")
    else:
        print("[06_build_report] ERREUR : le PDF n'a pas été créé.")


if __name__ == "__main__":
    main()
