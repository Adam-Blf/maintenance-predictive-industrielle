# -*- coding: utf-8 -*-
"""Génération du support de présentation PPTX (livrable explicite).

Le sujet liste comme livrable obligatoire un *"Support de Présentation
du projet"*. Ce script produit `reports/presentation.pptx` avec ·

  - Slide de titre (logo EFREI + auteurs)
  - Contexte métier et objectifs
  - Architecture du système (image diagram_architecture)
  - Pipeline ML (image diagram_ml_pipeline)
  - Méthodologie · 4 modèles
  - Résultats · barplot comparatif
  - Interprétabilité (SHAP)
  - Industrialisation (Dashboard + API)
  - Conclusion
  - Slide de remerciement / Q&A

Charte EFREI · bleu institutionnel #0D47A1, fonts sobres.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.util import Inches, Pt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    EFREI_LOGO,
    MODELS_DIR,
    REPORTS_DIR,
    REPORTS_FIGURES_DIR,
    ensure_directories,
)

COLOR_NAVY = RGBColor(0x0D, 0x47, 0xA1)
COLOR_BLUE = RGBColor(0x1E, 0x88, 0xE5)
COLOR_DARK = RGBColor(0x21, 0x21, 0x21)
COLOR_GRAY = RGBColor(0x6E, 0x6E, 0x6E)
COLOR_WHITE = RGBColor(0xFF, 0xFF, 0xFF)


def _set_slide_background(slide, rgb: RGBColor) -> None:
    """Ajoute un rectangle plein écran comme fond coloré."""
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.33), Inches(7.5))
    bg.line.fill.background()
    bg.fill.solid()
    bg.fill.fore_color.rgb = rgb
    # Mettre le rectangle en arrière-plan.
    spTree = bg._element.getparent()
    spTree.remove(bg._element)
    spTree.insert(2, bg._element)


def _add_textbox(
    slide,
    text: str,
    left: float,
    top: float,
    width: float,
    height: float,
    *,
    size: int = 18,
    bold: bool = False,
    color: RGBColor = COLOR_DARK,
    align: str = "left",
) -> None:
    """Helper · ajoute un textbox avec formatage uniforme."""
    from pptx.enum.text import PP_ALIGN

    tx = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = tx.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = {"left": PP_ALIGN.LEFT, "center": PP_ALIGN.CENTER, "right": PP_ALIGN.RIGHT}[align]
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = "Calibri"


def _add_title_bar(slide, title: str) -> None:
    """Bandeau de titre uniforme en haut de chaque slide de contenu."""
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.33), Inches(0.9))
    bar.line.fill.background()
    bar.fill.solid()
    bar.fill.fore_color.rgb = COLOR_NAVY
    _add_textbox(
        slide,
        title,
        0.5,
        0.18,
        12.3,
        0.6,
        size=24,
        bold=True,
        color=COLOR_WHITE,
    )


def _add_image_safe(
    slide,
    path: Path,
    left: float,
    top: float,
    width: float,
    max_height: float | None = None,
    recenter: bool = True,
) -> float:
    """Insere une image en calculant la hauteur reelle via PIL.

    Si `max_height` est fourni et que l'image scaled depasserait, on
    reduit la largeur pour rester dans la zone disponible (evite que
    l'image ecrase un textbox positionne en dessous).

    Returns
    -------
    float
        Hauteur reelle de l'image inseree (en inches) · utilisee par les
        slides qui placent un textbox dynamiquement sous l'image.
    """
    if not path.exists():
        return 0.0
    from PIL import Image

    with Image.open(path) as img:
        px_w, px_h = img.size
    scaled_h = width * px_h / px_w  # in inches, ratio preserve

    # Si max_height est specifie et qu'on deborde, on reduit la largeur
    # proportionnellement pour rester dans la zone disponible.
    if max_height is not None and scaled_h > max_height:
        scaled_h = max_height
        width = scaled_h * px_w / px_h
        if recenter:
            left = (13.33 - width) / 2  # recentre horizontalement

    slide.shapes.add_picture(
        str(path), Inches(left), Inches(top), width=Inches(width), height=Inches(scaled_h)
    )
    return scaled_h


def build_title_slide(prs: Presentation) -> None:
    """Slide de couverture · logo + titre + auteurs."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    _set_slide_background(slide, COLOR_WHITE)

    # Logo EFREI centré.
    if EFREI_LOGO.exists():
        slide.shapes.add_picture(str(EFREI_LOGO), Inches(4.65), Inches(0.6), height=Inches(1.4))

    # Bandeau bleu avec titre.
    bar = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0.5), Inches(2.6), Inches(12.33), Inches(1.6)
    )
    bar.line.fill.background()
    bar.fill.solid()
    bar.fill.fore_color.rgb = COLOR_NAVY

    _add_textbox(
        slide,
        "Système Intelligent Multi-Modèles",
        0.5,
        2.7,
        12.33,
        0.7,
        size=32,
        bold=True,
        color=COLOR_WHITE,
        align="center",
    )
    _add_textbox(
        slide,
        "Maintenance Prédictive Industrielle",
        0.5,
        3.4,
        12.33,
        0.7,
        size=26,
        bold=True,
        color=COLOR_WHITE,
        align="center",
    )

    _add_textbox(
        slide,
        "Projet Data Science · M1 Mastère Data Engineering & IA",
        0.5,
        4.5,
        12.33,
        0.5,
        size=18,
        color=COLOR_NAVY,
        align="center",
    )
    _add_textbox(
        slide,
        "Bloc 2 · BC2 RNCP40875 · Année 2025-2026",
        0.5,
        5.0,
        12.33,
        0.5,
        size=14,
        color=COLOR_GRAY,
        align="center",
    )
    _add_textbox(
        slide,
        "Adam BELOUCIF  ·  Emilien MORICE",
        0.5,
        5.9,
        12.33,
        0.6,
        size=22,
        bold=True,
        color=COLOR_NAVY,
        align="center",
    )
    _add_textbox(
        slide,
        "EFREI Paris Panthéon-Assas Université  ·  www.efrei.fr",
        0.5,
        6.6,
        12.33,
        0.4,
        size=12,
        color=COLOR_GRAY,
        align="center",
    )


def build_context_slide(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _add_title_bar(slide, "1. Contexte métier et objectifs")
    bullets = [
        "Capteurs IoT industriels · vibration / température / pression / RPM en continu.",
        "Une panne non anticipée provoque arrêt non planifié, coût correctif élevé.",
        "Cible retenue : classification binaire failure_within_24h.",
        "Tâches bonus : multi-classe failure_type + régression rul_hours.",
        "Objectif : transformer le signal capteur en alerte exploitable.",
    ]
    _add_textbox(
        slide,
        "\n\n".join("•  " + b for b in bullets),
        0.7,
        1.3,
        12.0,
        5.8,
        size=18,
        color=COLOR_DARK,
    )


def build_architecture_slide(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _add_title_bar(slide, "2. Architecture du système")
    # Zone disponible · entre titre (y=0.9) et caption (~y=6.5) → max 5.4
    img_top = 1.2
    img_h = _add_image_safe(
        slide,
        REPORTS_FIGURES_DIR / "diagram_architecture.png",
        0.5,
        img_top,
        12.3,
        max_height=5.2,
    )
    _add_textbox(
        slide,
        "Architecture médaillon Bronze / Silver / Gold · pipeline reproductible · API + Dashboard.",
        0.5,
        img_top + img_h + 0.2,
        12.3,
        0.6,
        size=14,
        color=COLOR_GRAY,
        align="center",
    )


def build_methodology_slide(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _add_title_bar(slide, "3. Méthodologie · 4 modèles comparés")
    img_top = 1.2
    img_h = _add_image_safe(
        slide,
        REPORTS_FIGURES_DIR / "diagram_ml_pipeline.png",
        0.5,
        img_top,
        12.3,
        max_height=4.0,
    )
    bullets = [
        "Logistic Regression · baseline interprétable",
        "Random Forest · ensemble d'arbres, capture les non-linéarités",
        "XGBoost · gradient boosting, état de l'art tabulaire",
        "MLP 64-32-16 · Deep Learning (early stopping, alpha = 1e-3)",
    ]
    # Bullets · 1 par ligne pour eviter la superposition horizontale.
    _add_textbox(
        slide,
        "\n".join("•  " + b for b in bullets),
        0.8,
        img_top + img_h + 0.3,
        11.7,
        1.8,
        size=15,
        color=COLOR_DARK,
        align="left",
    )


def build_results_slide(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _add_title_bar(slide, "4. Résultats · comparaison des 4 modèles")
    img_top = 1.1
    img_h = _add_image_safe(
        slide,
        REPORTS_FIGURES_DIR / "metrics_comparison_barplot.png",
        0.5,
        img_top,
        12.3,
        max_height=5.0,
    )

    # Lecture du modèle final.
    final_name_path = MODELS_DIR / "final_model_name.txt"
    final_name = (
        final_name_path.read_text(encoding="utf-8").strip()
        if final_name_path.exists()
        else "modèle final"
    )
    _add_textbox(
        slide,
        f"Modèle candidat retenu : {final_name} (compromis F1 / stabilité CV / interprétabilité)",
        0.5,
        img_top + img_h + 0.25,
        12.3,
        0.6,
        size=16,
        bold=True,
        color=COLOR_NAVY,
        align="center",
    )


def build_curves_slide(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _add_title_bar(slide, "5. ROC + Precision-Recall")
    # 2 images côte à côte · contrainte hauteur pour eviter overflow.
    _add_image_safe(
        slide,
        REPORTS_FIGURES_DIR / "roc_curves_comparison.png",
        0.3,
        1.1,
        6.4,
        max_height=5.5,
        recenter=False,
    )
    _add_image_safe(
        slide,
        REPORTS_FIGURES_DIR / "pr_curves_comparison.png",
        6.7,
        1.1,
        6.4,
        max_height=5.5,
        recenter=False,
    )


def build_interpret_slide(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _add_title_bar(slide, "6. Interprétabilité · SHAP")
    final_name = (MODELS_DIR / "final_model_name.txt").read_text(encoding="utf-8").strip()
    # 2 images côte à côte · recenter=False pour ne pas se chevaucher.
    h1 = _add_image_safe(
        slide,
        REPORTS_FIGURES_DIR / f"shap_summary_{final_name}.png",
        0.3,
        1.1,
        7.0,
        max_height=5.4,
        recenter=False,
    )
    h2 = _add_image_safe(
        slide,
        REPORTS_FIGURES_DIR / f"shap_bar_{final_name}.png",
        7.5,
        1.4,
        5.5,
        max_height=4.8,
        recenter=False,
    )
    bottom = 1.1 + max(h1, h2) + 0.2
    _add_textbox(
        slide,
        "Vibration_rms · Temperature_motor · Maintenance_age_days = top 3 variables explicatives.",
        0.5,
        bottom,
        12.3,
        0.5,
        size=14,
        color=COLOR_GRAY,
        align="center",
    )


def build_industrialization_slide(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _add_title_bar(slide, "7. Industrialisation · Dashboard + API")

    bullets = [
        "Dashboard Streamlit · 5 onglets, CSS personnalisé charte EFREI",
        "API REST FastAPI · /predict, /health, /model-info (Pydantic v2)",
        "Architecture front → API → modèle (joblib), pratiques production",
        "Container Docker + docker-compose, démarrage en docker compose up",
    ]
    _add_textbox(
        slide,
        "\n\n".join("•  " + b for b in bullets),
        0.7,
        1.3,
        12.0,
        5.8,
        size=18,
        color=COLOR_DARK,
    )


def build_eco_slide(prs: Presentation) -> None:
    """Slide écoresponsabilité (CodeCarbon)."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _add_title_bar(slide, "8. Écoresponsabilité (RNCP C4.3)")
    img_top = 1.1
    img_h = _add_image_safe(
        slide,
        REPORTS_FIGURES_DIR / "compute_cost_comparison.png",
        0.5,
        img_top,
        12.3,
        max_height=5.0,
    )
    _add_textbox(
        slide,
        "Empreinte CO2 mesurée via CodeCarbon · mix énergétique France ~80 gCO2/kWh.",
        0.5,
        img_top + img_h + 0.25,
        12.3,
        0.5,
        size=14,
        color=COLOR_GRAY,
        align="center",
    )


def build_conclusion_slide(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _add_title_bar(slide, "9. Conclusion et perspectives")
    bullets = [
        "MVP complet · pipeline + 4 modèles + dashboard + API + rapport",
        "Tâches bonus livrées · multi-classe failure_type + régression RUL",
        "Hyperparameter tuning Optuna + threshold métier optimisé (FN/FP)",
        "Limites : drift temporel non géré, événements rares sous-représentés",
        "Suite : monitoring drift, RNN / LSTM temporel, intégration GMAO",
    ]
    _add_textbox(
        slide,
        "\n\n".join("•  " + b for b in bullets),
        0.7,
        1.3,
        12.0,
        5.8,
        size=18,
        color=COLOR_DARK,
    )


def build_thanks_slide(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _set_slide_background(slide, COLOR_NAVY)
    if EFREI_LOGO.exists():
        slide.shapes.add_picture(str(EFREI_LOGO), Inches(4.65), Inches(0.7), height=Inches(1.2))
    _add_textbox(
        slide,
        "Merci pour votre attention",
        0.5,
        2.8,
        12.33,
        1.0,
        size=44,
        bold=True,
        color=COLOR_WHITE,
        align="center",
    )
    _add_textbox(
        slide,
        "Questions & démonstration live",
        0.5,
        4.0,
        12.33,
        0.6,
        size=22,
        color=COLOR_WHITE,
        align="center",
    )
    _add_textbox(
        slide,
        "Adam BELOUCIF · Emilien MORICE  ·  EFREI Paris Panthéon-Assas Université",
        0.5,
        6.5,
        12.33,
        0.4,
        size=14,
        color=COLOR_WHITE,
        align="center",
    )


def main() -> None:
    ensure_directories()

    prs = Presentation()
    prs.slide_width = Inches(13.33)
    prs.slide_height = Inches(7.5)

    build_title_slide(prs)
    build_context_slide(prs)
    build_architecture_slide(prs)
    build_methodology_slide(prs)
    build_results_slide(prs)
    build_curves_slide(prs)
    build_interpret_slide(prs)
    build_industrialization_slide(prs)
    build_eco_slide(prs)
    build_conclusion_slide(prs)
    build_thanks_slide(prs)

    output = REPORTS_DIR / "presentation.pptx"
    prs.save(str(output))
    size_kb = output.stat().st_size / 1024
    print(f"[SLIDES] Generated · {output} ({size_kb:.1f} Ko, 11 slides)")


if __name__ == "__main__":
    main()
