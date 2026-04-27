# -*- coding: utf-8 -*-
"""Calibration probabiliste + tuning du seuil de decision.

Sur un probleme de maintenance predictive desequilibre, le seuil de
decision par defaut (0.5) n'est presque jamais le bon · il faut le
choisir en fonction du COUT relatif des faux negatifs (panne ratee)
versus faux positifs (intervention inutile).

On produit ·
  - **Reliability diagram** + Brier score · qualite de la calibration.
  - **Courbe cout/recall** · seuil optimal pour un ratio coût FN/FP donné.
  - **Threshold optimal** · sauvegarde dans `models/optimal_threshold.json`
    pour utilisation par l'API et le dashboard.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from .config import (
    COLOR_ALERT_RED,
    COLOR_EFREI_BLUE,
    COLOR_EFREI_DARK,
    COLOR_OK_GREEN,
    REPORTS_FIGURES_DIR,
)


def reliability_diagram(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str,
    output_dir: Path = REPORTS_FIGURES_DIR,
) -> tuple[Path, float]:
    """Trace le reliability diagram + retourne le Brier score.

    Un modele parfaitement calibre suit la diagonale · la frequence
    observee dans chaque bucket de probabilite egale la probabilite
    moyenne predite.
    """
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10, strategy="quantile")
    brier = float(brier_score_loss(y_true, y_proba))

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Calibration parfaite")
    ax.plot(
        prob_pred,
        prob_true,
        "o-",
        color=COLOR_EFREI_BLUE,
        markersize=8,
        linewidth=2,
        label=f"{model_name} (Brier = {brier:.4f})",
    )
    ax.set_xlabel("Probabilité moyenne prédite", fontsize=11)
    ax.set_ylabel("Fréquence observée de la classe positive", fontsize=11)
    ax.set_title(
        f"Reliability diagram · {model_name}",
        fontsize=13,
        fontweight="bold",
        color=COLOR_EFREI_DARK,
    )
    ax.legend(framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()

    output_path = output_dir / f"reliability_diagram_{model_name}.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path, brier


def cost_recall_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str,
    cost_fn: float = 1000.0,
    cost_fp: float = 100.0,
    output_dir: Path = REPORTS_FIGURES_DIR,
) -> tuple[Path, float, dict]:
    """Trace le coût total en fonction du seuil + retourne le seuil optimal.

    Hypotheses metier ·
      - Faux negatif (panne ratee) · arret de production = 1000 EUR
      - Faux positif (intervention inutile) · main d'oeuvre = 100 EUR
    Ces valeurs sont indicatives · le ratio 10:1 reflete la realite du
    secteur (un arret non planifie coute 5 a 20x plus qu'une intervention
    preventive).

    Le seuil optimal est celui qui minimise le cout total sur le test set.
    """
    thresholds = np.linspace(0.05, 0.95, 91)
    costs = []
    fns_arr = []
    fps_arr = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        costs.append(cost_fn * fn + cost_fp * fp)
        fns_arr.append(fn)
        fps_arr.append(fp)

    optimal_idx = int(np.argmin(costs))
    optimal_threshold = float(thresholds[optimal_idx])
    optimal_cost = float(costs[optimal_idx])

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(thresholds, costs, "-", color=COLOR_EFREI_BLUE, linewidth=2, label="Coût total")
    ax.axvline(
        optimal_threshold,
        color=COLOR_ALERT_RED,
        linestyle="--",
        linewidth=2,
        label=f"Seuil optimal = {optimal_threshold:.2f}",
    )
    ax.axvline(0.5, color="gray", linestyle=":", linewidth=1, label="Seuil par défaut 0.50")
    ax.set_xlabel("Seuil de décision", fontsize=11)
    ax.set_ylabel(f"Coût total (FN={cost_fn:.0f}€, FP={cost_fp:.0f}€)", fontsize=11)
    ax.set_title(
        f"Optimisation du seuil métier · {model_name}",
        fontsize=13,
        fontweight="bold",
        color=COLOR_EFREI_DARK,
    )
    ax.legend(framealpha=0.95)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = output_dir / f"cost_threshold_{model_name}.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    info = {
        "optimal_threshold": optimal_threshold,
        "optimal_cost_eur": optimal_cost,
        "cost_fn_eur": cost_fn,
        "cost_fp_eur": cost_fp,
        "fn_at_default_0_5": int(fns_arr[np.argmin(np.abs(thresholds - 0.5))]),
        "fp_at_default_0_5": int(fps_arr[np.argmin(np.abs(thresholds - 0.5))]),
        "fn_at_optimal": int(fns_arr[optimal_idx]),
        "fp_at_optimal": int(fps_arr[optimal_idx]),
    }
    return output_path, optimal_threshold, info


def save_threshold(threshold: float, info: dict, output_path: Path) -> None:
    """Persiste le seuil optimal pour consommation par API/Dashboard."""
    payload = {"threshold": threshold, **info}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
