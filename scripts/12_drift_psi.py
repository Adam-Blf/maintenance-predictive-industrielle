# -*- coding: utf-8 -*-
"""Script 12 · Détection de dérive (data drift) par PSI (BONUS).

À QUOI ÇA SERT ?
----------------
Un modèle de maintenance se dégrade silencieusement si la distribution
des capteurs en production s'éloigne de celle d'entraînement (usure des
sondes, changement de régime machine, saison). Le PSI (Population
Stability Index) quantifie cet écart, feature par feature ·

    PSI = sum_bins (p_prod - p_ref) * ln(p_prod / p_ref)

Seuils d'interprétation usuels (industrie crédit/risque) ·
  - PSI < 0.10  · stable, pas de dérive significative
  - 0.10-0.25   · dérive modérée à surveiller
  - PSI > 0.25  · dérive forte, ré-entraînement recommandé

Ce script compare la distribution d'ENTRAÎNEMENT (référence) à celle du
TEST (proxy production). Il NE MODIFIE PAS les données · lecture seule.

USAGE
-----
    python scripts/12_drift_psi.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import DATA_PROCESSED_DIR, NUMERIC_FEATURES, REPORTS_DIR

OUT = REPORTS_DIR / "12"
OUT.mkdir(parents=True, exist_ok=True)


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """PSI entre une distribution de référence et une distribution courante."""
    # Bornes de bins fondées sur les quantiles de la référence (robuste).
    quantiles = np.linspace(0, 100, bins + 1)
    edges = np.percentile(reference, quantiles)
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)
    # Proportions lissées (évite ln(0) sur un bin vide).
    eps = 1e-6
    ref_p = np.clip(ref_counts / ref_counts.sum(), eps, None)
    cur_p = np.clip(cur_counts / cur_counts.sum(), eps, None)
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))


def main() -> None:
    X_ref = pd.read_csv(DATA_PROCESSED_DIR / "X_train.csv")
    X_cur = pd.read_csv(DATA_PROCESSED_DIR / "X_test.csv")
    feats = [f for f in NUMERIC_FEATURES if f in X_ref.columns]

    rows = []
    for f in feats:
        val = psi(X_ref[f].to_numpy(), X_cur[f].to_numpy())
        verdict = "stable" if val < 0.10 else ("modérée" if val < 0.25 else "forte")
        rows.append((f, val, verdict))
    rows.sort(key=lambda r: r[1], reverse=True)

    # Graphe PSI par feature.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    colors = ["#FF43B8" if v >= 0.25 else "#0C78B4" if v >= 0.10 else "#163767" for v in vals]
    ax.barh(names, vals, color=colors)
    ax.axvline(0.10, color="#5A6B82", ls="--", lw=1)
    ax.axvline(0.25, color="#FF43B8", ls="--", lw=1)
    ax.set_xlabel("PSI (train → test)")
    ax.set_title("Dérive de distribution par variable · seuils 0.10 / 0.25")
    fig.tight_layout()
    fig.savefig(OUT / "psi_barplot.png", dpi=130)
    plt.close(fig)

    # Rapport markdown.
    n_drift = sum(1 for _, v, _ in rows if v >= 0.10)
    lines = [
        "# Détection de dérive · PSI (train → test)",
        "",
        "Référence · jeu d'entraînement. Courant · jeu de test (proxy production).",
        "Lecture seule sur les données. Seuils · < 0.10 stable, 0.10-0.25 modérée, > 0.25 forte.",
        "",
        "| Variable | PSI | Verdict |",
        "|---|---|---|",
    ]
    for name, val, verdict in rows:
        lines.append(f"| {name} | {val:.4f} | {verdict} |")
    lines += [
        "",
        f"**{n_drift} variable(s)** avec PSI >= 0.10. "
        + ("Aucune dérive significative · le modèle reste valide." if n_drift == 0
           else "À surveiller · planifier un ré-entraînement si la dérive persiste."),
        "",
        "![PSI](psi_barplot.png)",
    ]
    (OUT / "drift_report.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"[DRIFT] {len(feats)} variables analysées · {n_drift} en dérive (PSI>=0.10)")
    print(f"  -> {OUT / 'drift_report.md'}")
    print(f"  -> {OUT / 'psi_barplot.png'}")


if __name__ == "__main__":
    main()
