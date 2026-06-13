# -*- coding: utf-8 -*-
"""Script 13 · Prédiction conforme · ensembles avec garantie de couverture (BONUS).

À QUOI ÇA SERT ?
----------------
Au lieu d'une décision binaire (panne / sain), on produit un ENSEMBLE de
prédiction avec une garantie statistique · à 90% de confiance, la vraie
classe est dans l'ensemble au moins 90% du temps, sans hypothèse de
distribution. Pour la maintenance ·
  - {1}     · panne quasi certaine → intervention prioritaire
  - {0}     · machine saine, garantie
  - {0, 1}  · zone d'incertitude → inspection humaine
  - {}      · hors domaine de confiance

On découpe le jeu de TEST en deux moitiés (calibration / évaluation),
sans jamais modifier les fichiers de données (lecture seule).

USAGE
-----
    python scripts/13_conformal.py
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from src.conformal import ConformalBinaryClassifier
from src.config import DATA_PROCESSED_DIR, MODELS_DIR, RANDOM_STATE, REPORTS_DIR

OUT = REPORTS_DIR / "13"
OUT.mkdir(parents=True, exist_ok=True)
ALPHAS = [0.20, 0.10, 0.05]  # couvertures cibles 80 / 90 / 95 %


def main() -> None:
    model = joblib.load(MODELS_DIR / "final_model.joblib")
    X = pd.read_csv(DATA_PROCESSED_DIR / "X_test.csv")
    y = pd.read_csv(DATA_PROCESSED_DIR / "y_test.csv").iloc[:, 0].to_numpy()

    # Découpe calibration / évaluation (en mémoire · données intactes).
    X_cal, X_eval, y_cal, y_eval = train_test_split(
        X, y, test_size=0.5, random_state=RANDOM_STATE, stratify=y
    )
    p_cal = model.predict_proba(X_cal)[:, 1]
    p_eval = model.predict_proba(X_eval)[:, 1]

    rows = []
    set_size_hist = {}
    for alpha in ALPHAS:
        cp = ConformalBinaryClassifier(alpha=alpha).calibrate(p_cal, y_cal)
        sets = cp.predict_sets(p_eval)
        cov = ConformalBinaryClassifier.coverage(sets, y_eval)
        size = ConformalBinaryClassifier.average_set_size(sets)
        dist = Counter(frozenset(s) for s in sets)
        rows.append((alpha, cov, size, cp.qhat, dist))
        set_size_hist[alpha] = [len(s) for s in sets]

    # Graphe · couverture visée vs empirique.
    fig, ax = plt.subplots(figsize=(7, 4.2))
    target = [1 - a for a in ALPHAS]
    empiric = [r[1] for r in rows]
    ax.plot(target, target, "--", color="#5A6B82", label="couverture idéale")
    ax.plot(target, empiric, "o-", color="#163767", lw=2, label="couverture empirique")
    for t, e in zip(target, empiric):
        ax.annotate(f"{e:.1%}", (t, e), textcoords="offset points", xytext=(6, 6), fontsize=9)
    ax.set_xlabel("Couverture cible (1 - alpha)")
    ax.set_ylabel("Couverture empirique")
    ax.set_title("Prédiction conforme · garantie de couverture vérifiée")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "conformal_coverage.png", dpi=130)
    plt.close(fig)

    def label(fs: frozenset) -> str:
        return {frozenset(): "{} vide", frozenset({0}): "{sain}",
                frozenset({1}): "{panne}", frozenset({0, 1}): "{0,1} incertain"}[fs]

    lines = ["# Prédiction conforme (split-conformal LAC)", "",
             "Garantie · à 1-alpha, la vraie classe est dans l'ensemble >= 1-alpha du temps.",
             "Découpe calibration/évaluation sur le test set (données non modifiées).", "",
             "| Cible (1-alpha) | Couverture empirique | Taille moy. ensemble | Seuil q |",
             "|---|---|---|---|"]
    for alpha, cov, size, q, _ in rows:
        lines.append(f"| {1 - alpha:.0%} | {cov:.1%} | {size:.2f} | {q:.3f} |")
    lines += ["", "## Répartition des ensembles (à 90% de couverture)", ""]
    for fs, n in sorted(rows[1][4].items(), key=lambda kv: -kv[1]):
        lines.append(f"- {label(fs)} · {n} cas ({n / len(set_size_hist[0.10]):.1%})")
    lines += ["", "![Couverture](conformal_coverage.png)"]
    (OUT / "conformal_report.md").write_text("\n".join(lines), encoding="utf-8")

    print("[CONFORMAL] couverture empirique vs cible ·")
    for alpha, cov, size, q, _ in rows:
        print(f"  cible {1 - alpha:.0%} -> empirique {cov:.1%} -- taille moy {size:.2f}")
    print(f"  -> {OUT / 'conformal_report.md'}")


if __name__ == "__main__":
    main()
