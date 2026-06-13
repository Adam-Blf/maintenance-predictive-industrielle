# -*- coding: utf-8 -*-
"""Script 15 · Robustesse au bruit capteur et aux pannes de sonde (BONUS).

À QUOI ÇA SERT ?
----------------
En conditions réelles, les capteurs sont bruités et tombent parfois en
panne. Un modèle entraîné sur des données « propres » peut s'effondrer.
Ce script stresse le modèle final ·

  1. **Bruit gaussien** · on ajoute un bruit d'écart-type sigma × std de
     chaque variable (sigma de 0 à 0.5) et on mesure la chute de F1/ROC-AUC.
  2. **Panne de sonde** · on remplace une fraction des valeurs d'une
     variable par sa médiane (sonde figée) et on mesure l'impact.

Les perturbations sont appliquées sur des COPIES EN MÉMOIRE · les
fichiers de données ne sont jamais modifiés (lecture seule).

USAGE
-----
    python scripts/15_noise_robustness.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_auc_score

from src.config import DATA_PROCESSED_DIR, MODELS_DIR, NUMERIC_FEATURES, RANDOM_STATE, REPORTS_DIR

OUT = REPORTS_DIR / "15"
OUT.mkdir(parents=True, exist_ok=True)
SIGMAS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]
DROPOUT_FRACS = [0.0, 0.05, 0.10, 0.20, 0.40]


def evaluate(model, X: pd.DataFrame, y: np.ndarray) -> tuple[float, float]:
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return f1_score(y, pred), roc_auc_score(y, proba)


def main() -> None:
    rng = np.random.default_rng(RANDOM_STATE)
    model = joblib.load(MODELS_DIR / "final_model.joblib")
    X = pd.read_csv(DATA_PROCESSED_DIR / "X_test.csv")
    y = pd.read_csv(DATA_PROCESSED_DIR / "y_test.csv").iloc[:, 0].to_numpy()
    feats = [f for f in NUMERIC_FEATURES if f in X.columns]
    stds = X[feats].std()

    # 1) Bruit gaussien croissant.
    noise_rows = []
    for sigma in SIGMAS:
        Xn = X.copy()
        if sigma > 0:
            for f in feats:
                Xn[f] = Xn[f] + rng.normal(0.0, sigma * stds[f], size=len(Xn))
        f1, auc = evaluate(model, Xn, y)
        noise_rows.append((sigma, f1, auc))

    # 2) Panne de sonde (valeurs figées à la médiane).
    medians = X[feats].median()
    drop_rows = []
    for frac in DROPOUT_FRACS:
        Xd = X.copy()
        if frac > 0:
            for f in feats:
                mask = rng.random(len(Xd)) < frac
                Xd.loc[mask, f] = medians[f]
        f1, auc = evaluate(model, Xd, y)
        drop_rows.append((frac, f1, auc))

    # Graphe combiné.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot([r[0] for r in noise_rows], [r[1] for r in noise_rows], "o-", color="#163767", label="F1")
    ax1.plot([r[0] for r in noise_rows], [r[2] for r in noise_rows], "s-", color="#FF43B8", label="ROC-AUC")
    ax1.set_xlabel("Bruit gaussien (× écart-type)")
    ax1.set_ylabel("Score")
    ax1.set_title("Robustesse au bruit capteur")
    ax1.legend()
    ax2.plot([r[0] for r in drop_rows], [r[1] for r in drop_rows], "o-", color="#163767", label="F1")
    ax2.plot([r[0] for r in drop_rows], [r[2] for r in drop_rows], "s-", color="#0C78B4", label="ROC-AUC")
    ax2.set_xlabel("Fraction de sondes figées")
    ax2.set_title("Robustesse aux pannes de sonde")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(OUT / "robustness_curves.png", dpi=130)
    plt.close(fig)

    base_f1 = noise_rows[0][1]
    lines = ["# Robustesse · bruit capteur & pannes de sonde", "",
             "Perturbations appliquées en mémoire · données non modifiées.", "",
             "## Bruit gaussien", "", "| Sigma (× std) | F1 | ROC-AUC | Chute F1 |", "|---|---|---|---|"]
    for s, f1, auc in noise_rows:
        lines.append(f"| {s:.2f} | {f1:.3f} | {auc:.3f} | {(base_f1 - f1):+.3f} |")
    lines += ["", "## Panne de sonde (valeurs figées)", "",
              "| Fraction figée | F1 | ROC-AUC |", "|---|---|---|"]
    for fr, f1, auc in drop_rows:
        lines.append(f"| {fr:.0%} | {f1:.3f} | {auc:.3f} |")
    lines += ["", "![Robustesse](robustness_curves.png)"]
    (OUT / "robustness_report.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"[ROBUST] F1 propre={base_f1:.3f} · F1 @sigma=0.30 ={noise_rows[4][1]:.3f}"
          f" · F1 @40% sondes figées={drop_rows[-1][1]:.3f}")
    print(f"  -> {OUT / 'robustness_report.md'}")


if __name__ == "__main__":
    main()
