# -*- coding: utf-8 -*-
"""Script 16 - Gestion du desequilibre de classes (BONUS avance).

A QUOI CA SERT ?
----------------
Ce script traite un probleme fondamental de la maintenance predictive :
les pannes sont RARES. Sur ce dataset, ~10% des observations correspondent
a une panne imminente (classe 1), contre ~90% de fonctionnement normal
(classe 0). Ce desequilibre est inherent aux machines bien entretenues -
une machine qui tombe souvent en panne n'est pas en production longtemps.

POURQUOI C'EST UN PROBLEME ?
-----------------------------
Si on ignore le desequilibre, un modele peut atteindre 90% d'accuracy en
predisant TOUJOURS "pas de panne". Sa precision est nulle sur les pannes.
C'est la "paradoxe de l'accuracy" : fort sur les chiffres, inutile en
production.

L'indicateur pertinent ici est le RECALL (aussi appele sensibilite) :
parmi toutes les pannes reelles, quelle fraction avons-nous detecte ?
Un Recall de 0.95 signifie que 95% des pannes ont declenche une alerte.
En industrie, une panne non detectee coute 10 a 100x plus cher qu'une
fausse alerte (arret de production, securite, reprise).

LES 5 STRATEGIES COMPAREES
---------------------------
Toutes les strategies utilisent le meme Random Forest (RF) de base pour
isoler l'effet du reequilibrage uniquement (pas de tuning cache).

  1. BASELINE (class_weight="balanced")
     Reference deja utilisee dans le script 03. Sklearn pondere la perte
     de chaque exemple inversement a sa frequence. Simple, sans retouche
     des donnees. C'est la strategie "gratuite" mais souvent insuffisante
     sur des desequilibres extremes (IR > 20).

  2. SMOTE - Synthetic Minority Oversampling TEchnique
     Chawla et al. (2002). Generation d'exemples SYNTHETIQUES de la classe
     minoritaire par interpolation entre k voisins proches (k=5 ici).
     Avantages : evite la copie exacte, augmente la diversite. Fontionne
     dans l'espace transforme (apres StandardScaler).
     Risque : peut generer des exemples "irrealistes" si les classes se
     chevauchent beaucoup en haute dimension.

  3. ADASYN - ADaptive SYNthetic Sampling
     He et al. (2008). Variante de SMOTE qui concentre la generation sur
     les zones DIFFICILES (exemples minoritaires entoures de majoritaires).
     Plus d'exemples synthetiques la ou le modele risque de se tromper.
     Souvent superieur a SMOTE quand la frontiere de decision est complexe.

  4. RandomUnderSampler
     Suppression ALEATOIRE d'exemples majoritaires jusqu'a equilibre.
     Simple, rapide, mais on PERD beaucoup d'information : 19k des 21k
     negatifs sont supprimes (train set reduit de 24k a ~4.8k).
     Utile si le dataset est tres grand et si le temps d'entrainement
     est une contrainte dure.

  5. SMOTE + Tomek Links (SMOTETomek) - hybride
     Phase 1 (SMOTE) : sureechantillonne la minorite.
     Phase 2 (Tomek Links) : nettoie les paires ambigues a la frontiere
     (un exemple majoritaire et un minoritaire proches sont supprimes).
     Resultat : dataset equilibre ET mieux separe en bords de decision.
     Compromis entre SMOTE pur et nettoyage du bruit.

ARCHITECTURE PIPELINE
---------------------
On utilise imblearn.pipeline.Pipeline (pas sklearn.pipeline) qui autorise
l'insertion d'un resampler entre le preprocesseur et le classifieur :

    imblearn.Pipeline([
        ("preprocessor", ColumnTransformer()),   # StandardScaler + OHE
        ("resampler",    SMOTE()),                # SEULEMENT pendant fit
        ("classifier",   RandomForestClassifier()),
    ])

GARANTIE CRITIQUE : le resampler ne s'active QUE pendant pipeline.fit().
Lors de pipeline.predict_proba(X_test), imblearn saute le resampler.
Le test set n'est donc JAMAIS contamine par le resampling.

CE QUI EST ENREGISTRE
---------------------
  reports/16/imbalance_analysis.png    : distribution avant/apres
  reports/16/pr_curves_all.png         : courbes PR par strategie
  reports/16/metrics_comparison.png    : barplot F1/Recall/Precision/PR-AUC
  reports/16/fit_time_comparison.png   : cout computationnel
  reports/16/summary.csv               : tableau recapitulatif
  reports/16/summary.json              : meme tableau en JSON machine-readable
  reports/16/threshold_optimization.png: seuil optimal sur la meilleure strat.
  reports/16/optimal_threshold_16.json : seuil + JSON de la meilleure strat.

USAGE
-----
    python scripts/16_gestion_desequilibre.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# On ajoute la racine du projet au PYTHONPATH avant tout import interne.
# Portable : fonctionne que ce soit lance depuis la racine, depuis
# scripts/, ou depuis n'importe quel dossier.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Bootstrap : installe les dependances manquantes automatiquement.
# imbalanced-learn n'est pas toujours dans l'environnement de base.
from src.bootstrap import ensure_dependencies  # noqa: E402
ensure_dependencies()

import pandas as pd  # noqa: E402

from src.config import (  # noqa: E402
    DATA_PROCESSED_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    TARGET_BINARY,
    ensure_directories,
)
from src.imbalance import (  # noqa: E402
    STRATEGY_META,
    analyze_imbalance,
    build_strategy_pipeline,
    compare_all_strategies,
    optimize_threshold,
    plot_class_distribution,
    plot_fit_time_comparison,
    plot_metrics_comparison,
    plot_pr_comparison,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUT = REPORTS_DIR / "16"


# ---------------------------------------------------------------------------
# Helpers console
# ---------------------------------------------------------------------------

def _banner(title: str, char: str = "=") -> None:
    """Separateur visuel dans les logs console."""
    line = char * 70
    print(f"\n{line}\n{title}\n{line}")


def _section(title: str) -> None:
    _banner(title, char="-")


# ---------------------------------------------------------------------------
# Chargement des donnees
# ---------------------------------------------------------------------------

def load_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Charge les splits train/test depuis DATA_PROCESSED_DIR.

    Les CSV X_train.csv, y_train.csv, X_test.csv, y_test.csv sont ecrits
    par les scripts precedents (03, 07, 08, ...). On les lit ici comme
    donnees BRUTES (non preprocessees) - le preprocessing est integre
    dans chaque imblearn.Pipeline.

    Returns
    -------
    X_train, y_train, X_test, y_test
    """
    X_train = pd.read_csv(DATA_PROCESSED_DIR / "X_train.csv")
    y_train = pd.read_csv(DATA_PROCESSED_DIR / "y_train.csv").squeeze()
    X_test = pd.read_csv(DATA_PROCESSED_DIR / "X_test.csv")
    y_test = pd.read_csv(DATA_PROCESSED_DIR / "y_test.csv").squeeze()

    # S'assurer que la cible est bien numerique entiere.
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    print(f"Train : {X_train.shape[0]:,} lignes | Test : {X_test.shape[0]:,} lignes")
    print(f"Features : {X_train.shape[1]} colonnes")
    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Orchestre la comparaison des strategies de gestion du desequilibre."""
    ensure_directories()
    OUT.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # ETAPE 1 : Chargement
    # -----------------------------------------------------------------------
    _banner("ETAPE 1 - Chargement du dataset")
    X_train, y_train, X_test, y_test = load_data()

    # -----------------------------------------------------------------------
    # ETAPE 2 : Analyse du desequilibre initial
    # -----------------------------------------------------------------------
    _section("ETAPE 2 - Analyse du desequilibre")
    stats = analyze_imbalance(y_train, output_dir=OUT)

    print(f"\nClasse majoritaire (0 = sain)   : {stats['count_majority']:,} obs")
    print(f"Classe minoritaire (1 = panne)  : {stats['count_minority']:,} obs")
    print(f"Imbalance Ratio (IR)            : {stats['ratio']:.1f}:1")
    print(f"Accuracy d'un modele naif       : {stats['accuracy_naive'] * 100:.1f}%")
    print(
        "\n=> Un modele qui predit toujours 'pas de panne' a une accuracy de"
        f" {stats['accuracy_naive'] * 100:.1f}% mais detecte 0 panne (Recall=0)."
    )
    print(
        "   On compare F1 et PR-AUC - metriques robustes au desequilibre.\n"
    )

    plot_class_distribution(y_train, OUT / "class_distribution.png")

    # -----------------------------------------------------------------------
    # ETAPE 3 : Comparaison des 5 strategies
    # -----------------------------------------------------------------------
    _section("ETAPE 3 - Comparaison des 5 strategies de reequilibrage")
    print("Entrainement des 5 pipelines sur le train set ...")
    print(f"  IR = {stats['ratio']:.1f}:1 -> chaque strategie doit l'adresser differemment\n")

    results_df = compare_all_strategies(
        X_train, y_train,
        X_test,  y_test,
        verbose=True,
    )

    # -----------------------------------------------------------------------
    # ETAPE 4 : Tableau de synthese
    # -----------------------------------------------------------------------
    _section("ETAPE 4 - Tableau recapitulatif")
    display_cols = ["label", "f1", "recall", "precision", "pr_auc", "roc_auc", "fit_time_s"]
    print(results_df[display_cols].to_string(index=False, float_format="{:.4f}".format))

    # Sauvegarde CSV + JSON
    results_df.to_csv(OUT / "summary.csv", index=False)
    results_df.to_json(OUT / "summary.json", orient="records", indent=2)
    print(f"\n  Tableau sauvegarde -> {OUT / 'summary.csv'}")

    # -----------------------------------------------------------------------
    # ETAPE 5 : Identification de la meilleure strategie
    # -----------------------------------------------------------------------
    _section("ETAPE 5 - Meilleure strategie")
    best = results_df.iloc[0]
    print(f"Strategie gagnante (F1 max) : {best['label']}")
    print(f"  F1      = {best['f1']:.4f}")
    print(f"  Recall  = {best['recall']:.4f}")
    print(f"  PR-AUC  = {best['pr_auc']:.4f}")

    # Comparaison avec baseline (toujours premier dans STRATEGY_META)
    baseline_row = results_df[results_df["strategy"] == "baseline"]
    if not baseline_row.empty:
        baseline_f1 = float(baseline_row["f1"].values[0])
        baseline_rec = float(baseline_row["recall"].values[0])
        delta_f1 = best["f1"] - baseline_f1
        delta_rec = best["recall"] - baseline_rec
        print(f"\n  vs baseline (class_weight seul) :")
        print(f"    delta F1     = {delta_f1:+.4f}")
        print(f"    delta Recall = {delta_rec:+.4f}")

    # -----------------------------------------------------------------------
    # ETAPE 6 : Visualisations comparatives
    # -----------------------------------------------------------------------
    _section("ETAPE 6 - Graphiques comparatifs")

    print("Generation des courbes Precision/Rappel...")
    plot_pr_comparison(
        X_train, y_train, X_test, y_test,
        out_path=OUT / "pr_curves_all.png",
    )

    print("Generation du barplot comparatif des metriques...")
    plot_metrics_comparison(results_df, out_path=OUT / "metrics_comparison.png")

    print("Generation du barplot des temps d'entrainement...")
    plot_fit_time_comparison(results_df, out_path=OUT / "fit_time_comparison.png")

    # -----------------------------------------------------------------------
    # ETAPE 7 : Optimisation du seuil sur la meilleure strategie
    # -----------------------------------------------------------------------
    _section("ETAPE 7 - Seuil de decision optimal (meilleure strategie)")
    best_strategy = str(best["strategy"])
    print(f"On re-entraine '{best['label']}' pour optimiser le seuil...")

    best_pipeline = build_strategy_pipeline(best_strategy)
    best_pipeline.fit(X_train, y_train)

    threshold_result = optimize_threshold(
        best_pipeline, X_test, y_test,
        metric="f1",
        output_dir=OUT,
    )

    print(f"\n  Seuil par defaut (0.50) :")
    print(f"    F1     = {threshold_result['f1_at_0_5']:.4f}")
    print(f"    Recall = {threshold_result['recall_at_0_5']:.4f}")
    print(f"    Faux negatifs (pannes non detectees) = {threshold_result['fn_at_0_5']}")
    print(f"\n  Seuil optimal ({threshold_result['optimal_threshold']:.2f}) :")
    print(f"    F1     = {threshold_result['optimal_score']:.4f}")
    print(f"    Faux negatifs = {threshold_result['fn_at_optimal']}")
    pannes_sauvees = threshold_result["fn_at_0_5"] - threshold_result["fn_at_optimal"]
    if pannes_sauvees > 0:
        print(
            f"\n  En utilisant le seuil optimal, on detecte {pannes_sauvees} pannes"
            " supplementaires qui etaient manquees au seuil 0.5."
        )

    # -----------------------------------------------------------------------
    # ETAPE 8 : Interpretation et conclusion
    # -----------------------------------------------------------------------
    _section("ETAPE 8 - Interpretation")
    print(
        "CONCLUSION :\n"
        f"  - L'imbalance ratio du dataset est {stats['ratio']:.1f}:1.\n"
        "  - Ignorer le desequilibre donne une accuracy trompeuse"
        f" ({stats['accuracy_naive'] * 100:.0f}%) avec Recall proche de 0.\n"
        f"  - La meilleure strategie '{best['label']}' ameliore le F1 a"
        f" {best['f1']:.3f} et le Recall a {best['recall']:.3f}.\n"
        f"  - Le seuil optimal ({threshold_result['optimal_threshold']:.2f}"
        f" au lieu de 0.50) reduit les faux negatifs"
        f" de {threshold_result['fn_at_0_5']} a {threshold_result['fn_at_optimal']}.\n"
        "  - En maintenance predictive, reduire les faux negatifs est"
        " critique : chaque panne non detectee peut immobiliser une ligne"
        " de production entiere."
    )

    _banner("SCRIPT 16 TERMINE")
    print(f"Tous les artefacts sauvegardes dans : {OUT}")
    print(
        "\nFichiers generes :\n"
        "  class_distribution.png      : distribution avant reequilibrage\n"
        "  pr_curves_all.png           : courbes PR par strategie\n"
        "  metrics_comparison.png      : barplot F1/Recall/Precision/PR-AUC\n"
        "  fit_time_comparison.png     : cout computationnel par strategie\n"
        "  threshold_optimization.png  : seuil optimal vs defaut\n"
        "  summary.csv / summary.json  : tableau recapitulatif\n"
        "  optimal_threshold_16.json   : seuil optimal de la meilleure strat."
    )


if __name__ == "__main__":
    main()
