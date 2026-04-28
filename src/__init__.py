# -*- coding: utf-8 -*-
"""Package source du projet Maintenance Prédictive Industrielle.

Ce package regroupe l'ensemble des modules nécessaires au pipeline complet ·
chargement des données, préparation, ingénierie de variables, modélisation
ML/DL, évaluation, interprétabilité et génération des schémas du rapport.

Structure des modules
---------------------
- bootstrap      · auto-install des dépendances pip au lancement.
- config         · constantes globales (chemins, hyperparamètres, noms de colonnes).
- data_loader    · chargement du CSV Kaggle v3.0 (lecture seule, pas de génération).
- preprocessing  · pipeline sklearn ColumnTransformer (imputation, scaling, OHE).
- models         · 4 pipelines de classification binaire (LogReg, RF, XGB, MLP).
- models_multiclass   · 4 pipelines classification multi-classe (failure_type).
- models_regression   · 4 pipelines régression (rul_hours, Remaining Useful Life).
- evaluation     · métriques, matrices de confusion, courbes ROC/PR.
- calibration    · calibration probabiliste Platt/Isotonic + seuil métier optimal.
- tuning         · recherche bayésienne Optuna (TPE) sur RF, XGB, MLP.
- interpretability    · feature importance native, permutation importance, SHAP.
- diagrams       · schémas pédagogiques matplotlib pur (pipeline, biais-variance, etc.).

Exports publics
---------------
Aucun export direct au niveau package · chaque module s'importe explicitement.
Le `__version__` est exposé pour être consommé par le générateur de rapport.

Auteurs · Adam Beloucif, Emilien Morice
Cours · Projet Data Science · M1 Mastère Data Engineering & IA
École · EFREI Paris Panthéon-Assas Université
Année · 2025-2026
"""

# Numéro de version sémantique du projet, mirrorée dans le README et
# le rapport PDF pour garantir la cohérence du livrable.
# Incrémenter MAJOR sur changement de schéma, MINOR sur nouvelle feature,
# PATCH sur correction d'un bug ou enrichissement documentaire.
__version__ = "1.0.0"

# Métadonnées exposées au reste du package (notamment au générateur de
# rapport FPDF2 qui imprime ces informations en page de garde).
__authors__ = ["Adam Beloucif", "Emilien Morice"]
__institution__ = "EFREI Paris Panthéon-Assas Université"
__course__ = "M1 Mastère Data Engineering & IA · BC2 RNCP40875"
