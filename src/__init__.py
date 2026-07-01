# -*- coding: utf-8 -*-
"""Package source du projet Maintenance Prédictive Industrielle.

Ce package regroupe l'ensemble des modules nécessaires au pipeline complet ·
chargement des données, préparation, ingénierie de variables, modélisation
ML/DL, évaluation, interprétabilité et génération des schémas du rapport.

Structure des modules
---------------------
Structure par domaine métier
---------------------------
- src/data/        · config, data_loader, preprocessing
- src/models/      · models (binaire), models_multiclass, models_regression, tuning
- src/validation/  · evaluation, calibration, conformal, bootstrap
- src/analysis/    · interpretability, diagrams, imbalance

Compatibilité : les anciens imports plats (from src.config import X) fonctionnent
via les shims de rétrocompatibilité à la racine src/.

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
