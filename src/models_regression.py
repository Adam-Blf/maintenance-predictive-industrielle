# -*- coding: utf-8 -*-
"""Modeles de regression sur `rul_hours` (Remaining Useful Life).

Contexte métier
---------------
Tâche bonus du sujet · estimer le nombre d'heures restantes avant
défaillance (Remaining Useful Life). Utilité opérationnelle · permettre
au planning de maintenance de prioriser les interventions sur les
machines à durée de vie restante la plus faible, en optimisant les
créneaux d'arrêt programmé et en évitant les immobilisations imprévues.

Métriques d'évaluation utilisées
----------------------------------
- MAE (Mean Absolute Error) en heures : interprétable directement
  par les techniciens ("en moyenne, on se trompe de X heures").
- RMSE : pénalise davantage les grandes erreurs (une prédiction
  de RUL = 50h alors que la panne arrive dans 2h est catastrophique).
- R² : proportion de variance expliquée (1 = prédiction parfaite).

Spécificités de la tâche régression RUL
-----------------------------------------
- `rul_hours` est dans l'intervalle [0.5, 99.0] par construction.
- Les features sont les mêmes que pour la classification binaire ·
  pas besoin d'un ensemble de features séparé.
- Ridge est le modèle de référence linéaire, utile pour vérifier la
  linéarité résiduelle après StandardScaler.
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from .config import RANDOM_STATE
from .preprocessing import build_preprocessor


def build_ridge() -> Pipeline:
    """Ridge regression · baseline linéaire régularisée L2.

    Modèle de référence · si ce modèle simple obtient déjà un R² élevé,
    cela signifie que la relation entre les features et le RUL est
    essentiellement linéaire.

    Hyperparamètres justifiés
    --------------------------
    - `alpha=1.0` : régularisation L2 par défaut, équilibre biais/variance.
      Augmenter si les features sont trop corrélées (multicolinéarité),
      diminuer si les coefficients sont trop pénalisés.
    """
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("regressor", Ridge(alpha=1.0, random_state=RANDOM_STATE)),
        ]
    )


def build_rf_regressor() -> Pipeline:
    """Random Forest regressor · capture les non-linéarités et interactions.

    Chaque arbre prédit une valeur de RUL continue, et la prédiction
    finale est la moyenne de tous les arbres. L'averaging réduit la
    variance et stabilise les prédictions.

    Hyperparamètres justifiés
    --------------------------
    - `n_estimators=250` : suffisant pour la stabilité, au-delà les
      gains marginaux ne justifient pas le coût computationnel.
    - `min_samples_leaf=5` : légèrement plus grand qu'en classification
      (4) pour lisser les prédictions de régression et éviter de
      mémoriser les valeurs individuelles de RUL.
    - `max_depth=None` : profondeur libre, le bagging suffit à éviter
      le surapprentissage en régression sur ce volume de données.
    """
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=250,
                    max_depth=None,
                    min_samples_leaf=5,
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def build_xgb_regressor() -> Pipeline:
    """XGBoost regressor · état de l'art tabulaire pour la régression.

    Minimise la MSE (Mean Squared Error) via gradient boosting. Produit
    généralement les meilleures performances sur les datasets tabulaires
    de taille petite-à-moyenne.

    Hyperparamètres justifiés
    --------------------------
    - `n_estimators=400` : plus que pour la classification (300) car la
      tâche de régression bénéficie de plus d'arbres pour affiner les
      prédictions dans le spectre continu [0.5, 99].
    - `learning_rate=0.05` : faible pour converger progressivement et
      éviter le surapprentissage (compromis classique : faible LR +
      beaucoup d'arbres).
    - `objective="reg:squarederror"` : minimise le RMSE, équivalent à
      la régression OLS mais en gradient boosting.
    - `subsample=0.85`, `colsample_bytree=0.85` : régularisation
      stochastique, améliore la généralisation et réduit le RMSE test.
    - `tree_method="hist"` : algorithme histogram-based, compatible avec
      les grandes dimensions et rapide sur CPU.
    """
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "regressor",
                XGBRegressor(
                    n_estimators=400,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    objective="reg:squarederror",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    tree_method="hist",
                ),
            ),
        ]
    )


def build_mlp_regressor() -> Pipeline:
    """MLP regressor · réseau neuronal pour la prédiction de RUL.

    Le sujet exige explicitement l'inclusion d'un modèle DL/réseau de
    neurones. Le MLP est le choix naturel pour les données tabulaires.
    La couche de sortie est un neurone linéaire (pas d'activation),
    compatible avec une cible continue.

    Hyperparamètres justifiés
    --------------------------
    - Architecture `(64, 32, 16)` : identique aux MLP de classification
      pour cohérence comparative. La couche de sortie est automatiquement
      configurée à 1 neurone pour la régression.
    - `alpha=1e-3` : régularisation L2 légèrement plus forte que le
      défaut pour éviter le surapprentissage sur les valeurs RUL
      extrêmes (0.5h et 99h).
    - `early_stopping=True` : crucial en régression car le MLP peut
      commencer à mémoriser les valeurs de RUL individuelles si trop
      entraîné. La validation split interne (10% par défaut) détecte
      le surapprentissage.
    """
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "regressor",
                MLPRegressor(
                    hidden_layer_sizes=(64, 32, 16),
                    activation="relu",
                    solver="adam",
                    alpha=1e-3,
                    batch_size=256,
                    max_iter=200,
                    early_stopping=True,
                    n_iter_no_change=12,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
