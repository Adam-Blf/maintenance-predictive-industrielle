# -*- coding: utf-8 -*-
"""Hyperparameter tuning via Optuna · TPE sampler.

GridSearchCV vs RandomizedSearch vs Optuna
--------------------------------------------
- **GridSearchCV** : explore exhaustivement toutes les combinaisons.
  Coût = produit cartésien des grilles. Impraticable dès que l'espace
  dépasse 3-4 dimensions ou que les plages sont larges.
- **RandomizedSearchCV** : échantillonnage aléatoire uniforme dans
  l'espace. Meilleur que Grid sur grands espaces, mais aveugle : il ne
  tire aucun enseignement des essais précédents.
- **Optuna TPE (Tree-Parzen Estimator)** : modèle probabiliste bayésien
  qui apprend quelles régions de l'espace donnent de bons résultats et
  y concentre les essais suivants. Converge vers un bon optimum local
  en 20-50 essais là où GridSearch en nécessiterait des centaines.

Pourquoi 3-fold au lieu de 5-fold dans `_cv_f1` ?
---------------------------------------------------
Le tuning est lancé sur un sous-échantillon de 8000 lignes. Avec 3-fold,
chaque fold de validation a ~2700 lignes, suffisant pour une estimation
stable du F1. 5-fold doublerait le temps de tuning sans gain significatif.

Logistic Regression non tunée
-------------------------------
LogReg a seulement C (inverse de la régularisation) comme hyperparamètre
impactant, et `class_weight='balanced'` est déjà optimal pour les données
déséquilibrées. Le gain attendu d'un tuning Optuna est trop faible pour
justifier le coût computationnel.
"""

from __future__ import annotations

import warnings
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from .config import RANDOM_STATE


def _silence_optuna() -> None:
    """Reduit la verbosite Optuna pour ne pas polluer la console."""
    try:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        pass


def _cv_f1(model, X, y, n_splits: int = 3) -> float:
    """Score F1 moyen via cross-validation stratifiee (rapide, 3-fold)."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for tr, te in skf.split(X, y):
        model.fit(X.iloc[tr], y.iloc[tr])
        scores.append(f1_score(y.iloc[te], model.predict(X.iloc[te])))
    return float(np.mean(scores))


def tune_random_forest(
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = 25,
) -> dict:
    """Optimise les hyperparamètres du Random Forest via Optuna TPE.

    Espace de recherche
    --------------------
    - `n_estimators` : [100, 400, pas 50] · peu d'impact au-delà de 300.
    - `max_depth` : [4, 30] · profondeur totalement libre non testée car
      risk de mémorisation sur les petits sous-échantillons.
    - `min_samples_leaf` : [1, 10] · contrôle la granularité des feuilles.
    - `max_features` : {sqrt, log2, None} · sqrt est le défaut RF classique,
      log2 plus agressif (moins de features par split = plus de diversité),
      None = toutes les features (converge vers Bagging).

    Parameters
    ----------
    X : pd.DataFrame
        Features d'entraînement (sous-échantillon de 8000 lignes).
    y : pd.Series
        Cible binaire failure_within_24h.
    n_trials : int, default=25
        Nombre d'essais Optuna. 25 est un compromis qualité/temps
        (chaque essai = 1 entraînement RF x 3 folds = 3 fits).

    Returns
    -------
    dict
        Dictionnaire avec "best_params" et "best_value" (F1 moyen CV).
    """
    import optuna
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline

    from .preprocessing import build_preprocessor

    _silence_optuna()

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
            "max_depth": trial.suggest_int("max_depth", 4, 30),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = Pipeline(
                steps=[
                    ("preprocessor", build_preprocessor()),
                    (
                        "classifier",
                        RandomForestClassifier(
                            **params,
                            class_weight="balanced",
                            n_jobs=-1,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            )
            return _cv_f1(model, X, y)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return {"best_params": study.best_params, "best_value": study.best_value}


def tune_xgboost(
    X: pd.DataFrame,
    y: pd.Series,
    scale_pos_weight: float,
    n_trials: int = 25,
) -> dict:
    """Optimise les hyperparamètres XGBoost via Optuna TPE.

    Espace de recherche
    --------------------
    - `n_estimators` : [150, 500] · plus large que RF car XGB utilise
      un faible learning_rate (besoin de plus d'itérations).
    - `learning_rate` : [0.02, 0.2] en log-scale · la log-scale explore
      équitablement les ordres de grandeur (0.02, 0.05, 0.1, 0.2).
    - `max_depth` : [3, 10] · arbres peu profonds pour un boosting rapide.
    - `subsample`, `colsample_bytree` : [0.6, 1.0] · régularisation.
    - `min_child_weight` : [1, 10] · poids minimum dans un noeud fils,
      équivalent XGBoost du `min_samples_leaf` de RF.

    Parameters
    ----------
    X : pd.DataFrame
        Features d'entraînement.
    y : pd.Series
        Cible binaire.
    scale_pos_weight : float
        Ratio négatifs/positifs pour compenser le déséquilibre de classes.
        Calculé depuis le script appelant : (y==0).sum() / (y==1).sum().
    n_trials : int, default=25
        Nombre d'essais Optuna.

    Returns
    -------
    dict
        Dictionnaire avec "best_params" et "best_value" (F1 moyen CV).
    """
    import optuna
    from sklearn.pipeline import Pipeline
    from xgboost import XGBClassifier

    from .preprocessing import build_preprocessor

    _silence_optuna()

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 500, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        }
        model = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    XGBClassifier(
                        **params,
                        scale_pos_weight=scale_pos_weight,
                        eval_metric="logloss",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                        tree_method="hist",
                    ),
                ),
            ]
        )
        return _cv_f1(model, X, y)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return {"best_params": study.best_params, "best_value": study.best_value}


def tune_mlp(
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = 15,
) -> dict:
    """Optimise les hyperparamètres MLP via Optuna TPE.

    Espace de recherche
    --------------------
    - `architecture` : 6 architectures candidates prédéfinies plutôt
      qu'un espace continu (évite des configurations absurdes comme
      128-8-256 et limite l'espace à des entonnoirs décroissants sensés).
    - `alpha` : [1e-5, 1e-2] en log-scale · régularisation L2.
    - `learning_rate_init` : [1e-4, 1e-2] en log-scale · step de l'
      optimiseur Adam. Trop grand → divergence, trop petit → lenteur.

    Note : seulement 15 essais par défaut (vs 25 pour RF/XGB) car
    chaque essai MLP est plus long à entraîner (forward + backward pass
    sur 200 epochs max) et l'espace est plus petit (3 hyperparamètres).

    Parameters
    ----------
    X : pd.DataFrame
        Features d'entraînement.
    y : pd.Series
        Cible binaire.
    n_trials : int, default=15
        Nombre d'essais Optuna (réduit vs RF/XGB pour équilibrer le temps).

    Returns
    -------
    dict
        Dictionnaire avec "best_params" et "best_value" (F1 moyen CV).
    """
    import optuna
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline

    from .preprocessing import build_preprocessor

    _silence_optuna()

    def objective(trial: "optuna.Trial") -> float:
        # Architectures candidates (compromis expressivité / surapprentissage).
        archi = trial.suggest_categorical(
            "architecture",
            ["32", "64", "32-16", "64-32", "64-32-16", "128-64-32"],
        )
        hidden = tuple(int(x) for x in archi.split("-"))
        alpha = trial.suggest_float("alpha", 1e-5, 1e-2, log=True)
        lr_init = trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = Pipeline(
                steps=[
                    ("preprocessor", build_preprocessor()),
                    (
                        "classifier",
                        MLPClassifier(
                            hidden_layer_sizes=hidden,
                            alpha=alpha,
                            learning_rate_init=lr_init,
                            max_iter=120,
                            early_stopping=True,
                            n_iter_no_change=10,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            )
            return _cv_f1(model, X, y)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return {"best_params": study.best_params, "best_value": study.best_value}


def tune_all(
    X: pd.DataFrame,
    y: pd.Series,
    n_trials_each: int = 20,
) -> dict[str, dict]:
    """Lance le tuning sur les 3 modeles tunables · RF, XGB, MLP."""
    scale_pos_weight = float((y == 0).sum() / max((y == 1).sum(), 1))
    results = {}
    print(f"[TUNING] Random Forest ({n_trials_each} trials)...")
    results["random_forest"] = tune_random_forest(X, y, n_trials=n_trials_each)
    print(f"  best F1 = {results['random_forest']['best_value']:.4f}")
    print(f"[TUNING] XGBoost ({n_trials_each} trials)...")
    results["xgboost"] = tune_xgboost(
        X, y, scale_pos_weight=scale_pos_weight, n_trials=n_trials_each
    )
    print(f"  best F1 = {results['xgboost']['best_value']:.4f}")
    print(f"[TUNING] MLP ({max(10, n_trials_each // 2)} trials)...")
    results["mlp"] = tune_mlp(X, y, n_trials=max(10, n_trials_each // 2))
    print(f"  best F1 = {results['mlp']['best_value']:.4f}")
    return results
