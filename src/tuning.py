# -*- coding: utf-8 -*-
"""Hyperparameter tuning via Optuna · TPE sampler.

Optuna est plus efficace que GridSearchCV exhaustif sur des espaces
larges · le sampler TPE (Tree-Parzen Estimator) apprend ou chercher
prometteur a partir des essais precedents, ce qui converge vers les
bonnes regions en quelques dizaines d'essais.

On expose une fonction par modele tunable (RF, XGB, MLP). Logistic
Regression est skip · trop peu d'hyperparametres pour justifier le
tuning (juste C avec class_weight='balanced').
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
    """Optimise RF · n_estimators, max_depth, min_samples_leaf, max_features."""
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
    """Optimise XGBoost · learning_rate, max_depth, subsample, colsample."""
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
    """Optimise MLP · hidden_layer_sizes, alpha, learning_rate_init."""
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
