# -*- coding: utf-8 -*-
"""Définition des modèles ML/DL comparés dans le projet.

Le sujet impose au minimum 4 modèles dont 1 Deep Learning. On retient ·

1. **Logistic Regression** · baseline interprétable (référence).
2. **Random Forest** · ensemble bagging, capture les non-linéarités.
3. **XGBoost** · gradient boosting, état de l'art sur tabulaire.
4. **MLP (Multi-Layer Perceptron)** · réseau de neurones (DL exigé).

Chaque modèle est encapsulé dans une `sklearn.Pipeline` qui chaîne le
préprocesseur et l'estimateur. Cela garantit que toute transformation
ajustée sur le train est rejouée à l'identique sur le test/inférence.
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from .config import RANDOM_STATE
from .preprocessing import build_preprocessor


def build_logistic_regression() -> Pipeline:
    """Régression logistique · baseline interprétable.

    Hyperparamètres ·
      - `class_weight="balanced"` · compense le déséquilibre des classes
        (les pannes sont minoritaires comme dans la réalité).
      - `max_iter=1000` · suffisant pour converger sur ~24k lignes
        avec des features standardisées.
      - solver `lbfgs` · adapté aux problèmes multi-classes et binaires
        de taille moyenne, sans pénalité non-L2.
    """
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def build_random_forest() -> Pipeline:
    """Random Forest · ensemble d'arbres avec bagging.

    Hyperparamètres ·
      - 200 arbres · compromis biais/variance/temps. Plus élève la
        stabilité de la feature importance.
      - `max_depth=None` · on laisse les arbres pousser, la régularisation
        vient du bagging et de la diversité des splits.
      - `min_samples_leaf=5` · empêche les feuilles à 1 sample qui
        sur-apprennent fortement sur 24k lignes.
      - `n_jobs=-1` · parallélisation sur tous les coeurs CPU.
    """
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=None,
                    min_samples_leaf=5,
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def build_xgboost(scale_pos_weight: float = 1.0) -> Pipeline:
    """Gradient Boosting (XGBoost) · état de l'art sur tabulaire.

    Hyperparamètres ·
      - `n_estimators=300` + `learning_rate=0.05` · descente lente, plus
        robuste à l'overfit que 100 arbres avec lr=0.1.
      - `max_depth=6` · profondeur maximale par arbre, suffisante pour
        capturer les interactions sans exploser la complexité.
      - `subsample=0.85` + `colsample_bytree=0.85` · bagging stochastique
        intégré qui régularise sans sacrifier la précision.
      - `scale_pos_weight` · ratio neg/pos calculé à l'entraînement pour
        gérer le déséquilibre de classes (plus efficace que SMOTE pour
        XGBoost).
      - `eval_metric="logloss"` · métrique de référence pour
        classification probabiliste.
    """
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                XGBClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    scale_pos_weight=scale_pos_weight,
                    eval_metric="logloss",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    tree_method="hist",  # Méthode rapide
                ),
            ),
        ]
    )


def build_mlp() -> Pipeline:
    """Multi-Layer Perceptron · le modèle Deep Learning du projet.

    Architecture choisie · 64 → 32 → 16 (3 couches cachées dégressives).
    Cette pyramide inversée est un design éprouvé sur tabulaire · elle
    apprend des features de plus en plus abstraites tout en évitant la
    sur-paramétrisation (un MLP de >100k params sur 24k samples
    surapprend fatalement).

    Hyperparamètres ·
      - `activation="relu"` · standard, évite les vanishing gradients
        en early training.
      - `solver="adam"` · adaptatif, robuste aux hyperparamètres mal
        choisis. Bonne option par défaut.
      - `alpha=1e-3` · régularisation L2 modérée pour combattre l'overfit
        dû à la capacité du réseau.
      - `early_stopping=True` · arrête l'entraînement si le score de
        validation interne (10% du train) ne progresse plus pendant
        `n_iter_no_change=10` époques. Économie de temps + protection
        anti-overfit.
      - `max_iter=200` · plafond de sécurité, on s'arrête en pratique
        bien avant grâce à l'early stopping.
    """
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                MLPClassifier(
                    hidden_layer_sizes=(64, 32, 16),
                    activation="relu",
                    solver="adam",
                    alpha=1e-3,
                    batch_size=256,
                    learning_rate_init=1e-3,
                    max_iter=200,
                    early_stopping=True,
                    n_iter_no_change=10,
                    validation_fraction=0.1,
                    random_state=RANDOM_STATE,
                    verbose=False,
                ),
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Catalogue des modèles · clé = nom court (utilisé dans les artefacts et
# graphiques), valeur = factory function. Cette structure permet d'itérer
# proprement dans le script d'entraînement sans dupliquer la logique.
# ---------------------------------------------------------------------------
MODEL_CATALOG: dict[str, callable] = {
    "logistic_regression": build_logistic_regression,
    "random_forest": build_random_forest,
    "xgboost": build_xgboost,
    "mlp": build_mlp,
}


def get_model(name: str, **kwargs) -> Pipeline:
    """Retourne une instance fraîche du modèle demandé.

    Parameters
    ----------
    name : str
        Clé du modèle dans `MODEL_CATALOG`.
    **kwargs : dict
        Arguments transmis à la factory (ex. `scale_pos_weight` pour XGB).

    Returns
    -------
    Pipeline
        Pipeline non entraîné, prêt pour `.fit()`.

    Raises
    ------
    KeyError
        Si `name` n'existe pas dans le catalogue.
    """
    if name not in MODEL_CATALOG:
        raise KeyError(f"Modèle inconnu · {name}. Valeurs valides · {list(MODEL_CATALOG)}")
    return MODEL_CATALOG[name](**kwargs)
