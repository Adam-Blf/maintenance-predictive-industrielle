# -*- coding: utf-8 -*-
"""Modeles de classification multi-classe sur `failure_type`.

Contexte métier
---------------
Tâche bonus du sujet · identifier le type précis de panne parmi
{bearing, motor_overheat, hydraulic, electrical} (4 classes positives,
on filtre les observations 'none' au préalable dans le script 07).
Utilité opérationnelle · permettre à l'équipe de maintenance de
mobiliser la bonne pièce détachée et la bonne compétence AVANT
l'intervention, réduisant le MTTR (Mean Time To Repair).

Spécificités de la tâche multi-classe
--------------------------------------
- Les 4 classes sont raisonnablement équilibrées au sein des pannes,
  mais le déséquilibre global (85% none filtrés) est déjà géré au
  niveau du script par le filtrage préalable.
- `class_weight="balanced"` est utilisé sur LogReg et RF pour compenser
  les éventuels déséquilibres résiduels entre types de panne.
- XGBoost utilise `objective="multi:softprob"` pour obtenir des
  probabilités de classe (utile pour les métriques de calibration).
- Le MLP est partagé avec la classification binaire (même architecture
  64-32-16) car la complexité de la tâche est similaire.
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from .config import RANDOM_STATE
from .preprocessing import build_preprocessor


def build_logistic_multiclass() -> Pipeline:
    """Régression logistique multi-classe · baseline interprétable.

    Stratégie OvR (One-vs-Rest) implicite avec lbfgs : sklearn entraîne
    un classifieur binaire par classe en mode multinomial. C'est le
    modèle de référence le plus simple, utile pour vérifier que les
    features engineered apportent bien un signal discriminant.

    Hyperparamètres justifiés
    --------------------------
    - `solver="lbfgs"` : seul solveur supportant multinomial + class_weight
      simultanément avec convergence stable sur ~4k lignes.
    - `max_iter=1500` : valeur augmentée par rapport au défaut (100) car
      la convergence est plus lente en multi-classe et avec class_weight.
    - `class_weight="balanced"` : compense les déséquilibres entre types
      de pannes (ex. motor_overheat peut être plus rare que bearing).
    """
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1500,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def build_rf_multiclass() -> Pipeline:
    """Random Forest · adapté multi-classe nativement sans OvR.

    RF gère nativement les N classes en splitant selon la réduction
    d'impureté Gini multi-classe à chaque noeud. Pas besoin de
    décomposer en problèmes binaires contrairement à LogReg.

    Hyperparamètres justifiés
    --------------------------
    - `n_estimators=250` : bon compromis stabilité/vitesse sur ~4k lignes.
      Au-delà de 300, les gains de performance sont marginaux.
    - `max_depth=None` : arbres profonds pour capturer les interactions
      entre types de pannes et modes opératoires. Le bagging contrôle
      le surapprentissage plutôt que la profondeur.
    - `min_samples_leaf=4` : évite les feuilles avec trop peu d'exemples,
      particulièrement utile quand certains types de pannes sont rares.
    - `class_weight="balanced"` : poids inversement proportionnels aux
      fréquences de classe pour traiter équitablement les types rares.
    """
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=250,
                    max_depth=None,
                    min_samples_leaf=4,
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def build_xgb_multiclass(num_class: int) -> Pipeline:
    """XGBoost multi-classe avec softmax probabiliste.

    XGBoost construit des arbres en gradient boosting : chaque arbre
    corrège les erreurs du précédent. En multi-classe, il minimise la
    cross-entropie multi-classe (mlogloss) via `multi:softprob`.

    Parameters
    ----------
    num_class : int
        Nombre de classes, obligatoire pour l'objectif multi:softprob.
        Passé dynamiquement depuis le script 07 après LabelEncoder.

    Hyperparamètres justifiés
    --------------------------
    - `objective="multi:softprob"` : retourne des probabilités de classe
      (contrairement à multi:softmax qui retourne la classe seule).
      Nécessaire pour les métriques de calibration et les courbes ROC.
    - `learning_rate=0.07` : légèrement plus élevé que pour la binaire
      (0.05) car le dataset multi-classe est plus petit (~4k lignes).
    - `max_depth=6` : arbres modérément profonds pour capturer les
      interactions features x type de panne.
    - `subsample=0.85`, `colsample_bytree=0.85` : régularisation par
      sous-échantillonnage, réduit le surapprentissage sans réduire
      significativement la précision.
    - `tree_method="hist"` : algorithme histogram-based, 5-10x plus
      rapide que "exact" sur ce dataset (important pour l'écoresponsabilité).
    """
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                XGBClassifier(
                    n_estimators=300,
                    learning_rate=0.07,
                    max_depth=6,
                    objective="multi:softprob",
                    num_class=num_class,
                    eval_metric="mlogloss",
                    subsample=0.85,
                    colsample_bytree=0.85,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    tree_method="hist",
                ),
            ),
        ]
    )


def build_mlp_multiclass() -> Pipeline:
    """MLP multi-classe · réseau neuronal pour les patterns non-linéaires complexes.

    Architecture identique à la classification binaire (64-32-16) car la
    complexité de la tâche multi-classe est comparable, et les deux datasets
    ont des dimensions d'entrée identiques. La couche de sortie adapte
    automatiquement son nombre de neurones au nombre de classes.

    Hyperparamètres justifiés
    --------------------------
    - `hidden_layer_sizes=(64, 32, 16)` : entonnoir décroissant (compression
      progressive des représentations). Taille modeste pour éviter le
      surapprentissage sur ~4k lignes.
    - `activation="relu"` : évite le problème du gradient évanescent
      (vanishing gradient) des activations sigmoïde/tanh sur les couches
      profondes. Standard actuel pour les MLP tabulaires.
    - `alpha=1e-3` : régularisation L2 légèrement plus forte que le
      défaut (1e-4) pour compenser le faible volume de données.
    - `early_stopping=True`, `n_iter_no_change=12` : arrêt si la
      validation loss ne s'améliore pas pendant 12 epochs consécutives.
      Évite de sur-entraîner et réduit la consommation énergétique.
    - `batch_size=256` : mini-batch raisonnable sur ~3k lignes de train.
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
                    max_iter=200,
                    early_stopping=True,
                    n_iter_no_change=12,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
