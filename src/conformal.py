# -*- coding: utf-8 -*-
"""Prédiction conforme (split-conformal) pour la classification binaire.

À QUOI ÇA SERT ?
----------------
Un modèle qui dit « probabilité de panne 0.71 » ne quantifie pas son
incertitude · ce 0.71 est-il fiable ? La prédiction conforme transforme
les probabilités en **ensembles de prédiction** avec une garantie de
couverture distribution-free · à un niveau 1-alpha (ex. 90%), la vraie
classe est dans l'ensemble au moins 90% du temps, sans hypothèse sur la
distribution des données.

Méthode · LAC (Least Ambiguous set-valued Classifier), la variante
split-conformal la plus simple pour la classification.
  - score de non-conformité de l'exemple i pour sa vraie classe y_i ·
    s_i = 1 - p_model(y_i | x_i)
  - seuil q = quantile (1-alpha) empirique des scores de calibration
    (avec la correction de finitude (n+1)(1-alpha)/n)
  - à l'inférence · l'ensemble contient la classe c ssi 1 - p_c <= q

Interprétation des ensembles pour la maintenance ·
  - {1}            · panne quasi certaine → intervention prioritaire
  - {0}            · machine saine avec garantie
  - {0, 1}         · zone d'incertitude → inspection humaine recommandée
  - {} (vide)      · le modèle est hors de son domaine de confiance

Référence · Sadinle, Lei & Wasserman (2019), « Least Ambiguous
Set-Valued Classifiers With Bounded Error Levels ».
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ConformalBinaryClassifier:
    """Conformaliseur LAC pour un classifieur binaire déjà entraîné.

    Le modèle sous-jacent n'est pas ré-entraîné · on calibre uniquement
    un seuil sur un jeu de calibration disjoint du jeu d'entraînement.
    """

    alpha: float = 0.10  # taux d'erreur cible (1-alpha = couverture visée)
    qhat: float | None = None  # seuil conforme appris à la calibration

    def calibrate(self, proba_pos: np.ndarray, y_true: np.ndarray) -> "ConformalBinaryClassifier":
        """Apprend le seuil conforme sur le jeu de calibration.

        Parameters
        ----------
        proba_pos : probabilité de la classe positive P(y=1 | x).
        y_true    : vraies étiquettes binaires (0/1).
        """
        proba_pos = np.asarray(proba_pos, dtype=float)
        y_true = np.asarray(y_true, dtype=int)
        # Score de non-conformité = 1 - proba de la VRAIE classe.
        p_true = np.where(y_true == 1, proba_pos, 1.0 - proba_pos)
        scores = 1.0 - p_true
        n = len(scores)
        # Quantile conforme avec correction de finitude.
        level = np.ceil((n + 1) * (1.0 - self.alpha)) / n
        level = min(level, 1.0)
        self.qhat = float(np.quantile(scores, level, method="higher"))
        return self

    def predict_sets(self, proba_pos: np.ndarray) -> list[set[int]]:
        """Retourne l'ensemble de prédiction conforme de chaque exemple."""
        if self.qhat is None:
            raise RuntimeError("Appeler calibrate() avant predict_sets().")
        proba_pos = np.asarray(proba_pos, dtype=float)
        sets: list[set[int]] = []
        for p1 in proba_pos:
            # Inclure la classe c ssi 1 - p_c <= qhat.
            #   classe 0 · p_0 = 1 - p1 → 1 - p_0 = p1
            #   classe 1 · 1 - p_1 = 1 - p1
            s: set[int] = set()
            if p1 <= self.qhat:
                s.add(0)
            if (1.0 - p1) <= self.qhat:
                s.add(1)
            sets.append(s)
        return sets

    @staticmethod
    def coverage(sets: list[set[int]], y_true: np.ndarray) -> float:
        """Couverture empirique · fraction où la vraie classe est dans l'ensemble."""
        y_true = np.asarray(y_true, dtype=int)
        hits = sum(int(y in s) for s, y in zip(sets, y_true))
        return hits / len(y_true)

    @staticmethod
    def average_set_size(sets: list[set[int]]) -> float:
        """Taille moyenne des ensembles (1 = décision nette, 2 = ambigu)."""
        return float(np.mean([len(s) for s in sets]))
