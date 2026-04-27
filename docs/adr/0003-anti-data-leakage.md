# ADR 0003 · Stratégie anti data-leakage

- **Statut** · accepté
- **Date** · 2026-04-27

## Contexte

Le data leakage (contamination du train par des informations du test) est l'une des erreurs méthodologiques les plus fréquentes en Data Science. Le sujet impose explicitement *"Évitez le data leakage : un impératif méthodologique"*.

## Décision

Tout le preprocessing (imputation, scaling, encodage) est encapsulé dans une `sklearn.Pipeline` qui intègre `ColumnTransformer` + estimateur. Cela garantit que ·

1. Les statistiques d'imputation (médiane, mode) sont calculées **uniquement** sur le train set.
2. Les paramètres du `StandardScaler` (μ, σ) sont fittés **uniquement** sur le train set.
3. Le `OneHotEncoder` apprend les modalités **uniquement** sur le train set, avec `handle_unknown="ignore"` pour gérer les modalités inédites en production.

Le split train/test (80/20 stratifié sur la cible) est effectué **avant** tout `fit`, garantissant l'isolation.

## Alternative écartée

- Imputation/scaling globaux puis split · contamination silencieuse, scores faussement optimistes.

## Vérification

- Tests pytest dans `tests/test_preprocessing.py` valident l'invariant.
- Cross-validation 5-fold confirme que l'écart-type des scores reste faible (< 0.04 en F1).
