# ADR 0002 · Choix des 4 modèles comparés

- **Statut** · accepté
- **Date** · 2026-04-27

## Contexte

Le sujet impose au moins 4 modèles dont 1 Deep Learning. Il faut couvrir le spectre complexité / interprétabilité pour permettre une comparaison didactique du compromis biais-variance.

## Décision

| # | Modèle              | Famille            | Rôle pédagogique                                       |
| - | ------------------- | ------------------ | ------------------------------------------------------ |
| 1 | Logistic Regression | Linéaire           | Baseline interprétable, point de référence             |
| 2 | Random Forest       | Bagging d'arbres   | Capture les non-linéarités, robuste aux outliers       |
| 3 | XGBoost             | Boosting d'arbres  | État de l'art, gère le déséquilibre via `scale_pos_weight` |
| 4 | MLP 64-32-16        | Deep Learning      | Modèle expressif, illustre le compromis surapprentissage |

## Justifications hyperparamètres

- **RF · 200 arbres + min_samples_leaf=5** · stabilité de la feature importance.
- **XGB · learning_rate=0.05 + n_estimators=300** · descente lente, plus robuste à l'overfit.
- **MLP · early_stopping + alpha=1e-3** · régularisation L2 modérée pour combattre l'overfit.
- Tous les modèles · `random_state=42` propagé pour reproductibilité.

## Alternatives écartées

- **LightGBM** · doublonne XGBoost, gain marginal sur 24k lignes.
- **CatBoost** · pas d'avantage sur features non textuelles ici.
- **SVM** · O(n²) prohibitif sur 24k lignes au fit.
