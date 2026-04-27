# ADR 0004 · Implémentation des tâches bonus (multi-classe + régression)

- **Statut** · accepté
- **Date** · 2026-04-27

## Contexte

Le sujet mentionne explicitement *"Points Bonus · Pour plusieurs tâches de prédiction"*. La tâche principale est la classification binaire `failure_within_24h`. Les tâches additionnelles possibles sont ·

1. Classification multi-classe sur `failure_type` (Mechanical / Thermal / Electrical / Hydraulic).
2. Régression sur `rul_hours` (Remaining Useful Life).

## Décision

On implémente les 3 tâches en parallèle, avec 4 modèles sur chacune ·

| Tâche | Variable | Modèles |
| --- | --- | --- |
| Binaire | `failure_within_24h` | Logistic / RF / XGBoost / MLP |
| Multi-classe | `failure_type` (filtre `!= None`) | Logistic / RF / XGBoost / MLP |
| Régression | `rul_hours` | Ridge / RF / XGBoost / MLP |

Métriques par tâche ·

- **Binaire** · accuracy, precision, recall, F1, ROC-AUC, PR-AUC.
- **Multi-classe** · accuracy, macro-F1, weighted-F1, classification report.
- **Régression** · MAE, RMSE, R².

## Conséquences

- 12 modèles entraînés au total (4 × 3 tâches), persistés dans `models/`.
- Le rapport présente une section dédiée par tâche.
- Le dashboard expose les 3 prédictions dans le simulateur.
- L'API ajoute des endpoints `/predict-failure-type` et `/predict-rul`.

## Justification métier

- **Binaire** répond à *"prioriser les interventions"*.
- **Multi-classe** répond à *"mobiliser la bonne équipe technique"* (mécanicien vs électricien).
- **Régression** répond à *"planifier les fenêtres d'arrêt programmé"*.
