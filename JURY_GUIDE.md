# Guide Jury - Maintenance Predictive Industrielle

Reference rapide besoin metier -> fichier:ligne. Utiliser `Ctrl+G` dans l'IDE pour aller directement a la ligne.

---

## 1. Configuration centrale

> "Quels sont les hyperparametres ? Ou est defini le dataset ?"

| Besoin | Fichier | Ligne |
|--------|---------|-------|
| Chemin dataset Kaggle (`predictive_maintenance_v3.csv`) | `src/config.py` | 93 |
| Graine aleatoire (`RANDOM_STATE = 42`) | `src/config.py` | 104 |
| Taille test set (`TEST_SIZE = 0.20`) | `src/config.py` | 108 |
| Nombre de folds CV (`CV_FOLDS = 5`) | `src/config.py` | 112 |
| Variable cible binaire (`failure_within_24h`) | `src/config.py` | 116 |
| Features numeriques (7 capteurs) | `src/config.py` | 134 |
| Features categorielles | `src/config.py` | 145 |
| Charte couleurs EFREI | `src/config.py` | 169 |
| Creation des dossiers de sortie | `src/config.py` | 178 |

---

## 2. Chargement des donnees

> "D'ou viennent les donnees ? Comment le dataset est-il valide ?"

| Besoin | Fichier | Ligne |
|--------|---------|-------|
| Schema officiel Kaggle (15 colonnes) | `src/data_loader.py` | 33 |
| Chargement CSV + validation schema | `src/data_loader.py` | 52 |
| Politique anti-donnees-synthetiques | `src/data_loader.py` | 11 |
| Reference Kaggle CC0 (24 042 lignes) | `src/data_loader.py` | 13 |

---

## 3. Preprocessing (encodage, scaling, split)

> "Comment les features sont-elles preparees ? Pourquoi StandardScaler ?"

| Besoin | Fichier | Ligne |
|--------|---------|-------|
| Construction du ColumnTransformer | `src/preprocessing.py` | 43 |
| Justification imputer mediane (robuste outliers) | `src/preprocessing.py` | 53 |
| Justification StandardScaler vs MinMaxScaler | `src/preprocessing.py` | 59 |
| One-Hot Encoder avec handle_unknown="ignore" | `src/preprocessing.py` | 65 |
| Recuperation noms de features post-OHE | `src/preprocessing.py` | 108 |

---

## 4. Modeles binaires (4 modeles)

> "Quels modeles avez-vous compare ? Justifiez vos choix."

| Modele | Fichier | Ligne | F1 |
|--------|---------|-------|----|
| Logistic Regression (baseline) | `src/models.py` | 28 | 0.747 |
| Random Forest (bagging) | `src/models.py` | 55 | 0.863 |
| XGBoost (gradient boosting) | `src/models.py` | 85 | **0.886** |
| MLP - couches 64-32-16 (Deep Learning) | `src/models.py` | 123 | 0.836 |
| Catalogue des 4 modeles + factory pattern | `src/models.py` | 175 |  |
| Methode `get_model(name)` | `src/models.py` | 183 |  |

---

## 5. Cross-validation 5-fold stratifiee

> "Comment avez-vous evite l'overfitting ?"

| Besoin | Fichier | Ligne |
|--------|---------|-------|
| Metriques F1/Precision/Recall/ROC-AUC/PR-AUC | `src/evaluation.py` | 59 |
| Calcul des metriques sur test set | `src/evaluation.py` | 84 |
| Matrice de confusion normalisee | `src/evaluation.py` | 129 |
| Courbes ROC superposees (4 modeles) | `src/evaluation.py` | 171 |
| Courbes Precision-Recall | `src/evaluation.py` | 224 |
| CV 3-fold rapide (tuning) | `src/tuning.py` | 53 |

---

## 6. Selection du meilleur modele (XGBoost)

> "Pourquoi XGBoost ? Quelle est la formule de selection ?"

- **Formule** : `score = F1_test - 0.5 * sigma(F1_CV)` - penalise l'instabilite
- XGBoost : F1=0.886, ROC-AUC=0.995 sur test set

| Justification | Fichier | Ligne |
|---------------|---------|-------|
| Parametres XGBoost + `scale_pos_weight` | `src/models.py` | 85 |
| Justification n_estimators=300, lr=0.05 | `src/models.py` | 88 |
| Regularisation subsample + colsample | `src/models.py` | 93 |

---

## 7. Gestion du desequilibre de classes (5 strategies)

> "Comment gerez-vous le desequilibre ? Qu'est-ce que SMOTE ?"

| Strategie | Fichier | Ligne |
|-----------|---------|-------|
| Analyse du desequilibre (ratio, IR, accuracy naive) | `src/imbalance.py` | 144 |
| Construction pipeline imblearn par strategie | `src/imbalance.py` | 250 |
| Evaluation d'une strategie | `src/imbalance.py` | 358 |
| Comparaison des 5 strategies (DataFrame) | `src/imbalance.py` | 404 |
| Optimisation du seuil de decision | `src/imbalance.py` | 602 |
| Strategies disponibles : baseline/SMOTE/ADASYN/under/SMOTETomek | `src/imbalance.py` | 119 |

---

## 8. Interpretabilite (SHAP, permutation importance)

> "Comment expliquez-vous les predictions ? Pourquoi vibration_rms est importante ?"

| Methode | Fichier | Ligne |
|---------|---------|-------|
| Importance native (Random Forest / XGBoost) | `src/interpretability.py` | 51 |
| Permutation Importance (agnostique au modele) | `src/interpretability.py` | 107 |
| SHAP values (additivite, coherence) | `src/interpretability.py` | 177 |
| Choix TreeExplainer vs KernelExplainer | `src/interpretability.py` | 219 |

---

## 9. Calibration et seuil optimal

> "Pourquoi le seuil 0.5 n'est pas optimal ? Quel est le cout metier ?"

| Besoin | Fichier | Ligne |
|--------|---------|-------|
| Reliability diagram + Brier Score | `src/calibration.py` | 56 |
| Cout metier : FN=1000EUR, FP=100EUR | `src/calibration.py` | 127 |
| Courbe cout total vs seuil | `src/calibration.py` | 127 |
| Sauvegarde du seuil optimal | `src/calibration.py` | 201 |

---

## 10. Prediction conforme (incertitude)

> "Comment quantifiez-vous l'incertitude ? Que signifie {0,1} ?"

| Besoin | Fichier | Ligne |
|--------|---------|-------|
| Classe ConformalBinaryClassifier | `src/conformal.py` | 38 |
| Calibration du seuil conforme (quantile) | `src/conformal.py` | 48 |
| Prediction d'ensembles ({0}, {1}, {0,1}, {}) | `src/conformal.py` | 68 |
| Couverture empirique | `src/conformal.py` | 87 |

---

## 11. Classification multi-classe (type de panne)

> "Pouvez-vous identifier le type de panne : bearing, hydraulic... ?"

| Modele | Fichier | Ligne |
|--------|---------|-------|
| Logistic Regression multi-classe (OvR) | `src/models_multiclass.py` | 38 |
| Random Forest multi-classe natif | `src/models_multiclass.py` | 71 |
| XGBoost multi:softprob | `src/models_multiclass.py` | 108 |
| MLP multi-classe | `src/models_multiclass.py` | 159 |

---

## 12. Regression RUL (duree de vie restante)

> "Pouvez-vous predire combien d'heures avant la prochaine panne ?"

| Modele | Fichier | Ligne |
|--------|---------|-------|
| Ridge regression (baseline lineaire) | `src/models_regression.py` | 41 |
| Random Forest regressor | `src/models_regression.py` | 62 |
| XGBoost regressor (reg:squarederror) | `src/models_regression.py` | 96 |
| MLP regressor | `src/models_regression.py` | 139 |

---

## 13. Hyperparameter tuning (Optuna TPE)

> "Comment avez-vous optimise les hyperparametres ? Pourquoi Optuna ?"

| Besoin | Fichier | Ligne |
|--------|---------|-------|
| Tuning Random Forest (espace + justification) | `src/tuning.py` | 63 |
| Tuning XGBoost (log-scale learning_rate) | `src/tuning.py` | 136 |
| Tuning MLP (architectures candidates) | `src/tuning.py` | 215 |
| Lancement sequentiel RF + XGB + MLP | `src/tuning.py` | 296 |

---

## 14. Diagrammes pedagogiques

> "Pouvez-vous expliquer l'architecture du systeme ?"

| Diagramme | Fichier | Ligne |
|-----------|---------|-------|
| Pipeline ML (EDA -> preprocessing -> train -> eval) | `src/diagrams.py` | ~80 |
| Architecture systeme (data -> modele -> API -> dashboard) | `src/diagrams.py` | ~150 |

---

## 15. Bootstrap et dependances

> "Comment installez-vous les dependances ? Le projet est-il portable ?"

| Besoin | Fichier | Ligne |
|--------|---------|-------|
| Auto-installation dependances manquantes | `src/bootstrap.py` | 112 |
| Portabilite des chemins (Win/Linux/Mac) | `src/bootstrap.py` | 14 |

---

## Metriques cles a retenir

| Modele | F1 | ROC-AUC | Precision | Recall |
|--------|----|---------|-----------|--------|
| Logistic Regression | 0.747 | - | - | - |
| Random Forest | 0.863 | - | - | - |
| **XGBoost** | **0.886** | **0.995** | - | - |
| MLP | 0.836 | - | - | - |

- Dataset : 24 042 lignes, 15 colonnes, CC0 Kaggle
- Split : 80% train / 20% test, stratifie, `RANDOM_STATE=42`
- CV : 5-fold stratifie (anti-overfitting)
- Desequilibre : ~15% pannes (classe 1), ratio ~6:1
