# Matrice de conformité · Sujet projet + RNCP40875 Bloc 2

Document jury-ready · pour chaque exigence du sujet et chaque critère du
référentiel RNCP, on indique la **preuve concrète** dans le repo.

Dataset utilisé · `tatheerabbas/industrial-machine-predictive-maintenance` v3.0
(Kaggle CC0). Aucune génération synthétique en production.

---

## Exigences fonctionnelles du sujet (EF1 → EF5)

| EF | Exigence | État | Preuve |
|---|---|---|---|
| **EF1** | Pipeline preprocessing · NaN, encodage, normalisation, EDA | ✅ | `src/preprocessing.py` (ColumnTransformer · SimpleImputer + StandardScaler + OneHotEncoder) · `scripts/02_eda.py` produit 8 figures + 2 CSV dans `reports/02/` |
| **EF2** | ≥ 4 modèles dont 1 Deep Learning · sélection finale justifiée | ✅ | `scripts/03_train_models.py` · Logistic Regression + Random Forest + XGBoost + **MLP (Deep Learning)** · sélection via score `F1 - 0.5×σ(F1_CV)` |
| **EF3** | Métriques classification + comparaison + analyse erreurs | ✅ | 6 métriques (Acc, Prec, Rec, F1, ROC-AUC, PR-AUC) dans `reports/03/metrics_summary.csv` · ROC + PR + barplot + 4 matrices de confusion · CV 5-fold |
| **EF4** | Dashboard interactif obligatoire | ✅ | `dashboard/app.py` · Streamlit 5 onglets vision métier (État du parc / Plan d'intervention / Impact économique / Diagnostic / Détails techniques) |
| **EF5** | API REST (optionnel) | ✅ bonus | `api/main.py` · FastAPI · `POST /predict`, `GET /health`, `GET /model-info` · Swagger auto sur `/docs` |

## Tâches de prédiction (1 obligatoire, plusieurs en bonus)

| Tâche | Variable cible | Script | Bonus ? |
|---|---|---|---|
| **Classification binaire 24h** | `failure_within_24h` | `scripts/03_train_models.py` | Tâche principale obligatoire |
| Classification multi-classe | `failure_type` | `scripts/07_train_multiclass.py` | ✅ Bonus |
| Régression RUL | `rul_hours` | `scripts/08_train_regression.py` | ✅ Bonus |

## Recommandations méthodologiques du sujet (page 11-15)

| Recommandation | État | Preuve |
|---|---|---|
| EDA approfondie · distributions, NaN, corrélations | ✅ | `scripts/02_eda.py` · 8 figures dans `reports/02/` |
| Distribution des classes · métriques adaptées au déséquilibre | ✅ | F1 + PR-AUC privilégiées, pas l'accuracy seule · `class_weight=balanced` sur Logistic Regression |
| Corrélations + redondance des variables | ✅ | `eda_correlation_heatmap.png` + `eda_scatter_vib_temp.png` |
| **Baseline simple → complexification progressive** | ✅ | Section "ÉTAPE 3 · Baseline" puis "ÉTAPE 4 · 3 modèles plus puissants" dans `scripts/03_train_models.py` |
| ≥ 4 modèles · sélection finale argumentée | ✅ | 4 modèles · score composite F1 − 0.5×σ pour pénaliser l'instabilité |
| **Pas de data leakage** · pipelines sklearn | ✅ | `sklearn.Pipeline` partout · `fit` sur train uniquement, `transform` sur test |
| Cross-validation | ✅ | `StratifiedKFold(n_splits=5)` dans `scripts/03_train_models.py` ÉTAPE 5 |
| Tuning hyperparamètres justifié | ✅ | `scripts/09_tune_hyperparams.py` · Optuna TPE · résultats dans `reports/09/tuning_results.json` |
| Analyse d'erreurs · matrices de confusion + résidus | ✅ | 4 matrices de confusion `reports/03/confusion_matrix_*.png` · scatter pred vs true régression `reports/08/regression_pred_vs_true.png` |
| **Interprétabilité** · 3 niveaux | ✅ | `scripts/04_interpret.py` · feature_importance natif + permutation_importance + SHAP (8 figures dans `reports/04/`) |
| Code structuré modulaire | ✅ | `src/` (13 modules) + `scripts/` (10 scripts numérotés) + `dashboard/` + `api/` + `tests/` |
| Versioning Git régulier | ✅ | `git log --oneline` · commits granulaires anglais impératif |
| Dashboard décisionnel autonome | ✅ | `dashboard/app.py` orienté responsable maintenance · KPIs métier en première page |
| API testée indépendamment | ✅ | `tests/test_api.py` · `httpx.TestClient` |

---

## Critères RNCP40875 Bloc 2 (BC2)

### A.3 · Préparation et visualisation de données

| Critère | Évaluation | Preuve |
|---|---|---|
| **C3.1** Préparer/nettoyer les données | Données prêtes pour ML | `src/preprocessing.py` (ColumnTransformer) · `scripts/02_eda.py` (NaN, schéma) · documentation inline pédagogique |
| **C3.2** Tableau de bord interactif inclusif | Dashboard responsif · graphiques interactifs | `dashboard/app.py` · Plotly (interactif), filtres multi-select, layout responsive Streamlit `width="stretch"` |
| **C3.3** EDA · techniques statistiques | Insights métier exploitables | 8 figures EDA · stats descriptives par capteur · % NaN · corrélations |

### A.4 · Implémentation d'algorithmes ML

| Critère | Évaluation | Preuve |
|---|---|---|
| **C4.1** Stratégie d'intégration de l'IA | Cas d'usage pertinent + ROI | Onglet "Impact économique" du dashboard · section "Contexte métier" du rapport (livré manuellement) |
| **C4.2** Développer modèles prédictifs | Modèle efficace | 4 modèles (LogReg, RF, XGBoost, MLP) · `models/final_model.joblib` · F1=0.886 sur test |
| **C4.3** Évaluer + écoresponsabilité | Comparaison rigoureuse + impact carbone | 6 métriques · CV 5-fold · `reports/03/metrics_summary.csv` colonnes `fit_time_s` + `predict_time_ms` · `reports/03/compute_cost_comparison.png` (temps train vs latence inference) |

---

## Livrables exigés par le sujet (page 16)

| Livrable | État | Chemin |
|---|---|---|
| **Code source de la solution fonctionnelle** | ✅ | Tout le repo · `src/` + `scripts/` + `dashboard/` + `api/` + `tests/` |
| **Rapport du projet** | 📝 manuel | `rapport_projet_data_science.pdf` rédigé à la main (Word/LaTeX) à partir des artefacts `reports/02..05,07-10/` |
| **Support de Présentation** | 📝 manuel | `presentation.pptx` rédigé à la main (PowerPoint) avec les figures `reports/02..05,07-10/` |
| Documentation technique annexe (optionnel) | ✅ | `docs/adr/` (Architecture Decision Records) · `README.md` (badges, arbo, install) |
| Soumission Git/GitHub (optionnel) | ✅ | https://github.com/Adam-Blf/maintenance-predictive-industrielle |

---

## Reproductibilité

- **Seed unique** · `RANDOM_STATE=42` propagé à tous les algorithmes stochastiques (numpy, sklearn, xgboost, MLP)
- **Auto-install des dépendances** · `src/bootstrap.py` lance `pip install -r requirements.txt` au premier run de chaque script
- **Chemins portables** · `Path(__file__).resolve().parent.parent` partout · marche sur Windows/Linux/macOS
- **Tests automatisés** · `pytest -q` · 23 tests passent en ~7s
- **Versioning sémantique** · `__version__` dans `src/__init__.py`, mirrée dans le README

## Que faire si on clone le repo from scratch ?

```bash
# 1. Cloner
git clone https://github.com/Adam-Blf/maintenance-predictive-industrielle
cd maintenance-predictive-industrielle

# 2. Télécharger le dataset Kaggle (clé API requise)
kaggle datasets download -d tatheerabbas/industrial-machine-predictive-maintenance
unzip industrial-machine-predictive-maintenance.zip -d data/raw/

# 3. Lancer la chaîne · le bootstrap installe les deps automatiquement
python scripts/02_eda.py             # EDA → reports/02/
python scripts/03_train_models.py    # train 4 modèles → models/ + reports/03/
python scripts/04_interpret.py       # SHAP → reports/04/
python scripts/05_generate_diagrams.py  # schémas → reports/05/
# Rapport PDF + slides PPTX rédigés ensuite à la main avec les figures produites

# 4. Lancer dashboard ou API
streamlit run dashboard/app.py       # dashboard métier
uvicorn api.main:app --reload        # API REST + Swagger sur /docs
```

---

*Document généré pour la soutenance du projet Data Science M1.*
*Auteurs · Adam Beloucif · Emilien Morice · EFREI Paris Panthéon-Assas Université.*
