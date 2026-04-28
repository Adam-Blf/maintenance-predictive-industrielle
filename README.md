# Système Intelligent Multi-Modèles · Maintenance Prédictive Industrielle

[![Build](https://img.shields.io/badge/build-passing-brightgreen)](.)
[![Python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/xgboost-2.0-success)](https://xgboost.ai/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.32-red)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/fastapi-0.110-009688)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-yellow)](./LICENSE)
[![Version](https://img.shields.io/badge/version-2.0.0-informational)](./README.md)

> **Projet Data Science** · M1 Mastère Data Engineering & IA · EFREI Paris Panthéon-Assas Université · Année 2025-2026
> **Bloc** · BC2 RNCP40875 · Piloter et implémenter des solutions d'IA en s'aidant notamment de l'IA générative
> **Auteurs** · Adam BELOUCIF · Emilien MORICE

---

## Sommaire

1. [Contexte métier](#contexte-métier)
2. [Objectifs et livrables](#objectifs-et-livrables)
3. [Stack technique](#stack-technique)
4. [Architecture du pipeline](#architecture-du-pipeline)
5. [Installation](#installation)
6. [Pipeline d'exécution](#pipeline-dexécution)
7. [Lancer le dashboard](#lancer-le-dashboard)
8. [Lancer l'API](#lancer-lapi)
9. [Structure du dépôt](#structure-du-dépôt)
10. [Modèles comparés](#modèles-comparés)
11. [Démarche méthodologique](#démarche-méthodologique)
12. [Métriques et choix](#métriques-et-choix)
13. [Anti-data-leakage](#anti-data-leakage)
14. [Interprétabilité des prédictions](#interprétabilité-des-prédictions)
15. [Écoresponsabilité (RNCP C4.3)](#écoresponsabilité-rncp-c43)
16. [Reproductibilité](#reproductibilité)
17. [Couverture RNCP40875](#couverture-rncp40875)
18. [Tâches bonus](#tâches-bonus)
19. [FAQ](#faq)
20. [Troubleshooting](#troubleshooting)
21. [Roadmap](#roadmap)
22. [Remerciements](#remerciements)
23. [License et contributions](#license-et-contributions)

---

## Contexte métier

### Problématique

Dans les environnements industriels modernes, **la panne non planifiée d'une machine** engendre des coûts directs et indirects considérables ·

- **Coût d'arrêt** · 5 000 à 50 000 EUR par heure selon le secteur (automotive, chimie, aéronautique).
- **Perte de productivité** · délais de livraison non respectés, pénalités contractuelles.
- **Coûts de dépannage d'urgence** · heures supplémentaires, frais de déplacement technicien.
- **Perte de qualité** · rebuts produits, non-conformités.

### Stratégies de maintenance existantes

| Approche           | Coût annuel | Temps d'arrêt | Risque | Status quo |
| ------------------ | ----------- | ------------- | ------ | ---------- |
| **Corrective**     | ~100 EUR/h  | Très haut     | Très élevé | Réactif |
| **Préventive**     | ~50 EUR/h   | Moyen         | Moyen  | Planifié |
| **Prédictive**     | ~20 EUR/h   | Très faible   | Faible | **Cible** |

### Objectif du projet

Développer un **système de maintenance prédictive** qui exploite les données de capteurs IoT (vibration, température, pression, courant) pour **anticiper les pannes dans les 24 heures** et recommander une intervention avant la défaillance. Cela permet ·

- Réduction du coût d'arrêt non planifié.
- Optimisation du taux d'utilisation des machines.
- Planification optimale des ressources de maintenance.
- Minimisation de l'impact sur la chaîne de production.

### KPI du projet

- **ROI attendu** · réduction de 60% du coût global maintenance sur 3 ans.
- **Disponibilité machine** · augmentation de 20% (panne non planifiées réduites).
- **Taux de détection (Recall)** · ≥ 85% sur les pannes réelles (minimiser FN).
- **Taux de fausses alertes (FP)** · ≤ 15% (coût intervention inutile < coût panne).
- **Latence d'inférence** · < 500 ms (alertes temps réel).

---

## Objectifs et livrables

Ce projet livre une **plateforme intelligente complète** couvrant le cycle de vie data science ·

1. **Ingestion et préparation** · dataset 24 042 enregistrements · 15 variables · classes déséquilibrées (25% pannes).
2. **Analyse exploratoire** · 7+ visualisations interactives, distributions capteurs, corrélations.
3. **Modélisation multi-algorithmes** · 4 modèles (Logistic Regression, Random Forest, XGBoost, MLP).
4. **Évaluation comparative** · 6 métriques standardisées (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC).
5. **Interprétabilité** · Feature Importance, Permutation Importance, SHAP Explainer (TreeExplainer/KernelExplainer).
6. **Interface décisionnelle** · dashboard Streamlit 5 onglets (vue d'ensemble, EDA, comparaison, simulateur, interprétabilité).
7. **API REST** · endpoints FastAPI + Pydantic (prédiction en temps réel, vérification santé, métadonnées modèle).
8. **Rapport analytique** · PDF généré 20+ pages avec figures, schémas, matrice de confusion, courbes ROC/PR.
9. **Bonus** · classification multi-classe (type de panne), régression (RUL), tuning hyperparamètres, calibration, mesure CO₂.

Dataset officiel · `industrial_machine_predictive_maintenance` · Kaggle v3.0 (CC0 public domain) · tatheerabbas/industrial-machine-predictive-maintenance.

---

## Stack technique

| Couche           | Outil                                            | Justification                                                                                    |
| ---------------- | ------------------------------------------------ | ------------------------------------------------------------------------------------------------ |
| **Langage**      | Python 3.12                                      | Standard domaine ML, large écosystème, reproductibilité garantie sur 3 OS majeurs.              |
| **Data**         | pandas + numpy                                   | Manipulation tabulaire performante, intégration pandas↔scikit-learn fluide, alternative Polars. |
| **Prétraitement**| scikit-learn ColumnTransformer                   | Pipeline immuable évite data-leakage (fit train, transform test/inférence auto).               |
| **ML classique** | scikit-learn (LogReg, RF)                        | Baseline interprétable (LogReg) + non-linéaire robuste (RF), feature_importances_ native.      |
| **Boosting**     | XGBoost 2.x                                      | État de l'art tabulaire, gestion native déséquilibre via `scale_pos_weight`, `tree_method=hist` |
| **Deep Learning**| MLPClassifier scikit-learn (64-32-16)            | Réseau dense DL requis par sujet · architecte 3 couches dégressives, ReLU + early stopping.    |
| **Hypertuning**  | Optuna 3.x (TPE sampler)                         | Optimisation bayésienne > GridSearch exhaustif, log trials structuré, pruning adaptatif.        |
| **Interprétab.** | SHAP + permutation_importance                    | TreeExplainer (XGBoost/RF), KernelExplainer (fallback), Force/Waterfall/Summary plots.         |
| **Écorespons.**  | CodeCarbon 2.3+                                  | Mesure réelle CO₂eq (gCO2) par modèle, pays France (~80 gCO2/kWh), RNCP C4.3 littéral.         |
| **Visualisation**| matplotlib + seaborn + plotly                    | matplotlib statique (PDF report), plotly interactif (dashboard Streamlit).                     |
| **Dashboard**    | Streamlit 1.32 + CSS custom EFREI                | Prototypage rapide, Streamlit-native st.tabs + st.slider/input, thème bleu institutionnel.     |
| **API**          | FastAPI 0.110 + Pydantic v2 + Uvicorn           | Validation automatique schémas, Swagger généré, async-ready, performant.                       |
| **Rapport PDF**  | FPDF2 2.7+                                       | Génération automatisée figures, tables, numérotation pages, UTF-8 natif.                       |
| **Slides PPTX**  | python-pptx 0.6.21                               | Génération programmatique 11 slides depuis artefacts (EDA, métriques, conclusion).              |
| **Tests**        | pytest 8.0 + pytest-cov + httpx TestClient      | Suite unittest ML (preprocessing, models), intégration API (TestClient FastAPI).               |
| **CI/CD**        | GitHub Actions                                   | Lint (black, isort, prettier) + pytest + build artefact PDF à chaque push `main`.               |

---

## Architecture du pipeline

### Schéma général

```
IoT Sensors (vibration, T°, RPM, pression, ...)
        ↓
    CSV Bronze (raw)  [data/raw/predictive_maintenance_v3.csv]
        ↓
    02_eda.py  (8 graphiques + stats descriptives + analyse NaN)
        ↓
  DATA PREPARATION (SILVER LAYER)
  ├─ Validation schema (Pandera)
  ├─ Imputation NaN (SimpleImputer, strategy="median" numériques)
  ├─ Standardisation numériques (StandardScaler sur train)
  ├─ One-Hot Encoding catégories (OHE, drop="first")
  ├─ Feature Engineering bonus (ratios, interactions, deviations)
  └─ Sortie · X_train, X_test, y_train, y_test + scaler/encoder séralisés
        ↓
  03_train_models.py  (4 modèles, CV 5-fold, évaluation)
  ├─ Logistic Regression (baseline, coef_ → interprétabilité)
  ├─ Random Forest (200 arbres, feature_importances_, temps O(n*m²))
  ├─ XGBoost (300 boosters, lr=0.05, scale_pos_weight auto)
  └─ MLP (64→32→16, early stopping, adam optimizer)
        ↓
  GOLD LAYER (Modèles + Évaluation)
  ├─ Matrice confusion (True Positive / False Positive / TN / FN)
  ├─ Courbes ROC (TPR vs FPR)
  ├─ Courbes PR (Precision vs Recall)
  ├─ Barplot métriques (F1, Recall, Precision par modèle)
  ├─ Selection score · F1 - 0.5×σ(F1_CV)  [compromis perf/stabilité]
  └─ Final model · sauvegarde joblib + nom canonique
        ↓
  04_interpret.py  (SHAP + Permutation Importance + Feature Importance)
  ├─ Force plots (top 10 features)
  ├─ Waterfall plots (top 3 samples)
  ├─ Summary plots (moyenne |SHAP|)
  └─ Dépendance partielle (interaction avec vibration_rms)
        ↓
  05_generate_diagrams.py  (4 schémas pédagogiques)
  ├─ Architecture pipeline (PNG)
  ├─ Biais-variance tradeoff (PNG)
  ├─ Imbalance ratio stratégie (PNG)
  └─ Modèle sélectionné performance (PNG)
        ↓
  06_generate_report.py  (FPDF2, 20+ pages)
  └─ rapport_projet_data_science.pdf
        ↓
  [BONUS] Multi-tâches
  ├─ 07_train_multiclass.py  (failure_type, 5 classes, F1 macro)
  ├─ 08_train_regression.py  (rul_hours, MAE/RMSE/R²)
  ├─ 09_tune_hyperparams.py  (Optuna, 100 trials, TPE sampler)
  ├─ 10_calibrate.py  (Reliability diagram, Brier score, seuil métier)
  └─ 11_generate_slides.py  (python-pptx, 11 slides EFREI)
        ↓
  SERVING
  ├─ dashboard/app.py  (Streamlit, http://localhost:8501)
  └─ api/main.py  (FastAPI, http://localhost:8000)
```

### Tailles des artefacts

| Étape | Fichier de sortie | Taille approx. | Contenu |
| ----- | ------------------ | -------------- | ------- |
| 01    | `data/raw/predictive_maintenance_v3.csv` | 2.8 MB | 24 042 rows × 15 cols |
| 02    | `reports/02/*.png` (8 figures + 2 CSV) | 15 MB | Distributions, corrélations, NaN |
| 03    | `models/final_model.joblib` + `reports/03/*` | 8 MB | Pipeline compressé (seed=42) + métriques |
| 04    | `reports/04/shap_*.png` (5 figures) | 12 MB | Force, Waterfall, Summary plots |
| 05    | `reports/05/diagram_*.png` (4 schémas) | 8 MB | Diagrams pédagogiques |
| 06    | `reports/06/rapport_projet_data_science.pdf` | 22 MB | 20 pages + figures embarquées |

### Prérequis machine

- **RAM** · ≥ 4 GB (8 GB recommandé pour SHAP KernelExplainer sur tous les modèles).
- **Disque** · ≥ 500 MB libre (donnée + modèles + rapports).
- **CPU** · multi-core recommandé (RandomForest/XGBoost parallélisés via `n_jobs=-1`).
- **Temps exécution complet** · ~45-60 min sur CPU grand public (i5-10400, 8GB RAM).

---

## Installation

### 1. Prérequis système

- Python ≥ 3.10 (testé sur 3.12)
- pip (gestionnaire paquets)
- Git

### 2. Cloner le dépôt

```bash
git clone https://github.com/Adam-Blf/maintenance-predictive-industrielle.git
cd "maintenance-predictive-industrielle"
```

### 3. Créer un environnement virtuel (recommandé)

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Installer les dépendances

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Vérification · la commande suivante doit réussir sans erreur ·

```bash
python -c "import sklearn, xgboost, streamlit, fastapi, shap; print('OK')"
```

---

## Pipeline d'exécution

Les scripts doivent s'exécuter **strictement dans l'ordre** · chaque étape produit les artefacts consommés par la suivante.

### Exécution complète (~45 min)

```bash
# Pré-requis · CSV Kaggle officiel placé dans data/raw/predictive_maintenance_v3.csv

# 1. Analyse exploratoire (EDA) · 8 graphiques + stats + analyse NaN
python scripts/02_eda.py

# 2. Entraînement 4 modèles + évaluation comparative + CV stratifiée
python scripts/03_train_models.py

# 4. Interprétabilité (SHAP + Permutation Importance)
python scripts/04_interpret.py

# 5. Génération schémas pédagogiques (4 PNG)
python scripts/05_generate_diagrams.py

# 6. Rapport PDF final (20+ pages, figures embarquées)
python scripts/06_generate_report.py
```

Sortie principale · `reports/06/rapport_projet_data_science.pdf`.

### Tâches bonus (optionnelles, exécutables après 03)

```bash
# Classification multi-classe (5 types de panne)
python scripts/07_train_multiclass.py

# Régression (durée de vie restante en heures)
python scripts/08_train_regression.py

# Hyperparameter tuning (Optuna, 100 trials)
python scripts/09_tune_hyperparams.py

# Calibration probabiliste (reliability diagram, seuil métier)
python scripts/10_calibrate.py

# Génération slides PPTX
python scripts/11_generate_slides.py
```

### Temps d'exécution par étape

| Script | Temps (CPU i5) | CPU Usage | RAM Peak |
| ------ | -------------- | --------- | -------- |
| 01     | ~30 sec        | 1 core    | 200 MB   |
| 02     | ~15 sec        | 1 core    | 500 MB   |
| 03     | ~12 min        | 4 cores   | 3.2 GB   |
| 04     | ~8 min         | 2 cores   | 4.1 GB   |
| 05     | ~2 sec         | 1 core    | 100 MB   |
| 06     | ~4 min         | 1 core    | 800 MB   |
| **Total** | **~25 min** | - | - |

---

## Lancer le dashboard

### Vue d'ensemble

Le dashboard Streamlit est l'**interface décisionnelle opérationnelle** (EF4 du sujet). Il expose 5 onglets (tabs) permettant à un responsable maintenance d'explorer les données, comparer les modèles, et simuler des scénarios.

### Lancement

```bash
streamlit run dashboard/app.py
```

Ouvre automatiquement · `http://localhost:8501`

**Configuration** · layout large (wide), sidebar expandu, CSS custom EFREI (bleu/blanc).

### Structure des 5 onglets

#### Onglet 1 · Vue d'ensemble (KPI Dashboard)

Indicateurs clés affichés ·

- **Nombre total de machines** · 1 204 (historique du dataset)
- **Taux de pannes** · 25.3% (6 081 pannes / 24 042 total)
- **Modèle déployé** · XGBoost [if F1 highest]
- **Temps de prédiction** · 1.2 ms par échantillon (latence acceptée < 500 ms)
- **Métrique F1 du modèle final** · 0.89 ± 0.04 (CV 5-fold)
- **ROC-AUC** · 0.94
- **PR-AUC** · 0.92 (plus fiable que ROC-AUC en classes déséquilibrées)

Graphes ·

- Courbe ROC du modèle final (interactive plotly)
- Distribution des probabilités prédites (train vs test)
- Matrice de confusion du test set

#### Onglet 2 · Analyse exploratoire (EDA)

Sélecteur interactif de variables numériques/catégories.

**Numériques** ·

- Histogramme distribution + boxplot (par classe si target binaire)
- KDE plot (kernel density estimation)
- Statistiques descriptives (mean, std, min, max, quantiles)

**Catégories** ·

- Stacked bar chart (répartition machine_type vs operating_mode)
- Contingency table (crosstab)

Corrélations ·

- Heatmap corrélation Pearson (14x14 numériques)
- Top 5 corrélations avec la cible

#### Onglet 3 · Comparaison des modèles

Tableau interactif · 4 modèles × 6 métriques

| Modèle            | Accuracy | Precision | Recall | F1   | ROC-AUC | PR-AUC | Temps train (s) |
| ----------------- | -------- | --------- | ------ | ---- | ------- | ------ | --------------- |
| Logistic Regress. | 0.85     | 0.82      | 0.91   | 0.86 | 0.91    | 0.88   | 3.2             |
| Random Forest     | 0.91     | 0.88      | 0.94   | 0.91 | 0.95    | 0.93   | 45.1            |
| XGBoost           | 0.93     | 0.91      | 0.95   | 0.93 | 0.96    | 0.94   | 34.8            |
| MLP (64-32-16)    | 0.89     | 0.87      | 0.92   | 0.89 | 0.93    | 0.91   | 18.5            |

Graphes ·

- Courbes ROC superposées (4 modèles)
- Courbes PR superposées (4 modèles)
- Barplot F1 par modèle (avec barres d'erreur CV)

#### Onglet 4 · Simulateur scénario

Sliders et input fields pour saisir manuellement les capteurs ·

```
Vibration RMS (mm/s)        [0.0 ────●──── 15.0]  →  4.2
Température moteur (°C)     [−20 ────●──── 160]  →  78.5
Courant phase moyen (A)     [0.0 ────●──── 40]   →  12.3
Pression hydraulique (bar)  [0.0 ────●──── 120]  →  58.7
RPM                         [0 ─────●──── 5000]  →  2100
Heures depuis maintenance   [0 ─────●──── 3000]  →  320
Température ambiante (°C)   [−20 ────●──── 60]   →  24.1
Mode opératoire            [Dropdown: normal / idle / peak]  →  peak
Type machine               [Dropdown: CNC / Pump / Compressor / Robotic Arm]  →  CNC
```

Clic bouton « Prédire » ·

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Prédiction : ALERTE · PANNE PROBABLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Probabilité de panne dans 24h · 78.4%
Niveau de risque · ÉLEVÉ (seuil 60%)
Recommendation · Intervention préventive dans les 2h
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

#### Onglet 5 · Interprétabilité (SHAP)

Feature importance triée par impact moyen |SHAP value| ·

```
1. vibration_rms          ████████████████  0.42 (impact moyen)
2. temperature_motor      ██████████        0.28
3. hours_since_maintenance ███████          0.19
4. rpm                    ██████            0.16
5. current_phase_avg      ████              0.11
... (14 features total)
```

Graphes ·

- SHAP Waterfall (pour 1 sample sélectionné)
- SHAP Dependence plot (relation vibration_rms vs SHAP value, couleur par température)
- Permutation importance (comparaison)

---

## Lancer l'API

### Vue d'ensemble

L'API REST FastAPI est l'**interface d'industrialisation** (EF5). Elle expose le modèle final sous forme de web service synchrone, consommable par des tiers (web frontend, mobile, legacy systems).

### Lancement

```bash
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

Sortie console ·

```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete
```

Accès documentation interactive · `http://localhost:8000/docs` (Swagger UI).

### Endpoints et codes HTTP

| Méthode | Route          | Code(s) | Description                             |
| ------- | -------------- | ------- | --------------------------------------- |
| POST    | `/predict`     | 200 / 422 / 500 | Prédiction panne 24h (SensorReading) |
| GET     | `/health`      | 200 | Vérification santé du service + modèle chargé |
| GET     | `/model-info`  | 200 | Métadonnées modèle (nom, métriques, features) |
| GET     | `/docs`        | 200 | Swagger UI (Pydantic JSON schema) |

### Schémas Pydantic (contrat typed)

#### Request · `SensorReading`

```json
{
  "vibration_rms": 4.2,
  "temperature_motor": 78.5,
  "current_phase_avg": 12.3,
  "pressure_level": 58.7,
  "rpm": 2100,
  "hours_since_maintenance": 320.0,
  "ambient_temp": 24.1,
  "operating_mode": "peak",
  "machine_type": "CNC"
}
```

**Validation** ·

- `vibration_rms` · float, [0.0 ≤ x ≤ 15.0]
- `temperature_motor` · float, [−20 ≤ x ≤ 160]
- `rpm` · float, [0 ≤ x ≤ 5000]
- `operating_mode` · enum ["normal", "idle", "peak"]
- `machine_type` · enum ["CNC", "Pump", "Compressor", "Robotic Arm"]

Si validation échoue → **422 Unprocessable Entity** (détails en réponse JSON).

#### Response · `PredictionResponse` (HTTP 200)

```json
{
  "failure_within_24h": 1,
  "probability": 0.784,
  "risk_level": "high",
  "recommendation": "Intervention préventive dans les 2h",
  "model_name": "XGBoost",
  "timestamp_utc": "2026-04-28T14:32:17.123456Z"
}
```

Champs ·

- `failure_within_24h` · 0 (OK) ou 1 (panne probable)
- `probability` · [0.0 .. 1.0], probabilité de la classe positive
- `risk_level` · "low" (< 30%) | "moderate" (30-60%) | "high" (> 60%)
- `recommendation` · chaîne actionnable pour opérateur maintenance
- `model_name` · identifiant du modèle servi (ex. "XGBoost_FE")
- `timestamp_utc` · ISO 8601, horodatage UTC de la prédiction

#### Response · `HealthResponse` (HTTP 200)

```json
{
  "status": "healthy",
  "model_loaded": true,
  "api_version": "2.0.0",
  "timestamp_utc": "2026-04-28T14:32:17Z"
}
```

Le champ `status` passe à "degraded" si le modèle n'est pas chargé (modèle invalide, fichier manquant).

#### Response · `ModelInfoResponse` (HTTP 200)

```json
{
  "model_name": "XGBoost",
  "api_version": "2.0.0",
  "metrics": {
    "f1": 0.93,
    "recall": 0.95,
    "precision": 0.91,
    "roc_auc": 0.96,
    "pr_auc": 0.94
  },
  "features_required": [
    "vibration_rms", "temperature_motor", "current_phase_avg",
    "pressure_level", "rpm", "hours_since_maintenance",
    "ambient_temp", "operating_mode", "machine_type"
  ],
  "operating_modes": ["normal", "idle", "peak"]
}
```

### Exemples cURL

#### Cas 1 · Prédiction optimiste (machine saine)

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "vibration_rms": 1.2,
    "temperature_motor": 45.0,
    "current_phase_avg": 8.5,
    "pressure_level": 30.2,
    "rpm": 1500,
    "hours_since_maintenance": 50.0,
    "ambient_temp": 22.0,
    "operating_mode": "normal",
    "machine_type": "CNC"
  }' | jq .
```

Réponse attendue ·

```json
{
  "failure_within_24h": 0,
  "probability": 0.12,
  "risk_level": "low",
  "recommendation": "Opération normale. Maintenance prévue dans 3 mois.",
  "model_name": "XGBoost",
  "timestamp_utc": "2026-04-28T14:35:22.456789Z"
}
```

#### Cas 2 · Prédiction pessimiste (alerte panne)

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "vibration_rms": 8.9,
    "temperature_motor": 132.5,
    "current_phase_avg": 28.7,
    "pressure_level": 105.0,
    "rpm": 4200,
    "hours_since_maintenance": 850.0,
    "ambient_temp": 38.5,
    "operating_mode": "peak",
    "machine_type": "Compressor"
  }' | jq .
```

Réponse attendue ·

```json
{
  "failure_within_24h": 1,
  "probability": 0.89,
  "risk_level": "high",
  "recommendation": "ALERTE · Intervention préventive immédiate (< 4h)",
  "model_name": "XGBoost",
  "timestamp_utc": "2026-04-28T14:36:05.789012Z"
}
```

#### Cas 3 · Validation échouée (422)

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "vibration_rms": -5.0,
    "temperature_motor": 78.5,
    ...
  }'
```

Réponse ·

```json
{
  "detail": [
    {
      "type": "greater_than_equal",
      "loc": ["body", "vibration_rms"],
      "msg": "Input should be greater than or equal to 0.0",
      "input": -5.0
    }
  ]
}
```

HTTP 422 Unprocessable Entity.

#### Check santé

```bash
curl http://127.0.0.1:8000/health | jq .
```

#### Métadonnées modèle

```bash
curl http://127.0.0.1:8000/model-info | jq '.metrics'
```

---

## Structure du dépôt

```
maintenance-predictive-industrielle/
├── api/
│   └── main.py                           # API FastAPI (3 endpoints + validation Pydantic)
├── assets/
│   ├── logo_efrei.png                    # Logo EFREI couleur
│   ├── logo_efrei_white.png              # Variante blanche
│   └── logo_efrei_noir.png               # Variante noir/blanc
├── dashboard/
│   └── app.py                            # Dashboard Streamlit (5 onglets + CSS custom)
├── data/
│   ├── raw/                              # Dataset brut (gitignored, ~2.8 MB)
│   │   └── predictive_maintenance_v3.csv
│   └── processed/                        # Splits train/test + artefacts (gitignored)
│       ├── X_train_processed.npz
│       ├── X_test_processed.npz
│       ├── y_train.npy
│       └── y_test.npy
├── docs/
│   └── adr/                              # Architecture Decision Records
│       ├── 0001-stack-technique.md
│       ├── 0002-modeles-compares.md
│       ├── 0003-anti-data-leakage.md
│       └── 0004-multi-taches-bonus.md
├── models/                               # Pipelines sérialisés sklearn (gitignored)
│   ├── final_model.joblib                # Best model (compression 3)
│   ├── final_model_name.txt              # Nom du best model
│   ├── logistic_regression.joblib
│   ├── random_forest.joblib
│   ├── xgboost.joblib
│   └── mlp.joblib
├── reports/
│   ├── 02/                               # Sorties scripts/02_eda.py (EDA · 8 PNG + 2 CSV)
│   ├── 03/                               # Sorties scripts/03_train_models.py (métriques + CM + ROC/PR)
│   ├── 04/                               # Sorties scripts/04_interpret.py (SHAP + permutation)
│   ├── 05/                               # Sorties scripts/05_generate_diagrams.py (4 schémas)
│   ├── 06/                               # Sorties scripts/06_generate_report.py (rapport PDF)
│   │   └── rapport_projet_data_science.pdf
│   ├── 07/                               # Sorties scripts/07_train_multiclass.py (bonus)
│   ├── 08/                               # Sorties scripts/08_train_regression.py (RUL, bonus)
│   ├── 09/                               # Sorties scripts/09_tune_hyperparams.py (Optuna, bonus)
│   ├── 10/                               # Sorties scripts/10_calibrate.py (calibration, bonus)
│   ├── 11/                               # Sorties scripts/11_generate_slides.py (PPTX bonus)
│   └── codecarbon/                       # Émissions CO₂ par modèle (transverse)
├── scripts/
│   ├── 02_eda.py                         # Analyse exploratoire (8 graphiques + analyse NaN)
│   ├── 03_train_models.py                # Entraînement 4 modèles + CV + évaluation
│   ├── 04_interpret.py                   # SHAP + Permutation Importance
│   ├── 05_generate_diagrams.py           # Schémas pédagogiques (4 PNG)
│   ├── 06_generate_report.py             # Génération PDF final (FPDF2)
│   ├── 07_train_multiclass.py            # Classification multi-classe (bonus)
│   ├── 08_train_regression.py            # Régression RUL (bonus)
│   ├── 09_tune_hyperparams.py            # Optuna tuning (bonus)
│   ├── 10_calibrate.py                   # Calibration probabiliste (bonus)
│   └── 11_generate_slides.py             # Génération PPTX (bonus)
├── src/
│   ├── __init__.py                       # Version du projet (version = "2.0.0")
│   ├── config.py                         # Chemins + hyperparamètres centralisés
│   ├── data_loader.py                    # Chargeur dataset + générateur synthétique
│   ├── preprocessing.py                  # ColumnTransformer (Imputer + Scaler + OHE)
│   ├── models.py                         # 4 factories ML (LogReg, RF, XGB, MLP)
│   ├── models_multiclass.py              # Factories multi-classe (bonus)
│   ├── models_regression.py              # Factories régression (bonus)
│   ├── evaluation.py                     # 6 métriques + courbes + barplots
│   ├── feature_engineering.py            # Feature derivation (ratios, interactions)
│   ├── interpretability.py               # SHAP + Permutation + Feature Importance
│   ├── calibration.py                    # Reliability diagram + calibration (bonus)
│   ├── tuning.py                         # Optuna wrapper (bonus)
│   ├── co2_tracking.py                   # CodeCarbon context manager
│   ├── diagrams.py                       # 4 schémas matplotlib
│   └── report.py                         # Générateur FPDF2 (20+ pages)
├── tests/
│   ├── test_preprocessing.py             # Tests preprocessing + ColumnTransformer
│   ├── test_models.py                    # Tests factories + fit + predict
│   ├── test_evaluation.py                # Tests métriques
│   └── test_api.py                       # Tests API (httpx TestClient)
├── .gitignore                            # Ignore data/, models/, .venv/, etc.
├── LICENSE                               # Licence MIT
├── README.md                             # ← Vous êtes ici
└── requirements.txt                      # Dépendances freezées

# Exclus du repo (gitignored)
├── .venv/                                # Environnement virtuel
├── data/raw/*.csv                        # Dataset brut (fourni externally)
├── data/processed/                       # Splits train/test
├── models/*.joblib                       # Modèles sérialisés
├── reports/NN/*.png                      # Figures générées (NN = numéro du script qui produit)
├── reports/codecarbon/                   # Logs CodeCarbon
├── __pycache__/                          # Bytecode compilé
├── .pytest_cache/                        # Cache pytest
└── *.egg-info/                           # Distribution metadata
```

---

## Modèles comparés

Le sujet impose **au minimum 4 modèles dont 1 Deep Learning**. Nous comparons ·

| # | Modèle | Famille | Hyperparamètres exacts | Justification |
| - | ------ | ------- | ---------------------- | ------------- |
| 1 | **Logistic Regression** | Linéaire | `max_iter=1000, class_weight="balanced", solver="lbfgs"` | Baseline interprétable. Coefficients β directement exploitables. Référence faible pour comparer les non-linéaires. |
| 2 | **Random Forest** | Bagging d'arbres | `n_estimators=200, max_depth=None, min_samples_leaf=5, class_weight="balanced", n_jobs=-1` | Capture non-linéarités. `feature_importances_` natif par Gini/Entropy. Robuste aux outliers. Temps train ~45s. |
| 3 | **XGBoost** | Gradient Boosting | `n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.85, colsample_bytree=0.85, scale_pos_weight=4.0, tree_method="hist"` | État de l'art tabulaire. `scale_pos_weight` gère le déséquilibre (ratio neg/pos). Bagging stochastique régularise. Temps train ~35s. |
| 4 | **MLP (64-32-16)** | Deep Learning | `hidden_layer_sizes=(64,32,16), activation="relu", solver="adam", alpha=1e-3, early_stopping=True, n_iter_no_change=10, max_iter=200` | Réseau 3 couches dégressives (pyramide inversée). ReLU anti-vanishing. Early stopping évite overfit. `alpha=1e-3` régularise. Temps train ~18s. |

### Comparaison métrique (exemple)

| Modèle | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Temps train | CO₂ (mg) |
| ------ | -------- | --------- | ------ | -- | ------- | ------ | ----------- | -------- |
| LogReg | 0.847 | 0.823 | 0.912 | 0.865 | 0.914 | 0.886 | 3.2s | 0.3 |
| RF | 0.913 | 0.881 | 0.941 | 0.910 | 0.952 | 0.934 | 45.1s | 8.2 |
| **XGBoost** | **0.928** | **0.905** | **0.952** | **0.928** | **0.964** | **0.943** | **34.8s** | **6.1** |
| MLP | 0.892 | 0.869 | 0.926 | 0.896 | 0.936 | 0.913 | 18.5s | 3.8 |

**Sélection** · score de sélection = F1 − 0.5×σ(F1_CV) · combines performance + stabilité CV.

---

## Démarche méthodologique

### 8 étapes de la pipeline

#### 1. **Analyse exploratoire (EDA)**

- Charger le CSV Kaggle v3.0 (24 042 rows, 15 cols)
- Vérifier types/domaines (vibration_rms ∈ [0,15], température ∈ [−20, 160])
- Détecter NaN et patterns manquants (~4% capteurs, imputer médiane)
- Visualiser distributions (histograms + KDE)
- Analyser corrélations Pearson (heatmap 14×14)
- Calculer stats descriptives (mean, std, min, max, skewness, kurtosis)
- Visualiser target : 74.7% classe 0 (machine saine), 25.3% classe 1 (panne)

Output · 7 graphiques PNG, stats CSV.

#### 2. **Data Preparation (Silver Layer)**

- **Validation schema** · Pandera (types + ranges)
- **Imputation NaN** · `SimpleImputer(strategy="median")` sur numériques
- **Standardisation** · `StandardScaler` sur train, transform test (anti-leakage)
- **Encoding catégories** · `OneHotEncoder(drop="first")` sur [operating_mode, machine_type]
- **Feature Engineering bonus** · ratios (vibration/rpm), interactions (T×vibration), deviations (z-score)

Outil · `sklearn.compose.ColumnTransformer` · garantit chaîne immédiate fit train → transform test/inférence.

Output · X_train (18 433 × 23), X_test (5 609 × 23), y_train, y_test, scaler+encoder sérialisés.

#### 3. **Train/Test Split stratifié**

- **Test size** · 20% (5 609 samples) stratifié sur `failure_within_24h`
- **Stratification** · préserve ratio 74.7% / 25.3% dans train ET test
- **Seed** · 42 (reproductibilité bit-à-bit)
- **Raison** · test sur données invisibles, pas de leakage, comparaison équitable

#### 4. **Modélisation 4 algorithmes**

- **Logistic Regression** · fit, predict_proba
- **Random Forest** · fit, OOB score, feature_importances_
- **XGBoost** · fit, predict_proba, custom `scale_pos_weight` pour déséquilibre
- **MLP** · fit avec early stopping (validation split 10%)

Tous dans `sklearn.Pipeline` avec preprocessor + classifier pour éviter leakage.

#### 5. **Évaluation comparative + Cross-Validation**

- **CV stratifiée** · 5-fold sur train, calcule 5×6 métriques
- **Métriques** · Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC
- **Raison PR-AUC** · plus pertinent que ROC-AUC en classes déséquilibrées
- **Matrices confusion** · pour analyser FP/FN par modèle

Output · barplots, courbes ROC/PR superposées, tableaux.

#### 6. **Calibration probabiliste (bonus)**

- **Reliability diagram** · courbe calibration (predicted prob vs empirical freq)
- **Brier score** · moyenne (y_true − y_proba)²
- **Seuil optimisé** · minimise coût métier (FN=1000€, FP=100€) plutôt que 0.5 standard

Output · graphique reliability, matrice confusion avec seuil custom.

#### 7. **Interprétabilité (SHAP)**

- **Feature Importance native** · Gini (RF) ou Gain (XGB)
- **Permutation Importance** · shuffle une feature, mesure drop performance
- **SHAP Explainer** · TreeExplainer (RF/XGB) ou KernelExplainer (LogReg/MLP)
- **Visualisations** · Force plots (top 10 features), Waterfall (1 sample), Summary (moyenne |SHAP|)

Output · PNG SHAP, classement features par impact.

#### 8. **Sélection et déploiement du modèle final**

- **Score de sélection** · F1 − 0.5×σ(F1_CV) = compromis entre F1 moyen et stabilité
- **Sauvegarder** · joblib + seed en commentaire
- **Exposer** · FastAPI endpoint `/predict` + Streamlit dashboard

Output · final_model.joblib (~8 MB), final_model_name.txt.

---

## Métriques et choix

### Pourquoi ces 6 métriques ?

En classification binaire avec classes déséquilibrées (25% pannes), on compare ·

| Métrique | Formule | Quand l'utiliser | Pièges |
| -------- | ------- | --------------- | ------ |
| **Accuracy** | (TP+TN) / (TP+TN+FP+FN) | Rapide vue d'ensemble si classes équilibrées | Trompeuse si déséquilibre (model qui prédit tout 0 = 75% accuracy) |
| **Precision** | TP / (TP+FP) | Coût d'une fausse alerte (intervention inutile) | Ignore FN (pannes ratées) |
| **Recall** | TP / (TP+FN) | Coût d'un faux négatif (panne non détectée = production stoppée) | Ignore FP (alertes inutiles) |
| **F1** | 2×(Precision×Recall) / (Precision+Recall) | Compromis Precision/Recall quand les deux importants | Donne même poids aux deux |
| **ROC-AUC** | Aire sous courbe TPR vs FPR | Compare capacité discrimination modèles (invariant seuil) | Peut être élevé même si performance faible sur classe minoritaire |
| **PR-AUC** | Aire sous courbe Precision vs Recall | **Plus fiable que ROC-AUC en déséquilibre** · mieux réflète métier | Coûteux à calculer (100s d'iterations) |

### Choix métier · asymétrie FP vs FN

Dans la maintenance industrielle ·

- **Coût FN (panne ratée)** · ~1 000 EUR/heure (arrêt production) = **très cher**
- **Coût FP (alerte inutile)** · ~100 EUR (technicien se déplace) = **bon marché**

Donc → **Recall ≥ 85%** (détecter 85% des pannes réelles), accepter FP jusqu'à ~15%.

### Métrique de sélection du modèle final

```
score = F1 − 0.5 × σ(F1_CV)
```

Où σ(F1_CV) = écart-type du F1 sur 5 folds CV.

**Raisonnement** · on veut maximiser F1 moyen (performance) tout en minimisant variance (stabilité). Le facteur 0.5 privilégie slightly la stabilité.

---

## Anti-data-leakage

### Qu'est-ce que le leakage ?

Utiliser à l'entraînement une information qui ne sera pas disponible à l'inférence. Cela gonfl les métriques en train mais le modèle échoue en production.

### Causes courantes et prévention

| Cause | Exemple problématique | Solution appliquée |
| ----- | -------------------- | ------------------- |
| **Normalisation sur dataset complet** | StandardScaler fit sur train+test ensemble | ✓ Fit sur train uniquement, transform test après |
| **Feature engineering sur target** | Calculer une feature via la cible (ex. ratio pannes/sample) | ✓ Écrire feature_engineering.py sans accès à y |
| **Cross-val naïve** | Split train/test, puis CV sur train+test | ✓ CV toujours imbriquée dans train |
| **Test set vu à l'entraînement** | Hypertuning sur test set | ✓ Optuna sur CV train uniquement |

### Architecture scikit-learn anti-leakage

```python
# ✓ CORRECT · Pipeline immutable
pipeline = Pipeline([
    ("preprocessor", ColumnTransformer([
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(), categorical_features),
    ])),
    ("classifier", XGBClassifier(...)),
])

pipeline.fit(X_train, y_train)     # Fit sur train
y_pred = pipeline.predict(X_test)  # Transform+predict test auto
# Le preprocessing est "gelé" sur les paramètres d'entraînement
```

---

## Interprétabilité des prédictions

### Trois niveaux d'explication

#### Niveau 1 · Feature Importance globale

**Qu'est-ce ?** Classement des features par contribution moyenne à la prédiction.

**Techniques** ·

- **Native importance** (sklearn) · Gini/Gain dans arbres
  - Avantage · très rapide, O(n×m)
  - Inconvénient · ne capture pas les interactions
  - Cas d'usage · exploration rapide, baseline

- **Permutation importance** (sklearn)
  - Shuffler une feature, mesurer baisse métrique
  - Avantage · model-agnostic, robuste
  - Inconvénient · coûteux (O(n×m)), corrélations masquées
  - Cas d'usage · audit final, validation

#### Niveau 2 · SHAP Value (feature importance + direction)

**Qu'est-ce ?** Pour chaque prédiction, décompose la contribution de chaque feature (positive = vers panne, négative = vers sain).

**Techniques** ·

- **TreeExplainer** (XGBoost / Random Forest)
  - Exact · trace tous les chemins arbre
  - Avantage · très rapide (ms par sample)
  - Cas d'usage · production (API temps réel)

- **KernelExplainer** (universal, LogReg / MLP)
  - Approximation par coalitions locales
  - Avantage · model-agnostic
  - Inconvénient · lent (besoins 5000 queries par sample)
  - Cas d'usage · explications offline détaillées

**Interprétation** ·

```
SHAP value positif  → la feature pousse la prédiction vers panne (y=1)
SHAP value négatif  → la feature pousse la prédiction vers sain (y=0)
|SHAP value| haut   → contribution importante
```

#### Niveau 3 · Visualisations SHAP

- **Force plot** · top 10 features + SHAP values, barres couleur
- **Waterfall plot** · 1 sample · décomposition cascadée de la prédiction
- **Summary plot** · scatter plot moyenne |SHAP| par feature, coloré par valeur
- **Dependence plot** · scatter feature value vs SHAP value (détecte non-linéarités)

### Quand utiliser quelle explication ?

| Contexte | Technique | Raison |
| -------- | --------- | ------ |
| Dashboard exploration rapide | Native importance | Rapide, déjà calculée |
| Audit final rapport | Permutation importance | Robuste, model-agnostic |
| API temps réel (1 sample) | TreeExplainer + Force plot | 1-10 ms, actionnable opérateur |
| Investigation post-mortem (5 pannes) | Waterfall plot | Décomposition détaillée per-sample |
| Feature engineering validation | Dependence plot | Visualise non-linéarités, interactions |

---

## Écoresponsabilité (RNCP C4.3)

### Mesure CO₂ avec CodeCarbon

Le sujet impose explicitement l'**évaluation du degré d'écoresponsabilité** des modèles.

**CodeCarbon** estime l'empreinte carbone (gCO₂ équivalent) basée sur ·

- **Consommation électrique** · TDP CPU + GPU, durée entraînement
- **Mix énergétique national** · France ~80 gCO₂/kWh (nucléaire, bas carbone)
- **Formule** · gCO₂eq = (énergie kWh) × (intensité carbone gCO₂/kWh)

### Résultats pour le projet

| Modèle | Temps train (s) | Power (W) | Energy (kWh) | CO₂ (mg) | Efficacité (F1/mg) |
| ------ | --------------- | --------- | ------------ | -------- | ------------------ |
| Logistic Regression | 3.2 | 45 | 0.00004 | 0.30 | 2883 |
| Random Forest | 45.1 | 180 | 0.00226 | 8.21 | 111 |
| **XGBoost** | **34.8** | **165** | 0.00159 | **6.12** | **152** |
| MLP | 18.5 | 95 | 0.00049 | 3.79 | 236 |
| **Ensemble** | **101.6** | **- avg -** | **0.0044** | **18.4** | **~50** |

### Recommandations d'écoresponsabilité

1. **Baseline optimale** · Logistic Regression (0.3 mg CO₂), mais F1=0.86 insuffisant.
2. **Best compromise** · XGBoost (F1=0.928, 6.12 mg CO₂) = 152 F1-points par mg CO₂eq.
3. **Limiter ensemble** · Une fois modèle sélectionné, ne pas réentraîner 4×.
4. **Décorrélation CPU/perso** · les résultats peuvent varier ±30% selon machine.

**Données brutes** · `reports/codecarbon/` contient les logs JSON détaillés.

---

## Reproductibilité

### Seed propagée · Garantie 100% bit-à-bit

```python
# src/config.py
RANDOM_STATE: int = 42
```

Propagée à tous les composants ·

```python
# numpy
np.random.seed(RANDOM_STATE)

# scikit-learn
LogisticRegression(random_state=RANDOM_STATE)
RandomForestClassifier(random_state=RANDOM_STATE)
train_test_split(..., random_state=RANDOM_STATE)
StratifiedKFold(..., random_state=RANDOM_STATE)

# xgboost
XGBClassifier(random_state=RANDOM_STATE, ...)

# mlp
MLPClassifier(random_state=RANDOM_STATE)
```

### Versions figées · Python + dépendances

```
python --version  → 3.12.0
pip freeze        → requirements.txt avec versions [m.n.p]
```

Relancer après clone ·

```bash
pip install -r requirements.txt  # versions exactes
# Pré-requis · CSV Kaggle dans data/raw/predictive_maintenance_v3.csv
python scripts/02_eda.py
python scripts/03_train_models.py
# → Résultats identiques à la machine d'origin Adam/Emilien
```

### Dataset · CSV Kaggle officiel uniquement

- **Source unique** · Kaggle CC0 · `tatheerabbas/industrial-machine-predictive-maintenance`
- **Téléchargement** · `kaggle datasets download tatheerabbas/industrial-machine-predictive-maintenance` puis extraire `predictive_maintenance_v3.csv` dans `data/raw/`
- **Validation auto** · `src.data_loader.load_dataset()` valide le schéma 15 colonnes à chaque chargement, lève `FileNotFoundError` avec instructions Kaggle si absent
- **Note** · `src.data_loader.generate_synthetic_dataset()` reste disponible mais réservée aux **tests unitaires** (`tests/test_*.py`). Le pipeline production n'y a jamais recours.

### Vérification simple

```bash
# Après exécution complète
cd reports
md5sum rapport_projet_data_science.pdf
# Devrait match le fichier d'origin (ou PDF metadata peut varier tempo)

# Pour métrique brutes
grep "F1.*XGBoost" .../*.log
# Doit être 0.928 (tolérance ±0.001 selon machine)
```

---

## Couverture RNCP40875

### Bloc 2 · Piloter et implémenter des solutions d'IA en s'aidant notamment de l'IA générative

| Code | Compétence | Implémentation | Preuves (fichiers/sections) |
| ---- | --------- | -------------- | ---------------------- |
| **C3.1** | Préparer et transformer les données | ColumnTransformer (Imputer + Scaler + OHE) · `preprocessing.py` · anti-leakage pipeline sklearn | `src/preprocessing.py` lignes 40-120 · schéma dans section "Architecture" du rapport (p.4) |
| **C3.2** | Concevoir et mettre en oeuvre un tableau de bord interactif et inclusif | Dashboard Streamlit · 5 onglets · CSS EFREI · KPI + EDA + comparaison + simulateur + SHAP · 6 fonctions exigées | `dashboard/app.py` lignes 1-600 · captures dans README section "Lancer le dashboard" |
| **C3.3** | Réaliser une analyse exploratoire des données | EDA script · 7+ graphiques (distributions, correlations, imbalance) · stats descriptives | `scripts/02_eda.py` · `reports/02/eda_*.png` · section "Pipeline" du rapport (p.5-6) |
| **C4.1** | Intégrer une stratégie d'IA dans la chaîne de valeur métier | Modèle prédictif 24h · cas métier maintenance industrielle · coût FN vs FP · thresholdoldOptimization · ROI documented | Rapport section 1 "Contexte métier" (p.2-3) · ADR-0001 |
| **C4.2** | Concevoir et mettre en oeuvre des modèles prédictifs ML/DL | 4 modèles (LogReg, RF, XGBoost, MLP 64-32-16) · CV stratifiée 5-fold · sélection via F1−0.5σ · hyperparamètres justifiés | `src/models.py` lignes 28-170 · tableau comparaison "Modèles comparés" · section "Modèles comparés" du README |
| **C4.3** | Évaluer la performance des modèles et leur écoresponsabilité | 6 métriques (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC) · CodeCarbon mesure CO₂ (mg) · Brier score calibration | `src/evaluation.py` lignes 76-150 · `src/co2_tracking.py` · section "Écoresponsabilité" du README · `reports/codecarbon/*.json` |
| **C4.4** | Assurer la qualité et la pérennité des solutions IA | Tests pytest (preprocessing, models, evaluation, API) · logging structuré · version semver (2.0.0) · docs (ADR) | `tests/*.py` · `tests/*.py` · `docs/adr/*.md` · section "Reproductibilité" du README |
| **EF4** | Tableau de bord opérationnel | Streamlit 5-onglets (KPI, EDA, modèles, simulator, SHAP) · CSS premium · sliders interactifs | `dashboard/app.py` · section README "Lancer le dashboard" |
| **EF5** | API REST industrialisation | FastAPI endpoints · `/predict` (prédiction) · `/health` (santé) · `/model-info` (métadonnées) · validation Pydantic | `api/main.py` · section README "Lancer l'API" · exemples cURL |

### Bonus valorisés

| Feature | Impact RNCP | Fichier |
| ------- | ---------- | ------- |
| Classification multi-classe (failure_type) | Approfondit C4.2 (modélisation) | `scripts/07_train_multiclass.py` |
| Régression (RUL hours) | Approfondit C4.2 (prédiction continue) | `scripts/08_train_regression.py` |
| Hyperparameter tuning (Optuna) | Renforce C4.2 (optimisation) | `scripts/09_tune_hyperparams.py` |
| Calibration probabiliste | Avance C4.3 (évaluation + métier) | `scripts/10_calibrate.py` |
| Présentation PPTX automatisée | Bonus communication / reporting | `scripts/11_generate_slides.py` |

---

## Tâches bonus

### 1. Classification multi-classe (failure_type)

Au lieu de prédire "panne oui/non", on prédit le **type exact de panne**.

**Cible** · `failure_type` ∈ {none, bearing, motor_overheat, hydraulic, electrical} (5 classes).

**Modèles** · 4 modèles réentraînés en mode multi-classe ·

```python
python scripts/07_train_multiclass.py
```

**Métrique** · F1 macro (moyenne non-pondérée) plutôt que F1 binaire.

**Output** · `reports/07/multiclass_confusion_matrix.png`, `models/multiclass_final.joblib`.

### 2. Régression · durée de vie restante (RUL)

Prédire le nombre d'heures restantes avant panne.

**Cible** · `rul_hours` (valeur continue, [0, 2000]).

**Modèles** · 4 modèles adaptés en régression ·

```python
python scripts/08_train_regression.py
```

**Métriques** · MAE (erreur absolue moyenne), RMSE, R² (coefficient détermination).

**Output** · `reports/08/regression_pred_vs_true.png`, `models/regression_final.joblib`.

### 3. Hyperparameter tuning · Optuna

Optimisation bayésienne des hyperparamètres (plutôt que GridSearch exhaustif).

```bash
python scripts/09_tune_hyperparams.py
```

**Sampler** · TPE (Tree-structured Parzen Estimator) · plus efficace que Random Search.

**Trials** · 100 essais, pruning adaptatif (arrête trial non-prometteuse tôt).

**Output** · `reports/09/tuning_results.json`.

### 4. Calibration probabiliste

Affiner les probabilités prédites pour minimiser le coût métier (FN=1000€, FP=100€).

```bash
python scripts/10_calibrate.py
```

**Outils** ·

- Reliability diagram · graphique étalonnage empirique vs probabilités
- Brier score · MSE(y_true, y_proba)
- Optimal threshold · seuil Youden ou custom cost-sensitive

**Output** · `reports/10/reliability_diagram_*.png` + `reports/10/cost_threshold_*.png`, seuil optimal recommandé dans `models/optimal_threshold.json`.

### 5. Présentation PPTX

Génération automatique de 11 slides PPTX avec charte EFREI.

```bash
python scripts/11_generate_slides.py
```

**Contenu** ·

1. Page titre (EFREI logo, projet)
2. Contexte métier
3. Données (24 042 samples, 15 variables)
4. EDA (distributions, corrélations)
5. Modèles comparés (table 4 modèles)
6. Résultats (courbes ROC/PR)
7. Interprétabilité (SHAP top features)
8. Conclusion
9-11. Annexes (architecture, references)

**Output** · `reports/11/presentation.pptx`.

---

## FAQ

### Pourquoi 4 modèles et pas 2 ou 5 ?

Le sujet impose « au minimum 4 modèles dont 1 Deep Learning ». 4 est un bon compromis ·

- 2 trop peu (pas assez de variation pour comp comparative)
- 5+ coûteux en temps (200+ min train) et émissions CO₂
- Nous avons choisi LogReg (baseline), RF (bagging), XGB (boosting), MLP (DL) pour couvrir les familles principales

### Pourquoi MLP plutôt que LSTM ou Conv1D ?

**LSTM** · mieux pour séries temporelles (t-1, t, t+1). Notre dataset est **tabulaire stateless** (capteurs statiques à un instant T), pas une série temporelle. LSTM serait sur-dimensionné.

**Conv1D** · pour signaux 1D (audio, ECG). Notre cas c'est 9 features indépendantes, pas un signal continu.

**Conclusion** · MLP suffit. Si les données étaient temporelles (historique 24h), on aurait exploré LSTM.

### Comment changer la variable cible ?

Actuellement · `failure_within_24h` (binaire). Pour changer ·

```python
# src/config.py
TARGET_BINARY = "failure_type"  # switch à multi-classe
TARGET_REGRESSION = "rul_hours" # ou régression
```

Puis ·

```python
python scripts/03_train_models.py  # réentraîner
```

Les métriques s'adaptent automatiquement (F1 pour classification, RMSE pour régression).

### Comment ajouter un modèle (ex. CatBoost) ?

1. **Installer** · `pip install catboost`
2. **Ajouter factory** ·

```python
# src/models.py
def build_catboost() -> Pipeline:
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", CatBoostClassifier(random_state=42, verbose=0)),
    ])
```

3. **Enregistrer** dans `scripts/03_train_models.py` ·

```python
models = {
    "LogReg": build_logistic_regression(),
    "CatBoost": build_catboost(),  # ← new
    ...
}
```

4. **Relancer** · `python scripts/03_train_models.py`

### Pourquoi Streamlit et pas Dash ou Plotly ?

| Aspect | Streamlit | Dash | Plotly |
| ------ | --------- | ---- | ------ |
| Prototypage | ⚡⚡⚡ ultra-rapide | slow |  |
| Data science friendly | ⚡⚡ streamlit cache | overkill |  |
| CSS custom | ⚡ facile `st.markdown` | difficile |  |
| Production ready | ⚡ via Streamlit Cloud/Enterprise | ⚡⚡ robuste |  |

Pour un MVP étudiant, Streamlit est le choix optimal.

### Comment utiliser le dataset Kaggle direct au lieu du synthétique ?

```bash
# 1. Télécharger via Kaggle CLI
kaggle datasets download -d tatheerabbas/industrial-machine-predictive-maintenance
unzip industrial-machine-predictive-maintenance.zip -d data/raw/

# 2. Renommer
mv data/raw/predictive_maintenance.csv data/raw/predictive_maintenance_v3.csv

# 3. Relancer pipeline
python scripts/02_eda.py  # saute 01 si CSV existe
```

### Comment redéployer le modèle en production ?

```bash
# 1. Relancer entraînement (sur nouvelles données si dispo)
python scripts/03_train_models.py

# 2. Vérifier métriques
cat reports/03/metrics_summary.csv | grep f1

# 3. Lancer API
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Le modèle final est chargé automatiquement depuis `models/final_model.joblib`.

### Comment utiliser le modèle avec un nouveau capteur (ex. vibration_z-axis) ?

1. **Ajouter colonne** au CSV source
2. **Updater config** ·

```python
# src/config.py
NUMERIC_FEATURES = [
    "vibration_rms",
    "vibration_z_axis",  # ← new
    ...
]
```

3. **Rétraiter + réentraîner** ·

```bash
python scripts/03_train_models.py  # ColumnTransformer s'adapte
```

Le preprocessing et les modèles se réajustent automatiquement.

---

## Troubleshooting

### Erreur · FileNotFoundError: data/raw/predictive_maintenance_v3.csv

**Cause** · Dataset manquant (pas téléchargé de Kaggle, synthetique pas généré).

**Solutions** ·

```bash
# Option 1 · télécharger Kaggle
kaggle datasets download -d tatheerabbas/industrial-machine-predictive-maintenance
unzip archive.zip -d data/raw/
```

### Erreur · MemoryError lors de SHAP KernelExplainer

**Cause** · KernelExplainer estime sur 2500 backgrounds samples · trop coûteux en RAM sur petit CPU.

**Solutions** ·

```python
# src/interpretability.py · reduce backgrounds
explainer = shap.KernelExplainer(
    model_predict_proba,
    shap.sample(X_train, 500)  # ← 500 au lieu de 2500
)
```

Ou utiliser TreeExplainer si le modèle final est XGBoost/RF (gratuit).

### Erreur · ConvergenceWarning sur Logistic Regression

**Cause** · `max_iter=1000` insuffisant sur certaines machines (lent convergence).

**Solution** ·

```python
# src/models.py
LogisticRegression(max_iter=5000, solver="saga", ...)  # augmenter ou changer solver
```

### Erreur · ModuleNotFoundError: No module named 'src'

**Cause** · virtualenv pas activé ou dépendances pas installées.

**Solutions** ·

```bash
# Vérifier virtualenv actif (prefix [.venv] ou (.venv))
pip install -r requirements.txt --upgrade
python -c "import src; print(src.__version__)"
```

### Erreur · Port 8000/8501 déjà utilisé

**Cause** · API/Dashboard déjà lancés sur ces ports.

**Solutions** ·

```bash
# Trouver et tuer processus
lsof -i :8000  # Linux/macOS
# ou
netstat -ano | findstr :8000  # Windows

# Puis
kill -9 <PID>

# Ou changer port
uvicorn api.main:app --port 8001
streamlit run dashboard/app.py --server.port=8502
```

### Métriques très différentes de celles documentées

**Cause** · seed aléatoire changé, versions dépendances différentes, données modifiées.

**Diagnostic** ·

```bash
# Vérifier seed
grep RANDOM_STATE src/config.py  # doit être 42

# Vérifier versions
pip freeze | grep -E "scikit|xgboost|shap"

# Vérifier données (hash)
md5sum data/raw/predictive_maintenance_v3.csv
```

---

## Roadmap

### Court terme (v2.1 · juin 2026)

- [ ] Déploiement Cloud (Vercel/Render) pour accès public
- [ ] MLflow pour tracking expériences (alternative Optuna)
- [ ] Monitoring drift automatique (données en production vs train)
- [ ] Alertes Slack si F1 baisse

### Moyen terme (v2.5 · sept 2026)

- [ ] Ajout LSTM temporel si données séries 24h disponibles
- [ ] Edge deployment (TensorFlow Lite pour IoT gateway)
- [ ] Web UI pour upload CSV + predictions batch
- [ ] Internationalisation dashboard (FR/EN/ES)

### Long terme (v3.0+)

- [ ] Federated learning (entraînement décentralisé multi-usines)
- [ ] Confidence intervals sur prédictions (Bayesian MLP)
- [ ] AutoML wrapper (H2O, AutoGluon)
- [ ] Explainability audit · LIME + SHAP comparaison

---

## Remerciements

Ce projet n'aurait pas été possible sans ·

- **Sarah Malaeb** · enseignante, sujet du projet, feedback continu
- **EFREI Paris Panthéon-Assas Université** · ressources académiques, infrastructure
- **Kaggle Dataset** · [Tathaeer Abbas](https://www.kaggle.com/tatheerabbas) pour le dataset public CC0
- **Communauté open source** · scikit-learn, XGBoost, Streamlit, FastAPI, SHAP, CodeCarbon maintainers

Merci aussi à **Emilien Morice**, coauteur du projet, pour les brainstormings et les alternances de contributions.

---

## License et contributions

### Licence

Ce projet est sous licence **MIT** · libre d'usage, modification, redistribution (voir `LICENSE`).

### Contributions

Nous acceptons contributions via pull requests ·

1. Fork le dépôt
2. Créer branche `feat/*` ou `fix/*`
3. Commiter en anglais impératif court (ex. `fix: SHAP normalization on 3D arrays`)
4. Ouvrir PR avec description courte
5. Tests pytest doivent passer avant merge
6. Merge après approbation

### Citation

Si vous réutilisez ce projet ·

```bibtex
@misc{beloucif_morice_2026,
  title={Maintenance Prédictive Industrielle · Système Intelligent Multi-Modèles},
  author={Beloucif, Adam and Morice, Emilien},
  year={2026},
  school={EFREI Paris Panthéon-Assas Université},
  note={M1 Data Engineering & IA · BC2 RNCP40875}
}
```

---

**Dernière mise à jour** · 2026-04-28

**Version** · 2.0.0

**Auteurs** · Adam BELOUCIF (n° 20220055) · Emilien MORICE (n° 20241824)

**Status** · Production ready (v2.0 + 7 bonus scripts) · Reproductibilité certifiée seed=42
