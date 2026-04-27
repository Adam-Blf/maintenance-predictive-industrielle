# Système Intelligent Multi-Modèles · Maintenance Prédictive Industrielle

[![Build](https://img.shields.io/badge/build-passing-brightgreen)](.)
[![Python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/xgboost-2.0-success)](https://xgboost.ai/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.32-red)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/fastapi-0.110-009688)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-yellow)](./LICENSE)
[![Version](https://img.shields.io/badge/version-1.0.0-informational)](./src/__init__.py)

> **Projet Data Science** · M1 Mastère Data Engineering & IA · EFREI Paris Panthéon-Assas Université · Année 2025-2026
> **Bloc** · BC2 RNCP40875 · Piloter et implémenter des solutions d'IA en s'aidant notamment de l'IA générative
> **Auteurs** · Adam BELOUCIF · Emilien MORICE

---

## Sommaire

1. [Contexte](#contexte)
2. [Stack technique](#stack-technique)
3. [Architecture](#architecture)
4. [Installation](#installation)
5. [Pipeline d'exécution](#pipeline-dexécution)
6. [Lancer le dashboard](#lancer-le-dashboard)
7. [Lancer l'API](#lancer-lapi)
8. [Structure du dépôt](#structure-du-dépôt)
9. [Modèles comparés](#modèles-comparés)
10. [Couverture RNCP40875](#couverture-rncp40875)
11. [Auteurs](#auteurs)

---

## Contexte

Ce projet livre une plateforme intelligente de maintenance prédictive industrielle qui exploite des données de capteurs (vibration, température, pression, RPM) pour anticiper les pannes machines dans les 24 heures à venir. La cible retenue est **`failure_within_24h`** (classification binaire), conformément au cahier des charges. Le système couvre l'intégralité du cycle de vie · ingestion, EDA, modélisation multi-algorithmes, évaluation comparative, interprétabilité (SHAP), interface décisionnelle, et exposition via API REST.

> Dataset · `industrial_machine_maintenance.csv` · 24 042 enregistrements · 15 variables · cible binaire déséquilibrée.
> Source officielle Kaggle · `tatheerabbas/industrial-machine-predictive-maintenance`.
> Un générateur synthétique reproductible (seed = 42) est embarqué pour garantir la reproductibilité hors connexion.

## Stack technique

| Couche           | Outils                                                   |
| ---------------- | -------------------------------------------------------- |
| Langage          | Python 3.12                                              |
| Data wrangling   | pandas · numpy                                           |
| ML classique     | scikit-learn (Logistic Regression, Random Forest, MLP)   |
| Boosting         | XGBoost                                                  |
| Deep Learning    | MLPClassifier scikit-learn (architecture 64-32-16, ReLU) |
| Interprétabilité | SHAP · `permutation_importance` · `feature_importances_` |
| Dashboard        | Streamlit (CSS personnalisé, charte EFREI)               |
| API              | FastAPI + Pydantic v2 + Uvicorn                          |
| Visualisation    | matplotlib · seaborn · plotly                            |
| Rapport          | FPDF2 (PDF analytique structuré)                         |
| Qualité code     | black · isort · prettier                                 |

## Architecture

```
Capteurs IoT  →  CSV Bronze  →  Pipeline Silver  →  Modèle Gold  →  API REST  →  Dashboard
                                                                         ↓
                                                              Responsable maintenance
```

Schéma détaillé · `reports/figures/diagram_architecture.png` (généré par `scripts/05_generate_diagrams.py`).

## Installation

```bash
# 1. Cloner le dépôt
git clone <url>
cd "Porjet Data Science"

# 2. Créer un environnement virtuel (optionnel mais recommandé)
python -m venv .venv
.venv\Scripts\activate         # Windows
# source .venv/bin/activate    # macOS/Linux

# 3. Installer les dépendances
pip install -r requirements.txt
```

## Pipeline d'exécution

Les scripts sont **strictement séquentiels** · chaque étape produit les artefacts consommés par la suivante.

```bash
# 1. Génération du dataset synthétique reproductible (24 042 lignes)
python scripts/01_generate_dataset.py

# 2. Analyse exploratoire des données (7 graphiques, stats descriptives)
python scripts/02_eda.py

# 3. Entraînement et évaluation comparative des 4 modèles + cross-validation
python scripts/03_train_models.py

# 4. Interprétabilité (Feature Importance + Permutation + SHAP)
python scripts/04_interpret.py

# 5. Génération des schémas pédagogiques pour le rapport
python scripts/05_generate_diagrams.py

# 6. Construction du rapport PDF final (à exécuter en dernier)
python scripts/06_generate_report.py
```

Sortie principale · `reports/rapport_projet_data_science.pdf`.

## Lancer le dashboard

Le dashboard Streamlit est l'**interface décisionnelle obligatoire** du projet (EF4). Il offre 5 onglets · vue d'ensemble (KPI), EDA interactive, comparaison des modèles, simulateur de scénario machine, interprétabilité.

```bash
streamlit run dashboard/app.py
```

Ouvre par défaut · `http://localhost:8501`.

## Lancer l'API

L'API REST FastAPI est l'**option d'industrialisation** (EF5). Elle expose le modèle final candidat sous forme de service consommable.

```bash
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

Endpoints ·

| Méthode | Chemin        | Description                                  |
| ------- | ------------- | -------------------------------------------- |
| POST    | `/predict`    | Prédiction de panne 24h sur scénario machine |
| GET     | `/health`     | Vérification de l'état du service            |
| GET     | `/model-info` | Métadonnées du modèle servi                  |
| GET     | `/docs`       | Swagger UI auto-généré                       |

Exemple cURL ·

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "vibration_rms": 4.2,
    "temperature_motor": 78.5,
    "rpm": 2100,
    "pressure_level": 8.7,
    "ambient_temperature": 24.1,
    "humidity": 58.0,
    "voltage": 402.1,
    "current": 62.3,
    "power_consumption": 21.3,
    "maintenance_age_days": 180,
    "operating_mode": "HighLoad"
  }'
```

## Structure du dépôt

```
.
├── api/
│   └── main.py                   # API FastAPI (3 endpoints)
├── assets/
│   └── logo_efrei.png            # Logo EFREI (page de garde + dashboard)
├── dashboard/
│   └── app.py                    # Dashboard Streamlit (5 onglets, CSS custom)
├── data/
│   ├── raw/                      # Dataset brut (gitignored)
│   └── processed/                # Splits train/test (gitignored)
├── docs/                         # Documentation projet
├── models/                       # Pipelines sérialisés (gitignored)
├── reports/
│   ├── figures/                  # PNG générés (EDA, courbes, schémas)
│   └── rapport_projet_data_science.pdf
├── scripts/
│   ├── 01_generate_dataset.py
│   ├── 02_eda.py
│   ├── 03_train_models.py
│   ├── 04_interpret.py
│   ├── 05_generate_diagrams.py
│   └── 06_generate_report.py
├── src/
│   ├── __init__.py
│   ├── config.py                 # Chemins + hyperparamètres centralisés
│   ├── data_loader.py            # Loader + générateur synthétique
│   ├── preprocessing.py          # ColumnTransformer (Imputer + Scaler + OHE)
│   ├── models.py                 # 4 factories ML/DL
│   ├── evaluation.py             # 6 métriques + courbes + barplots
│   ├── interpretability.py       # Feature importance + SHAP
│   ├── diagrams.py               # 4 schémas pédagogiques (matplotlib)
│   └── report.py                 # Générateur FPDF2 du rapport
├── tests/                        # Tests de fumée
├── .gitignore
├── LICENSE
├── README.md                     # ← vous êtes ici
└── requirements.txt
```

## Modèles comparés

Le sujet impose **au minimum 4 modèles dont 1 Deep Learning**. Nous comparons ·

| #   | Modèle              | Famille           | Justification                                                             |
| --- | ------------------- | ----------------- | ------------------------------------------------------------------------- |
| 1   | Logistic Regression | Linéaire          | Baseline interprétable · point de référence pour la comparaison.          |
| 2   | Random Forest       | Bagging d'arbres  | Capture les non-linéarités, robuste aux outliers, importance native.      |
| 3   | XGBoost             | Boosting d'arbres | État de l'art sur tabulaire, gère le déséquilibre via `scale_pos_weight`. |
| 4   | MLP (64-32-16)      | **Deep Learning** | Réseau de neurones imposé par le sujet, early stopping anti-overfit.      |

Métriques d'évaluation · Accuracy · Precision · Recall · F1 · ROC-AUC · PR-AUC · temps d'entraînement · latence d'inférence (cf. ecoresponsabilité C4.3).

Sélection du modèle final · `F1 - 0.5 × σ(F1_CV)` (compromis performance/stabilité).

## Couverture RNCP40875

| Code | Compétence                            | Implémentation                                                         |
| ---- | ------------------------------------- | ---------------------------------------------------------------------- |
| C3.1 | Préparation des données               | `src/preprocessing.py` · ColumnTransformer (imputer + scaler + OHE)    |
| C3.2 | Tableau de bord interactif inclusif   | `dashboard/app.py` · Streamlit, CSS custom, 5 onglets, simulateur, KPI |
| C3.3 | Analyse exploratoire                  | `scripts/02_eda.py` · 7 graphiques + stats descriptives                |
| C4.1 | Stratégie d'intégration IA            | Architecture API/Dashboard/Modèle (rapport section 3.1)                |
| C4.2 | Modèles prédictifs ML/DL              | `src/models.py` · 4 modèles dont MLP (DL), pipelines reproductibles    |
| C4.3 | Évaluation comparative + écoresponsa. | `src/evaluation.py` · 6 métriques + temps de calcul + interprétabilité |

## Auteurs

| Nom            | Rôle                | Email                    |
| -------------- | ------------------- | ------------------------ |
| Adam BELOUCIF  | Data Engineer & Dev | adam.beloucif@efrei.net  |
| Emilien MORICE | Data Engineer & Dev | emilien.morice@efrei.net |

École · **EFREI Paris Panthéon-Assas Université** · 30-32 avenue de la République, 94800 Villejuif · [www.efrei.fr](https://www.efrei.fr)

---

> Tous les schémas, figures et métriques inclus dans le rapport `reports/rapport_projet_data_science.pdf` sont reproductibles bit-à-bit en exécutant la pipeline dans l'ordre · seed `42` propagée à numpy et scikit-learn.
