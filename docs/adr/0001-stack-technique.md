# ADR 0001 · Choix de la stack technique

- **Statut** · accepté
- **Date** · 2026-04-27
- **Auteurs** · Adam Beloucif, Emilien Morice

## Contexte

Le sujet du projet impose un MVP complet (Data Science + Dashboard + API + Rapport) avec au moins 4 modèles dont 1 Deep Learning. Il faut une stack qui couvre ces besoins sans sur-engineering pour un livrable étudiant reproductible sur Windows / macOS / Linux.

## Décision

| Couche             | Outil retenu                       | Justification                                                                  |
| ------------------ | ---------------------------------- | ------------------------------------------------------------------------------ |
| Langage            | Python 3.12                        | Standard du domaine, large écosystème ML, requis par EFREI                     |
| ML classique       | scikit-learn 1.4+                  | Pipeline + ColumnTransformer = anti data-leakage natif                         |
| Boosting           | XGBoost 2.x                        | État de l'art tabulaire, gestion native du déséquilibre                        |
| Deep Learning      | scikit-learn `MLPClassifier`       | Évite la dépendance lourde TF/PyTorch, suffit pour le scope du sujet           |
| Hyperparam tuning  | Optuna 3.x                         | TPE plus efficace que GridSearch exhaustif, log structuré des trials           |
| Interprétabilité   | SHAP 0.44+                         | Feature importance + permutation natives sklearn, SHAP avancé                  |
| Dashboard          | Streamlit 1.32                     | Vitesse de prototypage > Dash, écosystème data-friendly                        |
| API                | FastAPI 0.110 + Pydantic v2        | Validation automatique, Swagger gratuit, async ready                           |
| Visualisation      | matplotlib + seaborn + plotly      | matplotlib pour figures statiques rapport, plotly pour dashboard interactif    |
| Génération PDF     | FPDF2                              | Plus simple que ReportLab, supporte UTF-8 et images natives                    |
| Présentation       | python-pptx                        | Génération automatique des slides depuis les artefacts                         |
| Tracking CO2       | CodeCarbon                         | Mesure réelle de l'empreinte carbone (RNCP C4.3 littéral)                      |
| Tests              | pytest + httpx TestClient          | Standard Python, intégration FastAPI native                                    |
| Conteneurisation   | Docker + docker-compose            | Reproductibilité absolue (un seul `docker compose up`)                         |
| CI                 | GitHub Actions                     | Gratuit, intégré à GitHub, exécute lint + tests + build PDF en artefact        |

## Conséquences

**Positives** ·

- Pas de stack hétérogène · uniquement Python 3.12.
- Une seule commande pour lancer · `docker compose up`.
- Reproductibilité bit-à-bit avec seed=42.

**Négatives** ·

- MLP scikit-learn limité (pas de couches conv/recurrent) · acceptable car tâche tabulaire.
- Pas de cluster Spark · acceptable car 24k lignes tiennent en RAM.

## Alternatives écartées

- **PyTorch** pour le DL · sur-dimensionné pour 24k lignes tabulaires.
- **MLflow comme orchestrateur** · CodeCarbon + Optuna + JSON/CSV manuels suffisent.
- **Kubernetes** · le scope MVP n'a pas besoin d'orchestration multi-machines.
