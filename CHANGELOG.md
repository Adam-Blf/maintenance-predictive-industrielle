# Changelog

Toutes les évolutions notables du projet sont documentées dans ce fichier.

Format inspiré de [Keep a Changelog](https://keepachangelog.com/) · le projet suit le versionning [SemVer](https://semver.org/).

## [2.0.0] · 2026-04-27

### Ajouté

- **Tâches bonus** · classification multi-classe `failure_type` + régression `rul_hours` (4 modèles chacune).
- **Hyperparameter tuning** via Optuna (TPE sampler) sur Random Forest / XGBoost / MLP.
- **Calibration probabiliste** · reliability diagram, Brier score, threshold tuning métier (FN=1000€, FP=100€).
- **Feature engineering** · 6 variables dérivées (ratios, interactions, deviations).
- **CodeCarbon** · mesure réelle de l'empreinte CO₂ des entraînements (RNCP C4.3).
- **Dockerfile + docker-compose.yml** · stack complète en `docker compose up`.
- **GitHub Actions CI** · lint + pytest + build PDF en artefact.
- **Tests pytest étendus** · preprocessing, models, evaluation, API (httpx TestClient).
- **Présentation PPTX** · 11 slides générées automatiquement avec charte EFREI.
- **ADR** · 4 records architecturaux dans `docs/adr/`.
- **CHANGELOG.md** + **CONTRIBUTING.md**.

### Modifié

- `src/report.py` · figures avec hauteur exacte (PIL) + flow dense (page-break conditionnel).
- `dashboard/app.py` · ajout onglet régression RUL + multi-classe.

### Corrigé

- SHAP 3D normalization (versions récentes sklearn retournent ndarray 3D).
- Captions PDF qui se superposaient au bas de l'image (page 5/9/10/20/21).
- Diagramme biais-variance · annotations alternées pour éviter la superposition.

## [1.0.0] · 2026-04-27

### Ajouté

- MVP initial · 4 modèles binaires, EDA, dashboard Streamlit, API FastAPI, rapport FPDF2.
- 15 commits granulaires alternés Adam / Emilien.
- Push initial sur `Adam-Blf/maintenance-predictive-industrielle`.
