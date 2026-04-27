# Contribuer au projet

Merci de votre intérêt pour le projet Maintenance Prédictive Industrielle. Ce document décrit les conventions à respecter pour toute contribution.

## Setup

```bash
git clone https://github.com/Adam-Blf/maintenance-predictive-industrielle.git
cd maintenance-predictive-industrielle
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
pip install pytest pytest-cov ruff black isort
```

## Pipeline complète

```bash
python scripts/01_generate_dataset.py
python scripts/02_eda.py
python scripts/05_generate_diagrams.py
python scripts/03_train_models.py
python scripts/07_train_multiclass.py
python scripts/08_train_regression.py
python scripts/09_tune_hyperparams.py
python scripts/04_interpret.py
python scripts/10_calibrate.py
python scripts/06_generate_report.py
python scripts/11_generate_slides.py
```

Ou en un seul docker compose ·

```bash
docker compose up
```

## Conventions

- **Code Python** · formaté `black -l 100` + `isort --profile black`.
- **Markdown / JSON** · formaté `prettier --write`.
- **Commits** · anglais impératif court, format `<type>(<scope>): <subject>` (ex. `feat(api): add /predict-rul endpoint`).
- **Branches** · `feat/*`, `fix/*`, `docs/*`.
- **Pull requests** · une feature par PR, tests passants.
- **Aucune mention** d'outil de génération automatique dans les commits.

## Tests

```bash
pytest tests/ --cov=src --cov-report=term -v
```

Cible · couverture > 70 % sur `src/`.

## ADR

Toute décision architecturale non triviale doit faire l'objet d'un Architecture Decision Record dans `docs/adr/NNNN-titre.md`. Voir les ADR existants pour le format.
