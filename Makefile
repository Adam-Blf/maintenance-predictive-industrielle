# =============================================================================
# Makefile · raccourcis pipeline + déploiement
# Usage · `make <cible>` (ou `mingw32-make` sur Windows avec MSYS).
# =============================================================================

.PHONY: install data eda train multiclass regression tune interpret calibrate diagrams report slides all dashboard api docker test clean

PYTHON ?= python

install:
	$(PYTHON) -m pip install -r requirements.txt

data:
	$(PYTHON) scripts/01_generate_dataset.py

eda:
	$(PYTHON) scripts/02_eda.py

train:
	$(PYTHON) scripts/03_train_models.py

multiclass:
	$(PYTHON) scripts/07_train_multiclass.py

regression:
	$(PYTHON) scripts/08_train_regression.py

tune:
	$(PYTHON) scripts/09_tune_hyperparams.py

interpret:
	$(PYTHON) scripts/04_interpret.py

calibrate:
	$(PYTHON) scripts/10_calibrate.py

diagrams:
	$(PYTHON) scripts/05_generate_diagrams.py

report:
	$(PYTHON) scripts/06_generate_report.py

slides:
	$(PYTHON) scripts/11_generate_slides.py

# Pipeline complète dans l'ordre.
all: data eda diagrams train multiclass regression tune interpret calibrate report slides

dashboard:
	streamlit run dashboard/app.py

api:
	uvicorn api.main:app --reload --host 127.0.0.1 --port 8000

docker:
	docker compose up --build

test:
	pytest tests/ --cov=src --cov-report=term -v

clean:
	rm -rf reports/figures/_pages
	rm -rf .pytest_cache .mypy_cache .ruff_cache __pycache__
	find . -name "__pycache__" -type d -exec rm -rf {} +
