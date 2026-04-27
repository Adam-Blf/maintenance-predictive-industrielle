# =============================================================================
# Image Docker · plateforme de maintenance prédictive industrielle
# Multi-stage minimaliste pour servir Dashboard (Streamlit) + API (FastAPI)
# =============================================================================
FROM python:3.12-slim AS base

# Variables d'environnement utiles
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Dépendances système · libgomp pour XGBoost, fontconfig pour matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        fontconfig \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Layer pip · copie le requirements en premier pour profiter du cache.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie le reste du code (ordre · src + scripts + dashboard + api).
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY dashboard/ ./dashboard/
COPY api/ ./api/
COPY assets/ ./assets/
COPY data/ ./data/
COPY models/ ./models/
COPY reports/ ./reports/

# Port par défaut · Streamlit 8501, FastAPI 8000.
EXPOSE 8000 8501

# Healthcheck pour Docker · ping API /health.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://127.0.0.1:8000/health', timeout=3)" || exit 1

# Par défaut · lancer l'API. Docker-compose surcharge la commande pour
# lancer Streamlit dans un service séparé.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
