# -*- coding: utf-8 -*-
"""Script · entraînement et évaluation comparative des 4 modèles.

Pipeline ·
  1. Chargement du dataset + split stratifié 80/20.
  2. Entraînement séquentiel des 4 modèles (Logistic, RF, XGBoost, MLP).
  3. Cross-validation 5-fold sur chaque modèle (stabilité).
  4. Évaluation sur test set · accuracy, precision, recall, F1, ROC-AUC,
     PR-AUC.
  5. Génération des matrices de confusion + courbes ROC + PR + barplots.
  6. Sauvegarde des artefacts modèles (joblib) dans `models/`.
  7. Sauvegarde du tableau de métriques (CSV + JSON) dans `reports/`.

Le modèle final candidat est celui qui maximise le F1 sur le test set
(compromis Precision/Recall, métrique adaptée au déséquilibre des
classes en maintenance prédictive).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    ALL_FEATURES,
    CV_FOLDS,
    DATA_PROCESSED_DIR,
    MODELS_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    REPORTS_FIGURES_DIR,
    TARGET_BINARY,
    TEST_SIZE,
    ensure_directories,
)
from src.data_loader import load_dataset  # noqa: E402
from src.evaluation import (  # noqa: E402
    compute_classification_metrics,
    plot_confusion_matrix,
    plot_metrics_barplot,
    plot_pr_curves,
    plot_roc_curves,
    plot_training_time_barplot,
)
from src.models import MODEL_CATALOG, get_model  # noqa: E402


def split_data(df: pd.DataFrame):
    """Split stratifié 80/20 + sauvegarde sur disque pour les autres scripts.

    Le `stratify=` est CRUCIAL · sans lui, on risque un fold de test sans
    aucune panne, ce qui rend ROC-AUC indéfini. Le déséquilibre 90/10 vu
    dans l'EDA impose la stratification.
    """
    X = df[ALL_FEATURES].copy()
    y = df[TARGET_BINARY].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # Persistance des splits pour réutilisation par les scripts
    # d'interprétabilité et le générateur de rapport.
    X_train.to_csv(DATA_PROCESSED_DIR / "X_train.csv", index=False)
    X_test.to_csv(DATA_PROCESSED_DIR / "X_test.csv", index=False)
    y_train.to_csv(DATA_PROCESSED_DIR / "y_train.csv", index=False)
    y_test.to_csv(DATA_PROCESSED_DIR / "y_test.csv", index=False)

    return X_train, X_test, y_train, y_test


def train_one_model(
    name: str,
    X_train,
    y_train,
    X_test,
    y_test,
    scale_pos_weight: float,
):
    """Entraîne un modèle, calcule métriques et sauvegarde l'artefact.

    Returns
    -------
    tuple
        (metrics: ClassificationMetrics, y_pred, y_proba, fitted_pipeline)
    """
    print(f"\n[TRAIN] === {name} ===")

    # XGBoost a un argument spécifique pour gérer le déséquilibre · on
    # l'injecte ici plutôt qu'à la création du catalogue.
    if name == "xgboost":
        pipeline = get_model(name, scale_pos_weight=scale_pos_weight)
    else:
        pipeline = get_model(name)

    # Mesure du temps d'entraînement (écoresponsabilité · cf. RNCP C4.3).
    t0 = time.time()
    pipeline.fit(X_train, y_train)
    fit_time = time.time() - t0

    # Mesure de la latence · temps de prédiction par échantillon.
    t0 = time.time()
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)
    predict_time_ms = (time.time() - t0) * 1000.0 / len(X_test)

    metrics = compute_classification_metrics(
        model_name=name,
        y_true=y_test.values,
        y_pred=y_pred,
        y_proba=y_proba,
        fit_time_s=fit_time,
        predict_time_ms=predict_time_ms,
    )

    print(f"  Fit · {fit_time:.2f}s · Predict · {predict_time_ms:.4f}ms/sample")
    print(
        f"  Acc={metrics.accuracy:.4f}  P={metrics.precision:.4f}  "
        f"R={metrics.recall:.4f}  F1={metrics.f1:.4f}  "
        f"ROC={metrics.roc_auc:.4f}  PR={metrics.pr_auc:.4f}"
    )

    # Persistance modèle entraîné · joblib > pickle pour les pipelines
    # sklearn (compression intégrée + perfs).
    model_path = MODELS_DIR / f"{name}.joblib"
    joblib.dump(pipeline, model_path, compress=3)
    print(f"  Saved · {model_path}")

    # Matrice de confusion individuelle.
    plot_confusion_matrix(y_test.values, y_pred, name)

    return metrics, y_pred, y_proba, pipeline


def cross_validate_all(X, y) -> dict[str, dict]:
    """Cross-validation 5-fold sur chaque modèle · validation de stabilité.

    Le sujet recommande explicitement la cross-validation pour montrer
    que le modèle généralise et n'est pas performant uniquement sur un
    découpage favorable.
    """
    print("\n[CV] Cross-validation 5-fold (F1) sur les 4 modèles...")
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    cv_results = {}
    # Calcul du scale_pos_weight pour XGB (sur le dataset complet, OK car
    # on cherche un ordre de grandeur).
    scale_pos_weight = float((y == 0).sum() / max((y == 1).sum(), 1))

    for name in MODEL_CATALOG:
        if name == "xgboost":
            model = get_model(name, scale_pos_weight=scale_pos_weight)
        else:
            model = get_model(name)

        scores = cross_val_score(model, X, y, scoring="f1", cv=skf, n_jobs=-1)
        cv_results[name] = {
            "f1_mean": float(scores.mean()),
            "f1_std": float(scores.std()),
            "f1_folds": scores.tolist(),
        }
        print(f"  {name:<22} · F1 = {scores.mean():.4f} ± {scores.std():.4f}")

    return cv_results


def main() -> None:
    """Point d'entrée · pipeline complet d'entraînement."""
    ensure_directories()

    # ------------------------------------------------------------------
    # Étape 1 · Chargement et split.
    # ------------------------------------------------------------------
    print("[1/5] Chargement du dataset...")
    df = load_dataset()
    print(f"  Shape · {df.shape}")
    print(f"  Pannes 24h · {df[TARGET_BINARY].mean():.2%}")

    print("[2/5] Split stratifié 80/20...")
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"  Train · {X_train.shape}, Test · {X_test.shape}")

    # Calcul du ratio neg/pos pour XGBoost.
    scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))

    # ------------------------------------------------------------------
    # Étape 2 · Entraînement séquentiel des 4 modèles.
    # ------------------------------------------------------------------
    print("\n[3/5] Entraînement des 4 modèles...")
    all_metrics = []
    roc_payload = {}
    fitted_pipelines = {}

    for name in MODEL_CATALOG:
        metrics, y_pred, y_proba, pipeline = train_one_model(
            name, X_train, y_train, X_test, y_test, scale_pos_weight
        )
        all_metrics.append(metrics.to_dict())
        roc_payload[name] = {"y_true": y_test.values, "y_proba": y_proba}
        fitted_pipelines[name] = pipeline

    # DataFrame récapitulatif des métriques.
    metrics_df = pd.DataFrame(all_metrics)

    # ------------------------------------------------------------------
    # Étape 3 · Cross-validation pour stabilité.
    # ------------------------------------------------------------------
    print("\n[4/5] Cross-validation 5-fold...")
    # On utilise un sous-échantillon pour rester rapide · CV sur 24k
    # avec MLP coûte cher.
    cv_sample_size = min(8_000, len(X_train))
    sample_idx = np.random.RandomState(RANDOM_STATE).choice(
        len(X_train), size=cv_sample_size, replace=False
    )
    cv_results = cross_validate_all(X_train.iloc[sample_idx], y_train.iloc[sample_idx])
    metrics_df["cv_f1_mean"] = metrics_df["model_name"].map(lambda n: cv_results[n]["f1_mean"])
    metrics_df["cv_f1_std"] = metrics_df["model_name"].map(lambda n: cv_results[n]["f1_std"])

    # ------------------------------------------------------------------
    # Étape 4 · Génération des graphiques de comparaison + sauvegarde.
    # ------------------------------------------------------------------
    print("\n[5/5] Graphiques comparatifs + persistance...")
    plot_roc_curves(roc_payload)
    plot_pr_curves(roc_payload)
    plot_metrics_barplot(metrics_df)
    plot_training_time_barplot(metrics_df)

    # Persistance · CSV pour Excel + JSON pour Streamlit/API.
    metrics_df.to_csv(REPORTS_DIR / "metrics_summary.csv", index=False)
    metrics_df.to_json(REPORTS_DIR / "metrics_summary.json", orient="records", indent=2)
    with open(REPORTS_DIR / "cv_results.json", "w", encoding="utf-8") as f:
        json.dump(cv_results, f, indent=2)

    # ------------------------------------------------------------------
    # Sélection du modèle final · F1 le plus élevé sur le test set.
    # On pondère légèrement par la stabilité CV (F1 - 0.5 × std) pour
    # privilégier les modèles robustes.
    # ------------------------------------------------------------------
    metrics_df["selection_score"] = metrics_df["f1"] - 0.5 * metrics_df["cv_f1_std"]
    best_idx = metrics_df["selection_score"].idxmax()
    best_name = metrics_df.loc[best_idx, "model_name"]
    print(f"\n[FINAL] Modèle candidat · {best_name}")
    print(metrics_df.to_string(index=False))

    # On copie le best model sous un nom canonique pour l'API/Dashboard.
    src_path = MODELS_DIR / f"{best_name}.joblib"
    final_path = MODELS_DIR / "final_model.joblib"
    joblib.dump(fitted_pipelines[best_name], final_path, compress=3)
    print(f"[FINAL] Modèle final sauvegardé · {final_path}")

    # On enregistre le nom du best model dans un fichier texte simple
    # (consommé par le dashboard et l'API pour afficher l'identité du
    # modèle servi).
    with open(MODELS_DIR / "final_model_name.txt", "w", encoding="utf-8") as f:
        f.write(best_name)

    print("\nDone.")


if __name__ == "__main__":
    main()
