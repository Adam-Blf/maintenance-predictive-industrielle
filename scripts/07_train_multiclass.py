# -*- coding: utf-8 -*-
"""Script · entraînement de la classification multi-classe (failure_type).

Rôle dans le pipeline
----------------------
Script n°7, indépendant du script 03 (binaire). Peut être exécuté
après le script 01 uniquement. Tâche bonus du sujet.

Entrées
-------
data/raw/predictive_maintenance_v3.csv
    Dataset complet (24 042 lignes). Filtrage sur les pannes dans le script.

Sorties
-------
models/multiclass_{model}.joblib (x4)
    Pipelines entraînés pour chacun des 4 modèles.
models/multiclass_final.joblib
    Copie du meilleur modèle (critère : macro-F1 sur test).
models/multiclass_final_name.txt
    Nom textuel du meilleur modèle.
models/multiclass_classes.json
    Liste ordonnée des classes (nécessaire pour décoder les prédictions
    XGBoost qui retourne des entiers).
reports/metrics_multiclass.csv / .json
    Tableau comparatif des 4 modèles (accuracy, macro-F1, weighted-F1).
reports/multiclass_confusion_matrix.png
    Matrice de confusion normalisée du meilleur modèle.
reports/multiclass_classification_report.json
    Precision/Recall/F1 par classe pour le meilleur modèle.

Pré-requis
----------
Script 01 exécuté (dataset disponible dans data/raw/).

Lien cahier des charges
-----------------------
Classification multi-classe demandée explicitement comme tâche bonus
dans le cahier des charges (section "Modélisation avancée"). Valorise
la grille de notation sur les 5 types de pannes.

Usage ·
    python scripts/07_train_multiclass.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from src.config import (  # noqa: E402
    ALL_FEATURES,
    COLOR_EFREI_DARK,
    DATA_PROCESSED_DIR,
    MODELS_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    REPORTS_FIGURES_DIR,
    TARGET_MULTICLASS,
    TEST_SIZE,
    ensure_directories,
)
from src.data_loader import load_dataset  # noqa: E402
from src.models_multiclass import (  # noqa: E402
    build_logistic_multiclass,
    build_mlp_multiclass,
    build_rf_multiclass,
    build_xgb_multiclass,
)


def main() -> None:
    """Orchestre l'entraînement, l'évaluation et la persistance des 4 modèles multi-classe."""
    ensure_directories()

    print("[MULTICLASS] Chargement dataset...")
    df = load_dataset()
    # `failure_type` est sauvegarde comme str "None" mais pandas peut le
    # relire en NaN. On filtre defensivement sur les deux.
    mask = df[TARGET_MULTICLASS].notna() & (df[TARGET_MULTICLASS].astype(str) != "None")
    df_failed = df.loc[mask].copy()
    print(f"  {len(df_failed):,} machines en panne ({len(df_failed)/len(df):.1%})")
    print("  Distribution avant filtrage classes rares :")
    for k, v in df_failed[TARGET_MULTICLASS].value_counts().items():
        print(f"    {k:<12} · {v}")

    # Filtre les classes avec moins de 30 echantillons · evite les erreurs
    # de stratification et la sur-paramétrisation sur des classes rares.
    counts = df_failed[TARGET_MULTICLASS].value_counts()
    valid_classes = counts[counts >= 30].index.tolist()
    df_failed = df_failed[df_failed[TARGET_MULTICLASS].isin(valid_classes)].copy()
    print(f"  Classes retenues (>=30 obs) : {valid_classes}")

    if df_failed[TARGET_MULTICLASS].nunique() < 2:
        print("[WARN] Trop peu de classes pour la classification multi-classe (skip).")
        return

    # Encodage des étiquettes texte en entiers (XGBoost l'exige).
    le = LabelEncoder()
    y = le.fit_transform(df_failed[TARGET_MULTICLASS])
    classes = list(le.classes_)
    n_classes = len(classes)
    X = df_failed[ALL_FEATURES].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    factories = {
        "logistic_regression": build_logistic_multiclass,
        "random_forest": build_rf_multiclass,
        "xgboost": lambda: build_xgb_multiclass(num_class=n_classes),
        "mlp": build_mlp_multiclass,
    }

    results = []
    cm_data = {}
    for name, factory in factories.items():
        print(f"\n[TRAIN] {name}...")
        pipeline = factory()
        t0 = time.time()
        pipeline.fit(X_train, pd.Series(y_train))
        fit_time = time.time() - t0

        y_pred = pipeline.predict(X_test)
        macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
        weighted_f1 = float(f1_score(y_test, y_pred, average="weighted"))
        acc = float(accuracy_score(y_test, y_pred))
        print(
            f"  acc={acc:.4f}  macro-F1={macro_f1:.4f}  "
            f"weighted-F1={weighted_f1:.4f}  fit={fit_time:.2f}s"
        )

        joblib.dump(pipeline, MODELS_DIR / f"multiclass_{name}.joblib", compress=3)
        results.append(
            {
                "model_name": name,
                "accuracy": acc,
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
                "fit_time_s": fit_time,
            }
        )
        cm_data[name] = (y_test.copy(), y_pred.copy())

    # Sélection du meilleur modèle (macro-F1).
    df_results = pd.DataFrame(results)
    best_idx = df_results["macro_f1"].idxmax()
    best_name = df_results.loc[best_idx, "model_name"]
    print(f"\n[FINAL] Multi-classe · meilleur modèle = {best_name}")
    print(df_results.to_string(index=False))

    df_results.to_csv(REPORTS_DIR / "metrics_multiclass.csv", index=False)
    df_results.to_json(REPORTS_DIR / "metrics_multiclass.json", orient="records", indent=2)

    # Matrice de confusion du meilleur modèle.
    y_true_best, y_pred_best = cm_data[best_name]
    cm = confusion_matrix(y_true_best, y_pred_best, normalize="true")
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2%",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        ax=ax,
        annot_kws={"size": 10, "weight": "bold"},
        cbar=False,
    )
    ax.set_xlabel("Prédiction", fontweight="bold")
    ax.set_ylabel("Vérité terrain", fontweight="bold")
    ax.set_title(
        f"Matrice de confusion multi-classe · {best_name}",
        fontsize=12,
        fontweight="bold",
        color=COLOR_EFREI_DARK,
    )
    plt.tight_layout()
    fig.savefig(
        REPORTS_FIGURES_DIR / "multiclass_confusion_matrix.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)

    # Rapport classification détaillé.
    report = classification_report(
        y_true_best,
        y_pred_best,
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    with open(REPORTS_DIR / "multiclass_classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Persistance · meilleur modèle multi-classe + classes.
    joblib.dump(
        joblib.load(MODELS_DIR / f"multiclass_{best_name}.joblib"),
        MODELS_DIR / "multiclass_final.joblib",
        compress=3,
    )
    (MODELS_DIR / "multiclass_classes.json").write_text(json.dumps(classes), encoding="utf-8")
    (MODELS_DIR / "multiclass_final_name.txt").write_text(best_name, encoding="utf-8")

    print("Done.")


if __name__ == "__main__":
    main()
