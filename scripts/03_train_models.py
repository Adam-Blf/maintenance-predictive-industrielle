# -*- coding: utf-8 -*-
"""Script unique · entraînement et évaluation comparative des 4 modèles.

Architecture pédagogique du fichier (lecture top-down) ·

  ÉTAPE 1 · Chargement du dataset Kaggle officiel + split stratifié 80/20.
  ÉTAPE 2 · Préprocesseur scikit-learn (Imputer + Scaler + OHE) · inline.
  ÉTAPE 3 · MODÉLISATION BASELINE · Logistic Regression sur données
            prétraitées. Sert de point de référence aux 3 modèles suivants.
  ÉTAPE 4 · 3 modèles avancés sur les mêmes données prétraitées ·
            4.1 · Random Forest (bagging d'arbres CART).
            4.2 · XGBoost (gradient boosting).
            4.3 · MLP · Multi-Layer Perceptron (Deep Learning exigé).
  ÉTAPE 5 · Cross-validation 5-fold stratifiée sur les 4 modèles.
  ÉTAPE 6 · Comparaison globale, sélection du modèle final candidat.

Ce script est **autonome** · il contient toute la logique de modélisation
(loading, preprocessing, définition des 4 modèles, training, CV,
évaluation, sélection finale) dans un seul fichier lisible top-down.

Les seuls imports externes restants vers `src/` concernent ·
  - les **constantes de configuration** (chemins, seed, etc.) · `src.config`
  - les **fonctions de plotting** matplotlib (matrices de confusion,
    courbes ROC/PR, barplots) · `src.evaluation`
Cela évite ~600 lignes de matplotlib dupliquées sans nuire à la lisibilité
de la chaîne d'apprentissage.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Constantes projet (chemins, seed, listes de features) · centralisees
# dans src/config.py pour eviter la duplication entre tous les scripts.
from src.config import (  # noqa: E402
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    CV_FOLDS,
    DATA_PROCESSED_DIR,
    DATASET_PATH,
    MODELS_DIR,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    S03_DIR,
    TARGET_BINARY,
    TEST_SIZE,
    ensure_directories,
)

# Helpers de plotting matplotlib (matrices de confusion, ROC, PR, bars).
# Externalises dans src/evaluation.py pour ne pas polluer ce script avec
# 600+ lignes de matplotlib (sns.heatmap, fig/axes layout, savefig, etc.).
from src.evaluation import (  # noqa: E402
    plot_confusion_matrix,
    plot_metrics_barplot,
    plot_pr_curves,
    plot_roc_curves,
    plot_training_time_barplot,
)


# ---------------------------------------------------------------------------
# Helper de mise en forme console
# ---------------------------------------------------------------------------
def _banner(title: str, char: str = "=") -> None:
    """Separateur visuel pour reperer les sections lors de la lecture du log."""
    line = char * 70
    print(f"\n{line}\n{title}\n{line}")


# ---------------------------------------------------------------------------
# Dataclass de metriques (1 ligne par modele dans le tableau final)
# ---------------------------------------------------------------------------
@dataclass
class ClassificationMetrics:
    """Conteneur des 6 metriques + temps · 1 instance par modele.

    Format · `model_name | accuracy | precision | recall | f1 | roc_auc |
    pr_auc | fit_time_s | predict_time_ms`. Serialisable en dict pour
    construire un DataFrame final comparant les 4 modeles.
    """

    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    fit_time_s: float
    predict_time_ms: float

    def to_dict(self) -> dict:
        return asdict(self)


def compute_metrics(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    fit_time_s: float,
    predict_time_ms: float,
) -> ClassificationMetrics:
    """Calcule les 6 metriques de classification binaire.

    Le sujet impose Accuracy + Precision + Recall + F1 + ROC-AUC. On
    ajoute PR-AUC (Average Precision) car plus informative que ROC-AUC
    sur des classes desequilibrees · cf. Saito & Rehmsmeier 2015.
    """
    return ClassificationMetrics(
        model_name=name,
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        roc_auc=roc_auc_score(y_true, y_proba),
        pr_auc=average_precision_score(y_true, y_proba),
        fit_time_s=fit_time_s,
        predict_time_ms=predict_time_ms,
    )


# ===========================================================================
# ÉTAPE 2 · Preprocesseur scikit-learn (anti-data-leakage)
# ===========================================================================
def build_preprocessor() -> ColumnTransformer:
    """Construit le preprocesseur · imputation + scaling + encodage.

    Architecture ·
      - Numeriques · SimpleImputer(median) → StandardScaler.
      - Categoriels · SimpleImputer(most_frequent) → OneHotEncoder.

    Pourquoi mediane sur les capteurs · robuste aux outliers (vs moyenne).
    Le dataset Kaggle injecte ~3-4% de NaN sur 5 capteurs (cf. EDA).

    Pourquoi OHE avec `handle_unknown="ignore"` · garantit que le modele
    ne plante pas en inference si une nouvelle modalite apparait
    (ex. nouveau machine_type ajoute en production).

    Anti-data-leakage · ce ColumnTransformer est ENCAPSULE dans une
    `sklearn.Pipeline` avec l'estimateur · `pipeline.fit(X_train, y_train)`
    n'ajuste l'imputer/scaler que sur le train. Le test n'est jamais vu
    pendant le fit.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",  # garantit qu'aucune colonne fuyante (timestamp,
                            # machine_id, cibles) ne passe en feature.
        verbose_feature_names_out=False,  # noms propres sans prefixe num__/cat__
    )


# ===========================================================================
# ÉTAPE 3 · BASELINE · Logistic Regression
# ===========================================================================
def build_logistic_regression() -> Pipeline:
    """Baseline interpretable · regression logistique sur features
    standardisees. Sert de point de reference aux modeles 4.1, 4.2, 4.3.

    Hyperparametres ·
      - `class_weight="balanced"` · compense le desequilibre des classes
        (les pannes sont minoritaires comme dans la realite).
      - `max_iter=1000` · suffisant pour converger sur ~24k lignes
        avec des features standardisees.
      - solver `lbfgs` · adapte aux problemes binaires de taille moyenne.
    """
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    solver="lbfgs",
                ),
            ),
        ]
    )


# ===========================================================================
# ÉTAPE 4 · 3 modeles avances
# ===========================================================================
def build_random_forest() -> Pipeline:
    """4.1 · Random Forest · ensemble bagging d'arbres CART.

    Hyperparametres ·
      - 200 arbres · compromis biais/variance/temps. Plus eleve la
        stabilite de la feature importance.
      - `max_depth=None` · on laisse les arbres pousser, la regularisation
        vient du bagging et de la diversite des splits.
      - `min_samples_leaf=5` · empeche les feuilles a 1 sample qui
        sur-apprennent fortement sur 24k lignes.
      - `n_jobs=-1` · parallelisation sur tous les coeurs CPU.
    """
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=None,
                    min_samples_leaf=5,
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def build_xgboost(scale_pos_weight: float = 1.0) -> Pipeline:
    """4.2 · XGBoost · gradient boosting d'arbres.

    Hyperparametres ·
      - `n_estimators=300` + `learning_rate=0.05` · descente lente, plus
        robuste a l'overfit que 100 arbres avec lr=0.1.
      - `max_depth=6` · profondeur maximale par arbre, suffisante pour
        capturer les interactions sans exploser la complexite.
      - `subsample=0.85` + `colsample_bytree=0.85` · bagging stochastique
        integre qui regularise sans sacrifier la precision.
      - `scale_pos_weight` · ratio neg/pos calcule a l'entrainement pour
        gerer le desequilibre de classes (plus efficace que SMOTE pour
        XGBoost).
    """
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                XGBClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    scale_pos_weight=scale_pos_weight,
                    eval_metric="logloss",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    tree_method="hist",
                ),
            ),
        ]
    )


def build_mlp() -> Pipeline:
    """4.3 · MLP · Multi-Layer Perceptron · le modele Deep Learning du projet.

    Architecture · 64 → 32 → 16 (3 couches cachees degressives). Cette
    pyramide inversee est un design eprouve sur tabulaire · elle apprend
    des features de plus en plus abstraites tout en evitant la
    sur-parametrisation.

    Hyperparametres ·
      - `activation="relu"` · standard, evite les vanishing gradients.
      - `solver="adam"` · adaptatif, robuste aux hyperparametres mal
        choisis.
      - `alpha=1e-3` · regularisation L2 moderee pour combattre l'overfit.
      - `early_stopping=True` · arret automatique si le score validation
        interne ne progresse plus pendant 10 iterations.
    """
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                MLPClassifier(
                    hidden_layer_sizes=(64, 32, 16),
                    activation="relu",
                    solver="adam",
                    alpha=1e-3,
                    batch_size=256,
                    learning_rate_init=1e-3,
                    max_iter=200,
                    early_stopping=True,
                    n_iter_no_change=10,
                    validation_fraction=0.1,
                    random_state=RANDOM_STATE,
                    verbose=False,
                ),
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Helper d'entrainement reutilise par les 4 blocs
# ---------------------------------------------------------------------------
def train_one_model(name: str, pipeline: Pipeline, X_train, y_train, X_test, y_test):
    """Entraine un pipeline (preprocessor + estimator), calcule les
    metriques sur le test set, sauvegarde l'artefact joblib et trace la
    matrice de confusion.

    Returns
    -------
    tuple
        (metrics: ClassificationMetrics, y_pred, y_proba, fitted_pipeline)
    """
    # Mesure du temps d'entrainement (ecoresponsabilite · cf. RNCP C4.3).
    t0 = time.time()
    pipeline.fit(X_train, y_train)
    fit_time = time.time() - t0

    # Mesure de la latence · temps de prediction par echantillon.
    t0 = time.time()
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)
    predict_time_ms = (time.time() - t0) * 1000.0 / len(X_test)

    metrics = compute_metrics(
        name=name,
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

    # Persistance modele entraine + matrice de confusion individuelle.
    model_path = MODELS_DIR / f"{name}.joblib"
    joblib.dump(pipeline, model_path, compress=3)
    print(f"  Saved · {model_path}")
    plot_confusion_matrix(y_test.values, y_pred, name, output_dir=S03_DIR)

    return metrics, y_pred, y_proba, pipeline


# ===========================================================================
# ÉTAPE 5 · Cross-validation 5-fold stratifiee
# ===========================================================================
def cross_validate_all(X, y, model_builders: dict) -> dict[str, dict]:
    """Cross-validation 5-fold stratifiee sur les 4 modeles · validation
    de la stabilite (anti-overfit, anti-coup-de-chance-du-split).

    Anti-data-leakage · `cross_val_score` re-fit le pipeline complet
    (preprocesseur + estimateur) sur chaque fold de train · le scaler
    et l'imputer ne voient JAMAIS le fold de validation pendant le fit.
    """
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_results: dict = {}

    for name, builder in model_builders.items():
        model = builder()  # instance fraiche pour eviter les fuites entre folds
        scores = cross_val_score(model, X, y, scoring="f1", cv=skf, n_jobs=-1)
        cv_results[name] = {
            "f1_mean": float(scores.mean()),
            "f1_std": float(scores.std()),
            "f1_folds": scores.tolist(),
        }
        print(f"  {name:<22} · F1 = {scores.mean():.4f} ± {scores.std():.4f}")

    return cv_results


# ===========================================================================
# Pipeline principal · 6 etapes sequentielles
# ===========================================================================
def main() -> None:
    """Point d'entree · pipeline complet d'entrainement comparatif."""
    ensure_directories()

    # ------------------------------------------------------------------
    # ÉTAPE 1 · Chargement du dataset + split stratifie 80/20
    # ------------------------------------------------------------------
    _banner("ÉTAPE 1 · Chargement du dataset Kaggle officiel")
    df = pd.read_csv(DATASET_PATH, low_memory=False)
    print(f"  Shape · {df.shape}")
    print(f"  Pannes 24h · {df[TARGET_BINARY].mean():.2%}")

    print("\n  Split stratifie 80/20 (seed=42)...")
    X = df[ALL_FEATURES].copy()
    y = df[TARGET_BINARY].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,  # CRUCIAL · evite un fold test sans pannes
        random_state=RANDOM_STATE,
    )
    # Persistance pour reutilisation par 04_interpret.py et le rapport.
    X_train.to_csv(DATA_PROCESSED_DIR / "X_train.csv", index=False)
    X_test.to_csv(DATA_PROCESSED_DIR / "X_test.csv", index=False)
    y_train.to_csv(DATA_PROCESSED_DIR / "y_train.csv", index=False)
    y_test.to_csv(DATA_PROCESSED_DIR / "y_test.csv", index=False)
    print(f"  Train · {X_train.shape}, Test · {X_test.shape}")

    # Calcul scale_pos_weight pour XGBoost (ratio neg/pos sur train).
    scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    print(f"  scale_pos_weight (XGBoost) · {scale_pos_weight:.2f}")

    # Conteneurs partages par les 4 blocs de modelisation.
    all_metrics: list[dict] = []
    roc_payload: dict = {}
    fitted_pipelines: dict = {}

    # ==================================================================
    # ÉTAPE 3 · MODÉLISATION BASELINE sur donnees pretraitees
    # ==================================================================
    # Pourquoi un baseline · avant tout modele complexe, on entraine un
    # modele simple et interpretable pour avoir une reference chiffree.
    # Cela permet ensuite de quantifier le gain reel apporte par les
    # modeles avances (RF, XGBoost, MLP) ci-dessous.
    # ==================================================================
    _banner("ÉTAPE 3 · BASELINE · Logistic Regression (modele lineaire)")
    print("  Role · point de reference interpretable. Coefficients lisibles,")
    print("         entrainement rapide, sert de seuil bas pour les 3 modeles 4.x.")
    print()

    baseline_metrics, _, baseline_proba, baseline_pipe = train_one_model(
        "logistic_regression",
        build_logistic_regression(),
        X_train, y_train, X_test, y_test,
    )
    all_metrics.append(baseline_metrics.to_dict())
    roc_payload["logistic_regression"] = {
        "y_true": y_test.values,
        "y_proba": baseline_proba,
    }
    fitted_pipelines["logistic_regression"] = baseline_pipe

    # ==================================================================
    # ÉTAPE 4 · 3 MODÈLES AVANCÉS sur les memes donnees pretraitees
    # ==================================================================
    # Chaque modele apporte une valeur ajoutee differente par rapport
    # au baseline · on les evalue tous sur le MEME split test pour que
    # la comparaison soit equitable.
    # ==================================================================

    # ---------- 4.1 · Random Forest ----------
    _banner("ÉTAPE 4.1 · Random Forest (bagging d'arbres CART)")
    print("  Apport vs baseline · capture les non-linearites et interactions")
    print("                       capteurs sans engineering manuel.")
    print()
    rf_metrics, _, rf_proba, rf_pipe = train_one_model(
        "random_forest", build_random_forest(),
        X_train, y_train, X_test, y_test,
    )
    all_metrics.append(rf_metrics.to_dict())
    roc_payload["random_forest"] = {"y_true": y_test.values, "y_proba": rf_proba}
    fitted_pipelines["random_forest"] = rf_pipe

    # ---------- 4.2 · XGBoost ----------
    _banner("ÉTAPE 4.2 · XGBoost (gradient boosting d'arbres)")
    print("  Apport vs baseline · etat de l'art sur tabulaire. Boosting")
    print("                       sequentiel, scale_pos_weight gere le")
    print("                       desequilibre nativement.")
    print()
    xgb_metrics, _, xgb_proba, xgb_pipe = train_one_model(
        "xgboost", build_xgboost(scale_pos_weight=scale_pos_weight),
        X_train, y_train, X_test, y_test,
    )
    all_metrics.append(xgb_metrics.to_dict())
    roc_payload["xgboost"] = {"y_true": y_test.values, "y_proba": xgb_proba}
    fitted_pipelines["xgboost"] = xgb_pipe

    # ---------- 4.3 · MLP (Deep Learning) ----------
    _banner("ÉTAPE 4.3 · MLP · Multi-Layer Perceptron (Deep Learning)")
    print("  Apport vs baseline · modele DL exige par le sujet. Architecture")
    print("                       64-32-16, ReLU, early stopping anti-overfit.")
    print()
    mlp_metrics, _, mlp_proba, mlp_pipe = train_one_model(
        "mlp", build_mlp(),
        X_train, y_train, X_test, y_test,
    )
    all_metrics.append(mlp_metrics.to_dict())
    roc_payload["mlp"] = {"y_true": y_test.values, "y_proba": mlp_proba}
    fitted_pipelines["mlp"] = mlp_pipe

    # DataFrame recapitulatif (4 lignes, une par modele).
    metrics_df = pd.DataFrame(all_metrics)

    # ==================================================================
    # ÉTAPE 5 · Cross-validation 5-fold stratifiee sur les 4 modeles
    # ==================================================================
    _banner("ÉTAPE 5 · Cross-validation 5-fold stratifiee (F1-score)")
    # Sous-echantillon CV pour rester rapide (CV complete avec MLP couteuse).
    cv_sample_size = min(8_000, len(X_train))
    sample_idx = np.random.RandomState(RANDOM_STATE).choice(
        len(X_train), size=cv_sample_size, replace=False
    )
    cv_builders = {
        "logistic_regression": build_logistic_regression,
        "random_forest": build_random_forest,
        "xgboost": lambda: build_xgboost(scale_pos_weight=scale_pos_weight),
        "mlp": build_mlp,
    }
    cv_results = cross_validate_all(
        X_train.iloc[sample_idx], y_train.iloc[sample_idx], cv_builders
    )
    metrics_df["cv_f1_mean"] = metrics_df["model_name"].map(
        lambda n: cv_results[n]["f1_mean"]
    )
    metrics_df["cv_f1_std"] = metrics_df["model_name"].map(
        lambda n: cv_results[n]["f1_std"]
    )

    # ==================================================================
    # ÉTAPE 6 · Comparaison globale + selection du modele final
    # ==================================================================
    _banner("ÉTAPE 6 · Comparaison globale + selection du modele final")
    plot_roc_curves(roc_payload, output_dir=S03_DIR)
    plot_pr_curves(roc_payload, output_dir=S03_DIR)
    plot_metrics_barplot(metrics_df, output_dir=S03_DIR)
    plot_training_time_barplot(metrics_df, output_dir=S03_DIR)

    # Persistance · CSV pour Excel/Pandas + JSON pour Streamlit/API.
    metrics_df.to_csv(S03_DIR / "metrics_summary.csv", index=False)
    metrics_df.to_json(
        S03_DIR / "metrics_summary.json", orient="records", indent=2
    )
    with open(S03_DIR / "cv_results.json", "w", encoding="utf-8") as f:
        json.dump(cv_results, f, indent=2)

    # Selection finale · F1 test - 0.5 × ecart-type CV.
    # On penalise l'instabilite car en production, un modele qui varie
    # de ±5% selon les donnees est risque.
    metrics_df["selection_score"] = (
        metrics_df["f1"] - 0.5 * metrics_df["cv_f1_std"]
    )
    best_idx = metrics_df["selection_score"].idxmax()
    best_name = metrics_df.loc[best_idx, "model_name"]
    print(f"\n  Modele candidat retenu · {best_name}")
    print(metrics_df.to_string(index=False))

    # On copie le best model sous un nom canonique pour l'API/Dashboard.
    final_path = MODELS_DIR / "final_model.joblib"
    joblib.dump(fitted_pipelines[best_name], final_path, compress=3)
    print(f"\n  Modele final sauvegarde · {final_path}")

    # Nom du best model dans un fichier texte (consomme par dashboard/API).
    with open(MODELS_DIR / "final_model_name.txt", "w", encoding="utf-8") as f:
        f.write(best_name)

    print("\nDone.")


if __name__ == "__main__":
    main()
