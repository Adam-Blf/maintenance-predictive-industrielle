# -*- coding: utf-8 -*-
"""Module d'interprétabilité des modèles entraînés.

Trois niveaux d'explicabilité (correspondant au cahier des charges)
---------------------------------------------------------------------
1. **Feature Importance native** (basique) · calculée pendant l'entraînement
   par réduction d'impureté Gini (RF) ou gain moyen (XGB). Rapide mais biaisée
   vers les variables continues et instable si les features sont corrélées.
   Disponible uniquement pour les modèles à base d'arbres.

2. **Permutation Importance** (recommandé) · agnostique au modèle. Permute
   aléatoirement chaque feature sur le test set et mesure la perte de
   performance (F1). Plus fiable que l'importance native car elle mesure
   l'effet réel sur la prédiction, pas un artefact de l'algorithme interne.
   Inconvénient : coûteuse en calcul (n_features x n_repeats x 1 inférence).

3. **SHAP** (avancé) · SHapley Additive exPlanations, fondées sur la théorie
   des jeux coopératifs. Attribue à chaque feature une contribution chiffrée
   à la prédiction individuelle (explicabilité locale) et agrège en importance
   globale (explicabilité globale). Propriétés ·
     - Additivité garantie : somme des SHAP values = prédiction.
     - Cohérence : si f(x) augmente quand x_i augmente, SHAP(x_i) >= 0.
     - Comparabilité inter-modèles.

Utilité métier
--------------
Un responsable maintenance doit pouvoir répondre · "Pourquoi le modèle
indique un risque de panne élevé pour cette machine ?". Les SHAP values
permettent de dire "principalement à cause d'une vibration de 8.2mm/s
(+0.34 log-odds) et de 450h sans maintenance (+0.18 log-odds)".
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from .config import (
    REPORTS_DIR,
    COLOR_ALERT_RED,
    COLOR_EFREI_BLUE,
    COLOR_EFREI_DARK,
)


def plot_native_feature_importance(
    model: Pipeline,
    feature_names: list[str],
    model_name: str,
    output_dir: Path = REPORTS_DIR,
    top_n: int = 15,
) -> Path | None:
    """Importance native des modèles à base d'arbres (Random Forest, XGBoost).

    Cette importance est calculée pendant l'entraînement par réduction
    d'impureté Gini (classification) ou de variance (régression). Elle est
    rapide mais a deux limites · biais en faveur des variables continues
    (vs catégorielles) et instabilité quand les features sont corrélées.

    Returns
    -------
    Path | None
        Chemin du PNG, ou None si le modèle n'expose pas l'attribut
        `feature_importances_` (ex. régression logistique, MLP).
    """
    # Le pipeline est composé d'un préprocesseur + classifieur · l'attribut
    # `feature_importances_` n'existe que sur le classifieur final.
    classifier = model.named_steps["classifier"]
    if not hasattr(classifier, "feature_importances_"):
        return None

    importances = classifier.feature_importances_

    # Tri décroissant + sélection des top_n features pour rester lisible.
    indices = np.argsort(importances)[::-1][:top_n]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        sorted_features[::-1],
        sorted_importances[::-1],
        color=COLOR_EFREI_BLUE,
        edgecolor="black",
    )
    ax.set_xlabel("Importance (réduction d'impureté)", fontsize=11)
    ax.set_title(
        f"Feature Importance native · {model_name}",
        fontsize=13,
        fontweight="bold",
        color=COLOR_EFREI_DARK,
    )
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()

    output_path = output_dir / f"feature_importance_native_{model_name}.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_permutation_importance(
    model: Pipeline,
    X_test,
    y_test,
    feature_names_raw: list[str],
    model_name: str,
    output_dir: Path = REPORTS_DIR,
    n_repeats: int = 10,
    random_state: int = 42,
) -> Path:
    """Permutation Importance · agnostique au modèle, robuste aux corrélations.

    Principe ·
      1. Mesurer la performance initiale.
      2. Permuter aléatoirement une variable.
      3. Recalculer la performance.
      4. La perte de performance = importance de la variable.

    Cette méthode est plus fiable que l'importance native pour comparer
    des modèles hétérogènes (Logistic vs RF vs MLP) car elle ne dépend
    pas de la mécanique interne du modèle.
    """
    # `n_repeats=10` · 10 permutations par feature pour stabiliser la mesure.
    # Plus on en fait, moins la mesure est bruitée, mais le coût explose.
    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=n_repeats,
        random_state=random_state,
        # `scoring="f1"` · cohérent avec la métrique principale en
        # classification déséquilibrée (panne minoritaire).
        scoring="f1",
        n_jobs=-1,
    )

    # Tri par importance moyenne décroissante (top features = en haut).
    sorted_idx = result.importances_mean.argsort()[::-1]
    top_features = [feature_names_raw[i] for i in sorted_idx[:15]]
    top_means = result.importances_mean[sorted_idx[:15]]
    top_stds = result.importances_std[sorted_idx[:15]]

    fig, ax = plt.subplots(figsize=(10, 6))
    # Barres horizontales avec barres d'erreur (1 sigma) · indiquent la
    # stabilité de la mesure entre permutations.
    ax.barh(
        top_features[::-1],
        top_means[::-1],
        xerr=top_stds[::-1],
        color=COLOR_ALERT_RED,
        edgecolor="black",
        ecolor="black",
        capsize=4,
    )
    ax.set_xlabel("Perte de F1 score lors de la permutation", fontsize=11)
    ax.set_title(
        f"Permutation Importance · {model_name}",
        fontsize=13,
        fontweight="bold",
        color=COLOR_EFREI_DARK,
    )
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()

    output_path = output_dir / f"permutation_importance_{model_name}.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def compute_shap_values(
    model: Pipeline,
    X_sample,
    feature_names_processed: list[str],
    model_name: str,
    output_dir: Path = REPORTS_DIR,
    max_samples: int = 500,
):
    """Calcule et trace les valeurs SHAP du modèle final.

    SHAP (SHapley Additive exPlanations) attribue à chaque feature une
    contribution chiffrée à la prédiction, avec ces propriétés ·
      - **Additivité** · la somme des contributions = la prédiction.
      - **Cohérence** · si une feature devient plus importante, son SHAP
        ne diminue pas.
      - **Local** · explique une prédiction individuelle.
      - **Global** · l'agrégation explique le modèle dans son ensemble.

    On limite à `max_samples` (500 par défaut) pour rester traitable en
    temps · SHAP a une complexité élevée sur les modèles non-linéaires.
    """
    try:
        import shap
    except ImportError:
        # SHAP non installé · on échoue silencieusement plutôt que de
        # casser tout le pipeline. Le rapport mentionnera l'absence.
        print("[WARN] SHAP non installé · skip SHAP plots.")
        return None, None

    # Échantillonnage aléatoire pour limiter la taille de calcul.
    if len(X_sample) > max_samples:
        X_sample = X_sample.sample(n=max_samples, random_state=42)

    # On extrait le préprocesseur et le classifieur séparément · SHAP
    # opère sur le classifieur "nu", après transformation des features.
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    X_transformed = preprocessor.transform(X_sample)
    # `feature_names_processed` est déjà la liste post-One-Hot.
    X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names_processed)

    # Sélection automatique de l'explainer adapté au type de modèle ·
    # TreeExplainer pour RF/XGB (rapide), KernelExplainer pour les autres.
    try:
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(X_transformed_df)
    except Exception:
        # Fallback · KernelExplainer sur petit échantillon de référence.
        background = X_transformed_df.sample(n=min(100, len(X_transformed_df)), random_state=42)
        explainer = shap.KernelExplainer(classifier.predict_proba, background)
        shap_values = explainer.shap_values(X_transformed_df, nsamples=100)

    # Normalisation de la forme · les versions recentes de SHAP retournent
    # soit une liste [class_0, class_1] (ancien format), soit un ndarray
    # 3D (n_samples, n_features, n_classes) pour le binaire. On extrait
    # systematiquement la contribution de la classe positive (panne).
    import numpy as np

    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    # ------------------------------------------------------------------
    # SHAP Summary Plot · vue globale, top features les plus influentes.
    # Le titre est mis APRES summary_plot (qui cree son propre axes) avec
    # `pad=20` pour ne pas se superposer au premier label de feature.
    # ------------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        X_transformed_df,
        show=False,
        plot_type="dot",
        max_display=10,
    )
    plt.title(
        f"SHAP Summary · {model_name}",
        fontsize=12,
        fontweight="bold",
        color=COLOR_EFREI_DARK,
        pad=18,
    )
    summary_path = output_dir / f"shap_summary_{model_name}.png"
    plt.savefig(summary_path, dpi=140, bbox_inches="tight", pad_inches=0.3)
    plt.close()

    # ------------------------------------------------------------------
    # SHAP Bar Plot · importance globale (moyenne des |SHAP|).
    # ------------------------------------------------------------------
    plt.figure(figsize=(10, 5.5))
    shap.summary_plot(
        shap_values,
        X_transformed_df,
        show=False,
        plot_type="bar",
        max_display=10,
        color=COLOR_EFREI_BLUE,
    )
    plt.title(
        f"SHAP Importance globale · {model_name}",
        fontsize=12,
        fontweight="bold",
        color=COLOR_EFREI_DARK,
        pad=18,
    )
    bar_path = output_dir / f"shap_bar_{model_name}.png"
    plt.savefig(bar_path, dpi=140, bbox_inches="tight", pad_inches=0.3)
    plt.close()

    return summary_path, bar_path
