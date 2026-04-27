# -*- coding: utf-8 -*-
"""Pipeline de préparation des données.

Ce module construit un `ColumnTransformer` sklearn qui applique en une
seule passe :
  - imputation médiane sur les variables numériques (robuste aux outliers
    capteurs),
  - standardisation StandardScaler (centrage/réduction obligatoire pour
    le MLP et bénéfique pour les modèles à base de distance),
  - encodage One-Hot pour `operating_mode` (faible cardinalité, pas de
    risque d'explosion dimensionnelle).

Le pipeline est sérialisé via joblib pour être réappliqué à l'identique
côté API et dashboard, ce qui élimine tout risque de **data leakage**
(les statistiques d'imputation et de scaling sont calculées sur le
train uniquement, jamais sur le test).
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """Construit le préprocesseur unifié pour features numériques + cat.

    Returns
    -------
    ColumnTransformer
        Préprocesseur prêt à être chaîné dans une `sklearn.Pipeline`.

    Notes
    -----
    Choix techniques justifiés ·

    * **Imputer médiane** plutôt que moyenne · les capteurs industriels
      produisent occasionnellement des valeurs aberrantes (saturation,
      coupures de transmission). La médiane est insensible à ces
      outliers contrairement à la moyenne.

    * **StandardScaler** plutôt que MinMaxScaler · on conserve la forme
      des distributions (utile pour la régularisation L2 du MLP) et on
      résiste mieux aux outliers résiduels après imputation.

    * **OneHotEncoder(handle_unknown="ignore")** · si l'API reçoit une
      valeur de mode opératoire jamais vue à l'entraînement, on retourne
      un vecteur de zéros plutôt que de lever une exception.

    * **sparse_output=False** · les modèles arbres (Random Forest,
      XGBoost) gèrent mieux les matrices denses, et la mémoire n'est pas
      un goulot d'étranglement avec ~24k lignes.
    """
    # Pipeline numérique · imputation -> standardisation. L'ordre est
    # crucial · si on standardisait avant d'imputer, les NaN pollueraient
    # les statistiques.
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Pipeline catégoriel · imputation par mode (most_frequent) puis
    # one-hot. `handle_unknown="ignore"` est un filet de sécurité pour
    # l'inférence en production.
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    # ColumnTransformer · applique chaque sous-pipeline aux colonnes voulues
    # et concatène les résultats. `remainder="drop"` garantit qu'aucune
    # colonne (timestamp, machine_id, cibles) ne fuite dans les features.
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Retourne les noms de features après transformation One-Hot.

    Utile pour l'interprétabilité (feature_importances_ et SHAP) et
    pour afficher des barplots correctement étiquetés dans le rapport.

    Parameters
    ----------
    preprocessor : ColumnTransformer
        Préprocesseur déjà ajusté (`fit` ou `fit_transform`).

    Returns
    -------
    list[str]
        Noms de colonnes dans l'ordre où elles sortent du transformer.
    """
    # `get_feature_names_out` n'est disponible qu'après fit, on délègue
    # à sklearn la logique de nommage post-OneHot ("operating_mode_Normal"
    # etc.).
    return list(preprocessor.get_feature_names_out())
