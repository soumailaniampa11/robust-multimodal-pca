"""Baseline models used for comparison with the fusion ensemble."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config


class SklearnBaseline:
    """Wraps a sklearn classifier on a fixed feature list.

    `drop` simulates missing modalities at test time by setting their features
    to NaN, which is what a deployed early-fusion model would receive.
    """

    def __init__(self, name: str, estimator, features: list[str]):
        self.name = name
        self.estimator = estimator
        self.features = features

    def _X(self, df: pd.DataFrame, drop: tuple[str, ...] = ()) -> np.ndarray:
        X = df[self.features].copy()
        for m in drop:
            for col in config.MODALITIES[m]:
                if col in X:
                    X[col] = np.nan
        return X.to_numpy(dtype=float)

    def fit(self, df_train: pd.DataFrame, df_val: pd.DataFrame | None = None) -> "SklearnBaseline":
        self.estimator.fit(self._X(df_train), df_train[config.TARGET].to_numpy())
        return self

    def predict_proba(self, df: pd.DataFrame, drop: tuple[str, ...] = ()) -> np.ndarray:
        return self.estimator.predict_proba(self._X(df, drop))[:, 1]


def clinical_logistic() -> SklearnBaseline:
    est = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
        LogisticRegression(max_iter=1000, C=1.0),
    )
    return SklearnBaseline("Clinical LR", est, config.MODALITIES["clinical"])


def early_fusion_logistic() -> SklearnBaseline:
    feats = [c for cols in config.MODALITIES.values() for c in cols]
    est = make_pipeline(
        SimpleImputer(strategy="mean", add_indicator=False), StandardScaler(),
        LogisticRegression(max_iter=1000, C=1.0),
    )
    return SklearnBaseline("Early-fusion LR (mean imputation)", est, feats)


def early_fusion_gbdt() -> SklearnBaseline:
    feats = [c for cols in config.MODALITIES.values() for c in cols]
    est = HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=300, l2_regularization=1.0, random_state=config.SEED
    )
    return SklearnBaseline("Early-fusion GBDT", est, feats)


class PiradsReference:
    """Radiologist reference: the maximum PI-RADS score used directly as a risk score."""

    name = "PI-RADS (radiologist)"

    def fit(self, *args, **kwargs) -> "PiradsReference":
        return self

    def predict_proba(self, df: pd.DataFrame, drop: tuple[str, ...] = ()) -> np.ndarray:
        if "radiology" in drop:
            return np.full(len(df), 3.0)
        return df["pirads_max"].fillna(3.0).to_numpy(dtype=float)
