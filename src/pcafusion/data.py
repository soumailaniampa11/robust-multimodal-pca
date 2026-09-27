"""Loading and feature engineering for the PI-CAI public dataset.

The raw marksheet stores one row per MRI study. PI-RADS scores are stored as a
comma-separated string (one score per lesion, ``N/A`` for MRI-invisible
lesions). Missing values are kept as NaN: the downstream models are expected to
handle them explicitly rather than rely on silent imputation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def parse_pirads(value: str) -> tuple[float, int]:
    """Return (max PI-RADS, number of lesions with PI-RADS >= 3)."""
    if pd.isna(value):
        return np.nan, 0
    scores = []
    for token in str(value).split(","):
        token = token.strip()
        if token.isdigit():
            scores.append(int(token))
    if not scores:
        return np.nan, 0
    return float(max(scores)), int(sum(s >= 3 for s in scores))


def load_marksheet(path=config.MARKSHEET) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["case_id"] = df["patient_id"].astype(str) + "_" + df["study_id"].astype(str)
    df[config.TARGET] = (df["case_csPCa"] == "YES").astype(int)

    df["age"] = df["patient_age"].astype(float)
    df["log_psa"] = np.log1p(df["psa"])
    df["psad_reported"] = df["psad"]
    df["volume_reported"] = df["prostate_volume"]

    parsed = df["lesion_PIRADS"].apply(parse_pirads)
    df["pirads_max"] = [p[0] for p in parsed]
    df["n_suspicious_lesions"] = [float(p[1]) for p in parsed]
    df["pirads_ge4"] = (df["pirads_max"] >= 4).astype(float).where(df["pirads_max"].notna())

    # Verification-bias flag: PI-RADS <= 2 cases without tissue sampling were
    # labelled negative on MRI alone (see picai_labels README).
    df["biopsied"] = df["histopath_type"].notna()
    return df


def load_dataset(with_imaging: bool = True) -> pd.DataFrame:
    df = load_marksheet()
    if with_imaging:
        if not config.IMAGING_FEATURES.exists():
            raise FileNotFoundError(
                f"{config.IMAGING_FEATURES} not found. Run scripts/extract_imaging_features.py first."
            )
        img = pd.read_csv(config.IMAGING_FEATURES)
        df = df.merge(img, on="case_id", how="left")
    else:
        for col in config.MODALITIES["imaging"]:
            df[col] = np.nan
    return df


def feature_columns() -> list[str]:
    return [c for cols in config.MODALITIES.values() for c in cols]


def modality_availability(df: pd.DataFrame) -> pd.DataFrame:
    """Boolean frame: is at least one feature of each modality observed?"""
    return pd.DataFrame(
        {m: df[cols].notna().any(axis=1) for m, cols in config.MODALITIES.items()},
        index=df.index,
    )
