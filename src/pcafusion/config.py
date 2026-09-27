"""Paths and modality definitions shared across the project."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "picai_labels"
MARKSHEET = RAW_DIR / "clinical_information" / "marksheet.csv"
WHOLE_GLAND_DIR = RAW_DIR / "anatomical_delineations" / "whole_gland" / "AI" / "Bosma22b"
ZONAL_DIR = RAW_DIR / "anatomical_delineations" / "zonal_pz_tz" / "AI" / "Yuan23"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
IMAGING_FEATURES = PROCESSED_DIR / "imaging_features.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
RESULTS_DIR = OUTPUT_DIR / "results"
MODELS_DIR = OUTPUT_DIR / "models"
REPORTS_DIR = OUTPUT_DIR / "reports"

# Each modality is a named group of features. The fusion model encodes each
# group separately so that an entire modality can be absent at inference time.
MODALITIES: dict[str, list[str]] = {
    # Clinical / laboratory information (EHR-like)
    "clinical": ["age", "log_psa", "psad_reported", "volume_reported"],
    # Radiologist's structured MRI report (PI-RADS assessment)
    "radiology": ["pirads_max", "n_suspicious_lesions", "pirads_ge4"],
    # Quantitative MRI-derived anatomy from AI segmentations (whole gland + zones)
    "imaging": ["gland_volume_seg", "pz_volume_seg", "tz_volume_seg", "tz_fraction_seg"],
}

TARGET = "cspca"
GROUP = "patient_id"
CENTER = "center"
SEED = 42
