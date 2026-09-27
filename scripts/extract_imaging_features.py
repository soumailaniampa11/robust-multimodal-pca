"""Extract quantitative anatomical features from AI-derived MRI segmentations.

For every PI-CAI case we read the whole-gland mask (Bosma22b) and the zonal
mask (Yuan23: 1 = peripheral zone, 2 = transition zone) and compute physical volumes in mL from the
voxel spacing. Results are cached in data/processed/imaging_features.csv.

Note: csPCa lesion masks are deliberately NOT used as features: they only
exist for positive cases and would leak the label.
"""
from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pcafusion import config  # noqa: E402
from pcafusion.data import load_marksheet  # noqa: E402


def _volume_ml(mask: np.ndarray, voxel_mm3: float, label: int) -> float:
    return float((mask == label).sum() * voxel_mm3 / 1000.0)


def extract_case(case_id: str) -> dict:
    row: dict = {"case_id": case_id}
    gland_path = config.WHOLE_GLAND_DIR / f"{case_id}.nii.gz"
    zonal_path = config.ZONAL_DIR / f"{case_id}.nii.gz"

    if gland_path.exists():
        img = nib.load(str(gland_path))
        vox = float(np.prod(img.header.get_zooms()[:3]))
        mask = np.asarray(img.dataobj)
        vol = _volume_ml(mask, vox, 1)
        # An empty or implausible segmentation is treated as a failed extraction.
        row["gland_volume_seg"] = vol if 5.0 < vol < 500.0 else np.nan

    if zonal_path.exists():
        img = nib.load(str(zonal_path))
        vox = float(np.prod(img.header.get_zooms()[:3]))
        mask = np.asarray(img.dataobj)
        a, b = _volume_ml(mask, vox, 1), _volume_ml(mask, vox, 2)
        if a + b > 5.0:
            row["pz_volume_seg"] = a  # label 1: peripheral zone
            row["tz_volume_seg"] = b  # label 2: transition zone
            row["tz_fraction_seg"] = b / (a + b)
    return row


def main() -> None:
    cases = load_marksheet()["case_id"].tolist()
    with ProcessPoolExecutor() as pool:
        rows = list(pool.map(extract_case, cases, chunksize=16))
    df = pd.DataFrame(rows)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.IMAGING_FEATURES, index=False)
    print(f"Saved {len(df)} cases -> {config.IMAGING_FEATURES}")
    print(df.describe().T[["count", "mean", "50%"]])


if __name__ == "__main__":
    main()
