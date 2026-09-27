"""Train the deployable fusion ensemble used by the decision-support agent.

A small set of patients (default 30) is held out as *demo cases*: the agent is
demonstrated on cases the model never saw. The remaining patients are split
80/20 into training and calibration (temperature scaling, conformal
calibration, operating threshold).
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pcafusion import config  # noqa: E402
from pcafusion.data import load_dataset  # noqa: E402
from pcafusion.evaluation import threshold_for_sensitivity  # noqa: E402
from pcafusion.models import FusionEnsemble  # noqa: E402
from pcafusion.uncertainty import ClassConditionalConformal  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-demo-patients", type=int, default=30)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    df = load_dataset().reset_index(drop=True)
    rng = np.random.default_rng(config.SEED)
    patients = df[config.GROUP].unique()
    demo_patients = set(rng.choice(patients, args.n_demo_patients, replace=False))
    is_demo = df[config.GROUP].isin(demo_patients).to_numpy()
    demo, dev = df[is_demo], df[~is_demo]

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=config.SEED)
    fit_idx, cal_idx = next(sgkf.split(dev, dev[config.TARGET], dev[config.GROUP]))
    fit_df, cal_df = dev.iloc[fit_idx], dev.iloc[cal_idx]

    n_members, epochs = (2, 30) if args.quick else (5, 200)
    model = FusionEnsemble(n_members=n_members, epochs=epochs).fit(fit_df, cal_df)
    cal_p, cal_y = model.predict_proba(cal_df), cal_df[config.TARGET].to_numpy()

    bundle = {
        "model": model,
        "conformal": ClassConditionalConformal(0.10).fit(cal_p, cal_y),
        "threshold_sens95": threshold_for_sensitivity(cal_y, cal_p, 0.95),
        "train_case_ids": fit_df["case_id"].tolist(),
        "calibration_case_ids": cal_df["case_id"].tolist(),
        "demo_case_ids": demo["case_id"].tolist(),
    }
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.MODELS_DIR / "fusion_bundle.pkl", "wb") as f:
        pickle.dump(bundle, f)
    print(f"Trained on {len(fit_df)} cases, calibrated on {len(cal_df)}, "
          f"{len(demo)} demo cases held out ({demo[config.TARGET].sum()} csPCa).")
    print(f"Temperature T={model.temperature.T:.2f}; conformal q={bundle['conformal'].q_}; "
          f"threshold (95% sens) = {bundle['threshold_sens95']:.3f}")


if __name__ == "__main__":
    main()
