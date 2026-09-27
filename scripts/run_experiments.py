"""Full experimental protocol.

1. Internal validation: 5-fold cross-validation, stratified and grouped by patient.
   Inside each training fold, 20% of patients are held out as a calibration set
   (early stopping, temperature scaling, conformal calibration, threshold choice).
2. Missing-modality robustness: every trained model is re-evaluated with whole
   modalities removed at test time.
3. Transferability: leave-one-center-out validation (RUMC / PCNN / ZGT).
4. Uncertainty: calibration (ECE, reliability) and class-conditional conformal
   prediction; clinical utility with decision curves on the biopsied subset.

Usage:  python scripts/run_experiments.py [--quick]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pcafusion import config, plots  # noqa: E402
from pcafusion.baselines import (  # noqa: E402
    PiradsReference, clinical_logistic, early_fusion_gbdt, early_fusion_logistic,
)
from pcafusion.data import load_dataset  # noqa: E402
from pcafusion.evaluation import (  # noqa: E402
    net_benefit, operating_point, summarize, threshold_for_sensitivity,
)
from pcafusion.models import FusionEnsemble  # noqa: E402
from pcafusion.uncertainty import ClassConditionalConformal  # noqa: E402

# Spurious matmul warnings from numpy 2.0 + Apple Accelerate (numpy/numpy#26669).
warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

FUSION = "Fusion ensemble (ours)"
SCENARIOS: dict[str, tuple[str, ...]] = {
    "All": (),
    "No imaging": ("imaging",),
    "No radiology": ("radiology",),
    "No clinical": ("clinical",),
    "Clinical only": ("radiology", "imaging"),
}


def model_factories(n_members: int, epochs: int):
    return {
        "PI-RADS (radiologist)": PiradsReference,
        "Clinical LR": clinical_logistic,
        "Early-fusion LR (mean imputation)": early_fusion_logistic,
        "Early-fusion GBDT": early_fusion_gbdt,
        FUSION: lambda: FusionEnsemble(n_members=n_members, epochs=epochs),
    }


def inner_split(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    fit_idx, cal_idx = next(sgkf.split(df, df[config.TARGET], df[config.GROUP]))
    return df.iloc[fit_idx], df.iloc[cal_idx]


def run_fold(train: pd.DataFrame, test: pd.DataFrame, factories, seed: int) -> dict:
    fit_df, cal_df = inner_split(train, seed)
    out = {"test_index": test.index.to_numpy(), "pred": {}, "cal": {}}
    for name, make in factories.items():
        model = make().fit(fit_df, cal_df)
        out["pred"][name] = {s: model.predict_proba(test, drop) for s, drop in SCENARIOS.items()}
        out["cal"][name] = model.predict_proba(cal_df)
        if name == FUSION:
            fp = model.predict(test)
            out["fusion_std"] = fp.epistemic_std
            out["fusion_attention"] = fp.attention
            out["modality_names"] = fp.modality_names
    out["cal_y"] = cal_df[config.TARGET].to_numpy()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fewer ensemble members/epochs for a smoke test")
    args = ap.parse_args()
    n_members, epochs, n_boot = (2, 30, 200) if args.quick else (5, 200, 1000)

    torch.set_num_threads(4)
    for d in (config.FIGURES_DIR, config.RESULTS_DIR, config.MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    df = load_dataset().reset_index(drop=True)
    y = df[config.TARGET].to_numpy()
    groups = df[config.GROUP].to_numpy()
    biopsied = df["biopsied"].to_numpy()
    factories = model_factories(n_members, epochs)
    names = list(factories)
    t0 = time.time()

    # ------------------------------------------------------------------ 1. internal CV
    oof = {n: {s: np.zeros(len(df)) for s in SCENARIOS} for n in names}
    fusion_std = np.zeros(len(df))
    fusion_att = np.zeros((len(df), len(config.MODALITIES)))
    thr95 = np.zeros(len(df))
    conf_sets = {}
    alphas = np.array([0.30, 0.20, 0.15, 0.10, 0.05])
    for a in alphas:
        conf_sets[a] = np.zeros((len(df), 2), dtype=bool)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=config.SEED)
    for k, (tr, te) in enumerate(sgkf.split(df, y, groups)):
        res = run_fold(df.iloc[tr], df.iloc[te], factories, seed=config.SEED + k)
        for n in names:
            for s in SCENARIOS:
                oof[n][s][te] = res["pred"][n][s]
        fusion_std[te] = res["fusion_std"]
        fusion_att[te] = res["fusion_attention"]
        cal_p, cal_y = res["cal"][FUSION], res["cal_y"]
        thr95[te] = threshold_for_sensitivity(cal_y, cal_p, 0.95)
        for a in alphas:
            conf_sets[a][te] = ClassConditionalConformal(a).fit(cal_p, cal_y).predict_sets(oof[FUSION]["All"][te])
        print(f"[CV] fold {k + 1}/5 done  ({time.time() - t0:.0f}s)")

    rows = []
    for subset, mask in (("All cases", np.ones(len(df), bool)), ("Biopsied only", biopsied)):
        for n in names:
            s = summarize(y[mask], oof[n]["All"][mask], groups[mask], n_boot=n_boot,
                          probabilistic=n != "PI-RADS (radiologist)")
            rows.append({"subset": subset, "model": n, **s})
    cv_table = pd.DataFrame(rows)
    cv_table.to_csv(config.RESULTS_DIR / "internal_cv.csv", index=False)
    print(cv_table.round(3).to_string(index=False))

    # ROC figure
    roc_models = [FUSION, "Early-fusion GBDT", "Clinical LR", "PI-RADS (radiologist)"]
    panels = []
    for subset, mask in (("All cases (n=1500)", np.ones(len(df), bool)), (f"Biopsied only (n={biopsied.sum()})", biopsied)):
        key = "All cases" if subset.startswith("All") else "Biopsied only"
        summ = {r["model"]: r for r in rows if r["subset"] == key}
        panels.append((subset, y[mask], {n: oof[n]["All"][mask] for n in roc_models}, summ))
    plots.roc_panels(panels, config.FIGURES_DIR / "roc_internal_cv.png")

    # Calibration figure
    cal_models = [FUSION, "Early-fusion GBDT", "Clinical LR"]
    plots.reliability(y, {n: oof[n]["All"] for n in cal_models},
                      {n: cv_table.query("subset=='All cases' and model==@n").ece.item() for n in cal_models},
                      config.FIGURES_DIR / "calibration.png")

    # ------------------------------------------------------------------ 2. missing modalities
    mm = pd.DataFrame(
        {n: [roc_auc_score(y, oof[n][s]) for s in SCENARIOS] for n in names if n != "PI-RADS (radiologist)"},
        index=list(SCENARIOS),
    )
    mm.to_csv(config.RESULTS_DIR / "missing_modality_auroc.csv")
    print("\nAUROC under missing modalities\n", mm.round(3))
    plots.missing_modality_bars(mm[[FUSION, "Early-fusion GBDT", "Early-fusion LR (mean imputation)"]],
                                config.FIGURES_DIR / "missing_modalities.png")

    # ------------------------------------------------------------------ 3. uncertainty
    conf_rows = []
    for a in alphas:
        sets = conf_sets[a]
        conf_rows.append({
            "alpha": a,
            "coverage_cspca": sets[y == 1, 1].mean(),
            "coverage_non_cspca": sets[y == 0, 0].mean(),
            "ambiguous_frac": sets.all(1).mean(),
            "confident_negative_frac": (sets[:, 0] & ~sets[:, 1]).mean(),
            "confident_positive_frac": (sets[:, 1] & ~sets[:, 0]).mean(),
            "empty_frac": (~sets.any(1)).mean(),
        })
    conf_table = pd.DataFrame(conf_rows)
    conf_table.to_csv(config.RESULTS_DIR / "conformal.csv", index=False)
    print("\nConformal prediction\n", conf_table.round(3).to_string(index=False))
    plots.conformal_curve(alphas, conf_table.coverage_cspca.to_numpy(), conf_table.coverage_non_cspca.to_numpy(),
                          conf_table.ambiguous_frac.to_numpy(), config.FIGURES_DIR / "conformal.png")

    # Does epistemic uncertainty flag errors? (AUROC of std for detecting wrong confident calls)
    err = (oof[FUSION]["All"] >= 0.5).astype(int) != y
    unc = {"epistemic_std_auroc_for_errors": float(roc_auc_score(err, fusion_std)),
           "ambiguous_set_error_rate": float(err[conf_sets[0.10].all(1)].mean()),
           "confident_set_error_rate": float(err[~conf_sets[0.10].all(1)].mean())}
    print("\nUncertainty vs error\n", unc)

    # ------------------------------------------------------------------ 4. clinical utility (biopsied subset)
    yb, pb = y[biopsied], oof[FUSION]["All"][biopsied]
    pir = df.loc[biopsied, "pirads_max"].to_numpy()
    ops = []
    pred_fusion = pb >= thr95[biopsied]
    ops.append({"strategy": "Fusion model, target sensitivity 95% (calibration-set threshold)",
                **operating_point(yb, pred_fusion.astype(float), 0.5)})
    ops.append({"strategy": "PI-RADS >= 3", **operating_point(yb, pir, 3)})
    ops.append({"strategy": "PI-RADS >= 4", **operating_point(yb, pir, 4)})
    psad = df.loc[biopsied, "psad_reported"].fillna(df.loc[biopsied, "psa"] / df.loc[biopsied, "gland_volume_seg"]).to_numpy()
    rule = (pir >= 4) | ((pir == 3) & (psad >= 0.15))
    ops.append({"strategy": "PI-RADS >= 4, or PI-RADS 3 with PSAD >= 0.15", **operating_point(yb, rule.astype(float), 0.5)})
    ops_table = pd.DataFrame(ops).drop(columns="threshold")
    ops_table.to_csv(config.RESULTS_DIR / "operating_points_biopsied.csv", index=False)
    print("\nBiopsy decision strategies (biopsied subset)\n", ops_table.round(3).to_string(index=False))

    ts = np.linspace(0.02, 0.5, 49)
    prev = yb.mean()
    curves = {
        FUSION: net_benefit(yb, pb, ts),
        "Early-fusion GBDT": net_benefit(yb, oof["Early-fusion GBDT"]["All"][biopsied], ts),
        "Biopsy all": prev - (1 - prev) * ts / (1 - ts),
        "Biopsy none": np.zeros_like(ts),
    }
    plots.decision_curve(ts, curves, config.FIGURES_DIR / "decision_curve.png",
                         "Decision curve (biopsied subset)")

    # Attention by PI-RADS group
    grp = pd.cut(df["pirads_max"], [0, 2, 3, 5], labels=["PI-RADS 1–2", "PI-RADS 3", "PI-RADS 4–5"])
    att = np.stack([fusion_att[(grp == g).to_numpy()].mean(0) for g in grp.cat.categories])
    plots.attention_by_group(list(grp.cat.categories), att, list(config.MODALITIES),
                             config.FIGURES_DIR / "attention_by_pirads.png")

    # ------------------------------------------------------------------ 5. leave-one-center-out
    loco_rows = []
    for center in sorted(df[config.CENTER].unique()):
        te = (df[config.CENTER] == center).to_numpy()
        res = run_fold(df[~te], df[te], factories, seed=config.SEED)
        for n in names:
            p = res["pred"][n]["All"]
            s = summarize(y[te], p, groups[te], n_boot=n_boot, probabilistic=n != "PI-RADS (radiologist)")
            loco_rows.append({"held_out_center": center, "model": n, **s})
        print(f"[LOCO] {center} done ({time.time() - t0:.0f}s)")
    loco = pd.DataFrame(loco_rows)
    loco.to_csv(config.RESULTS_DIR / "leave_one_center_out.csv", index=False)
    print("\nLeave-one-center-out AUROC\n",
          loco.pivot(index="model", columns="held_out_center", values="auroc").round(3))

    summary = {
        "internal_cv": cv_table.round(4).to_dict(orient="records"),
        "missing_modality_auroc": mm.round(4).to_dict(),
        "conformal": conf_table.round(4).to_dict(orient="records"),
        "uncertainty_vs_error": unc,
        "operating_points_biopsied": ops_table.round(4).to_dict(orient="records"),
        "leave_one_center_out": loco.round(4).to_dict(orient="records"),
        "runtime_s": round(time.time() - t0, 1),
    }
    (config.RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nDone in {time.time() - t0:.0f}s. Results in {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
