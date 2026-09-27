"""Evaluation metrics with uncertainty (bootstrap CIs) and clinical utility."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from .uncertainty import expected_calibration_error


def bootstrap_ci(metric, y: np.ndarray, p: np.ndarray, n_boot: int = 1000, seed: int = 0,
                 groups: np.ndarray | None = None) -> tuple[float, float, float]:
    """Point estimate and percentile 95% CI; resamples patients if `groups` is given."""
    rng = np.random.default_rng(seed)
    point = metric(y, p)
    if groups is None:
        groups = np.arange(len(y))
    uniq = np.unique(groups)
    index = {g: np.flatnonzero(groups == g) for g in uniq}
    stats = []
    for _ in range(n_boot):
        idx = np.concatenate([index[g] for g in rng.choice(uniq, len(uniq))])
        if len(np.unique(y[idx])) < 2:
            continue
        stats.append(metric(y[idx], p[idx]))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def summarize(y: np.ndarray, p: np.ndarray, groups: np.ndarray | None = None, n_boot: int = 1000,
              probabilistic: bool = True) -> dict:
    """Discrimination with bootstrap CI; calibration metrics only if `p` is a probability."""
    auc, lo, hi = bootstrap_ci(roc_auc_score, y, p, n_boot=n_boot, groups=groups)
    if not probabilistic:
        return {"n": int(len(y)), "prevalence": float(y.mean()), "auroc": auc, "auroc_lo": lo,
                "auroc_hi": hi, "auprc": float(average_precision_score(y, p)), "brier": np.nan, "ece": np.nan}
    return {
        "n": int(len(y)),
        "prevalence": float(y.mean()),
        "auroc": auc, "auroc_lo": lo, "auroc_hi": hi,
        "auprc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "ece": expected_calibration_error(y, p),
    }


def threshold_for_sensitivity(y: np.ndarray, p: np.ndarray, target: float = 0.95) -> float:
    """Largest threshold whose sensitivity on (y, p) is >= target."""
    pos = np.sort(p[y == 1])
    k = int(np.floor((1 - target) * len(pos)))
    return float(pos[k]) if len(pos) else 0.5


def operating_point(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    pred = p >= thr
    tp = int((pred & (y == 1)).sum())
    fn = int((~pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum())
    fp = int((pred & (y == 0)).sum())
    return {
        "threshold": thr,
        "sensitivity": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "biopsies_avoided_frac": float((~pred).mean()),
        "missed_cspca": fn,
        "missed_cspca_frac_of_all_cases": fn / len(y),
    }


def net_benefit(y: np.ndarray, p: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Decision-curve net benefit (Vickers & Elkin, 2006)."""
    n = len(y)
    out = []
    for t in thresholds:
        pred = p >= t
        tp = (pred & (y == 1)).sum() / n
        fp = (pred & (y == 0)).sum() / n
        out.append(tp - fp * t / (1 - t))
    return np.array(out)
