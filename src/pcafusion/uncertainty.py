"""Calibration and conformal prediction utilities."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

LABELS = ("no csPCa", "csPCa")


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


class TemperatureScaler:
    """Single-parameter post-hoc calibration of logits (Guo et al., 2017)."""

    def __init__(self):
        self.T = 1.0

    def fit(self, logits: np.ndarray, y: np.ndarray) -> "TemperatureScaler":
        def nll(t):
            p = np.clip(sigmoid(logits / t), 1e-7, 1 - 1e-7)
            return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

        self.T = float(minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded").x)
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        return sigmoid(logits / self.T)


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        sel = idx == b
        if sel.any():
            ece += sel.mean() * abs(y[sel].mean() - p[sel].mean())
    return float(ece)


class ClassConditionalConformal:
    """Mondrian (class-conditional) split conformal classifier for binary outcomes.

    With calibration data exchangeable with test data, the prediction set
    contains the true label with probability >= 1 - alpha *separately for each
    class*. For the positive class this is a distribution-free guarantee on
    sensitivity, which matters more clinically than marginal coverage when the
    classes are imbalanced.

    Prediction sets:  {0} confident negative, {1} confident positive,
                      {0, 1} ambiguous -> defer to clinician, {} atypical case.
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.q_: dict[int, float] = {}

    def fit(self, p_cal: np.ndarray, y_cal: np.ndarray) -> "ClassConditionalConformal":
        for k in (0, 1):
            pk = p_cal[y_cal == k] if k == 1 else 1 - p_cal[y_cal == k]
            scores = 1.0 - pk  # nonconformity: 1 - probability of the true class
            n = len(scores)
            level = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)
            self.q_[k] = float(np.quantile(scores, level, method="higher"))
        return self

    def predict_sets(self, p: np.ndarray) -> np.ndarray:
        """Return (n, 2) boolean array: column k is True if label k is in the set."""
        return np.stack([p <= self.q_[0], (1 - p) <= self.q_[1]], axis=1)

    @staticmethod
    def describe(row: np.ndarray) -> str:
        if row.all():
            return "ambiguous"
        if row[1]:
            return "csPCa"
        if row[0]:
            return "no csPCa"
        return "atypical"
