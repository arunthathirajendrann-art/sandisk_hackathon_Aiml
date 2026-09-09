"""Probability Calibration module for SanDisk Yield AI.

Provides Platt scaling (sigmoid calibration) and Isotonic regression calibrators,
along with calibration curve statistics and Expected Calibration Error (ECE) metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def _logit(p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    p_clipped = np.clip(p, eps, 1.0 - eps)
    return np.log(p_clipped / (1.0 - p_clipped))


@dataclass
class PlattCalibrator:
    """Platt scaling (sigmoid calibration) fitted via Logistic Regression on probability log-odds."""

    C: float = 1.0
    random_state: int = 42
    model_: LogisticRegression | None = field(default=None, repr=False)

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> PlattCalibrator:
        z = _logit(y_prob).reshape(-1, 1)
        self.model_ = LogisticRegression(
            C=self.C, solver="lbfgs", max_iter=1000, random_state=self.random_state
        )
        self.model_.fit(z, y_true)
        return self

    def predict_proba(self, y_prob: np.ndarray) -> np.ndarray:
        z = _logit(y_prob).reshape(-1, 1)
        return self.model_.predict_proba(z)[:, 1]


@dataclass
class IsotonicCalibrator:
    """Non-parametric Isotonic Regression calibrator."""

    model_: IsotonicRegression | None = field(default=None, repr=False)

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> IsotonicCalibrator:
        self.model_ = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.model_.fit(y_prob, y_true)
        return self

    def predict_proba(self, y_prob: np.ndarray) -> np.ndarray:
        return self.model_.predict(y_prob)


def compute_calibration_curve(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> tuple[pd.DataFrame, float]:
    """Computes binned calibration curve statistics and Expected Calibration Error (ECE)."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    rows = []
    total_samples = len(y_true)
    ece = 0.0

    for b in range(n_bins):
        mask = bin_indices == b
        count = int(np.sum(mask))
        if count > 0:
            mean_pred = float(np.mean(y_prob[mask]))
            obs_rate = float(np.mean(y_true[mask]))
            ece += (count / total_samples) * abs(obs_rate - mean_pred)
        else:
            mean_pred = float(0.5 * (bins[b] + bins[b + 1]))
            obs_rate = 0.0

        rows.append({
            "bin": b + 1,
            "bin_min": float(bins[b]),
            "bin_max": float(bins[b + 1]),
            "mean_predicted_probability": mean_pred,
            "observed_failure_rate": obs_rate,
            "sample_count": count,
        })

    df_curve = pd.DataFrame(rows)
    return df_curve, float(ece)
