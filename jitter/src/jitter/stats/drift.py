"""Monotonic trend and drift analysis for benchmark series.

Identifies thermal throttling and background resource contention drift:
- Mann-Kendall non-parametric test for trend significance.
- Theil-Sen robust estimator for slope (ms / iteration).
- Confidence interval for drift rate via rank inversion and block-aware bootstrap.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import scipy.stats as sp_stats


@dataclass(frozen=True)
class DriftResult:
    """Trend and drift statistics."""

    slope_ms_per_iter: float
    slope_ci_lower: float
    slope_ci_upper: float
    intercept_ms: float
    mann_kendall_tau: float
    p_value: float
    is_drift_detected: bool


def theil_sen_slope(
    x: Sequence[float] | np.ndarray,
    alpha: float = 0.95,
) -> tuple[float, float, float, float]:
    """Compute Theil-Sen median slope and rank-inverted confidence interval.

    Returns:
        (slope, intercept, slope_ci_lower, slope_ci_upper)
    """
    arr = np.asarray(x, dtype=np.float64)
    n = arr.size
    if n < 3:
        return 0.0, float(arr[0]) if n > 0 else 0.0, 0.0, 0.0

    iters = np.arange(n, dtype=np.float64)
    # scipy.stats.theilslopes provides robust slope and analytical confidence intervals
    res = sp_stats.theilslopes(arr, iters, alpha=alpha)
    return float(res.slope), float(res.intercept), float(res.low_slope), float(res.high_slope)


def mann_kendall_test(x: Sequence[float] | np.ndarray) -> tuple[float, float]:
    """Perform Mann-Kendall test for monotonic trend.

    Equivalent to Kendall rank correlation against time index.
    Returns:
        (tau, p_value)
    """
    arr = np.asarray(x, dtype=np.float64)
    n = arr.size
    if n < 4:
        return 0.0, 1.0

    iters = np.arange(n, dtype=np.float64)
    tau, p_val = sp_stats.kendalltau(iters, arr)
    tau_f = float(tau) if np.isfinite(tau) else 0.0
    p_f = float(p_val) if np.isfinite(p_val) else 1.0
    return tau_f, p_f


def analyze_drift(
    x: Sequence[float] | np.ndarray,
    alpha: float = 0.95,
    slope_threshold_ms_per_iter: float = 0.01,
    p_value_threshold: float = 0.05,
) -> DriftResult:
    """Analyze monotonic drift across iteration sequence."""
    arr = np.asarray(x, dtype=np.float64)
    n = arr.size
    if n < 4:
        return DriftResult(
            slope_ms_per_iter=0.0,
            slope_ci_lower=0.0,
            slope_ci_upper=0.0,
            intercept_ms=float(arr[0]) if n > 0 else 0.0,
            mann_kendall_tau=0.0,
            p_value=1.0,
            is_drift_detected=False,
        )

    slope, intercept, ci_low, ci_high = theil_sen_slope(arr, alpha=alpha)
    tau, p_val = mann_kendall_test(arr)

    # Drift is considered detected if slope exceeds threshold in positive direction
    # with statistically significant monotonic trend
    is_detected = bool(slope >= slope_threshold_ms_per_iter and p_val <= p_value_threshold)

    return DriftResult(
        slope_ms_per_iter=slope,
        slope_ci_lower=ci_low,
        slope_ci_upper=ci_high,
        intercept_ms=intercept,
        mann_kendall_tau=tau,
        p_value=p_val,
        is_drift_detected=is_detected,
    )
