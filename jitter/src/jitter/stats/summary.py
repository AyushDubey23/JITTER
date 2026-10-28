"""Robust non-parametric summary statistics for benchmark latency series.

Per Jitter specifications:
- Mean and standard deviation are forbidden due to asymmetric distributions, outliers, and heavy tails.
- Central tendency is measured via sample median.
- Scale/dispersion is measured via scaled Median Absolute Deviation (MAD, factor 1.4826).
- Tail percentiles (p95, p99) use linear interpolation method.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RobustSummary:
    """Non-parametric summary statistics of a latency sample."""

    n: int
    median: float
    mad: float
    p95: float
    p99: float


def _validate_input(x: Sequence[float] | np.ndarray) -> np.ndarray:
    """Validate and convert input to 1D float64 numpy array."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"Input must be 1-dimensional, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError("Cannot compute summary statistics for empty sample")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Sample contains non-finite values (NaN or Inf)")
    return arr


def sample_median(x: Sequence[float] | np.ndarray) -> float:
    """Calculate the sample median."""
    arr = _validate_input(x)
    return float(np.median(arr))


def scaled_mad(x: Sequence[float] | np.ndarray, scale: float = 1.4826) -> float:
    """Calculate the scaled Median Absolute Deviation (MAD).

    Formula: scale * median(|x - median(x)|)
    Default scale = 1.4826 matches normal distribution standard deviation asymptotically.
    """
    arr = _validate_input(x)
    med = np.median(arr)
    abs_deviations = np.abs(arr - med)
    return float(scale * np.median(abs_deviations))


def percentile_95(x: Sequence[float] | np.ndarray) -> float:
    """Calculate the 95th percentile using linear interpolation."""
    arr = _validate_input(x)
    return float(np.percentile(arr, 95.0, method="linear"))


def percentile_99(x: Sequence[float] | np.ndarray) -> float:
    """Calculate the 99th percentile using linear interpolation."""
    arr = _validate_input(x)
    return float(np.percentile(arr, 99.0, method="linear"))


def robust_summary(x: Sequence[float] | np.ndarray) -> RobustSummary:
    """Compute complete robust summary for a latency series."""
    arr = _validate_input(x)
    med = float(np.median(arr))
    mad_val = float(1.4826 * np.median(np.abs(arr - med)))
    p95_val = float(np.percentile(arr, 95.0, method="linear"))
    p99_val = float(np.percentile(arr, 99.0, method="linear"))
    return RobustSummary(
        n=int(arr.size),
        median=med,
        mad=mad_val,
        p95=p95_val,
        p99=p99_val,
    )
