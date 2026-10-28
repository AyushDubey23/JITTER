"""Outlier detection using Iglewicz and Hoaglin (1993) modified Z-score.

Standard Z-scores fail on benchmark data because outliers heavily distort
the sample mean and sample variance. The modified Z-score uses the median
and Median Absolute Deviation (MAD), achieving high breakdown robustness.

Per Jitter specifications:
- Threshold is fixed at 3.5.
- Outliers are NEVER silently discarded from latency series.
- Outliers are identified and tagged so diagnostic engines can inspect burst noise.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OutlierResult:
    """Outlier detection flags and modified Z-scores."""

    outlier_mask: list[bool]
    outlier_count: int
    outlier_indices: list[int]
    modified_z_scores: list[float]
    threshold: float


def compute_modified_z_scores(x: Sequence[float] | np.ndarray) -> np.ndarray:
    """Compute Iglewicz & Hoaglin (1993) modified Z-scores:

    M_i = 0.6745 * (x_i - median(x)) / MAD
    where MAD = median(|x - median(x)|).
    """
    arr = np.asarray(x, dtype=np.float64)
    n = arr.size
    if n == 0:
        return np.array([], dtype=np.float64)
    if n < 3:
        return np.zeros(n, dtype=np.float64)

    med = np.median(arr)
    abs_deviations = np.abs(arr - med)
    raw_mad = np.median(abs_deviations)

    if raw_mad > 1e-12:
        return np.asarray(0.6745 * (arr - med) / raw_mad, dtype=np.float64)

    # If MAD is zero (e.g. over 50% identical values), fall back to mean absolute deviation
    mean_ad = np.mean(abs_deviations)
    if mean_ad > 1e-12:
        return np.asarray(0.6745 * (arr - med) / (1.253314 * mean_ad), dtype=np.float64)

    # Completely identical sample
    return np.zeros(n, dtype=np.float64)


def detect_outliers(
    x: Sequence[float] | np.ndarray,
    threshold: float = 3.5,
) -> OutlierResult:
    """Identify outliers where |M_i| > threshold (default 3.5)."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.size == 0:
        return OutlierResult(
            outlier_mask=[],
            outlier_count=0,
            outlier_indices=[],
            modified_z_scores=[],
            threshold=threshold,
        )

    z_scores = compute_modified_z_scores(arr)
    mask = np.abs(z_scores) > threshold
    indices = [int(i) for i in np.flatnonzero(mask)]

    return OutlierResult(
        outlier_mask=[bool(b) for b in mask],
        outlier_count=len(indices),
        outlier_indices=indices,
        modified_z_scores=[float(z) for z in z_scores],
        threshold=threshold,
    )
