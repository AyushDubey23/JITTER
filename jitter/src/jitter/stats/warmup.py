"""Warmup detection and changepoint analysis for benchmark runs.

On-device ML benchmarks typically suffer from cold-start anomalies:
- JIT compilation and graph optimization
- Weight loading into GPU/NPU memory
- Kernel tuning and cache warming

This module identifies:
1. Cumulative sum (CUSUM) changepoint to pinpoint the transition index.
2. Stable-run baseline: the earliest index k where adjacent windows of length W
   differ by no more than 1.0 * MAD.
3. Warmup slope (latency delta per iteration during the warmup phase).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from jitter.stats.summary import scaled_mad


@dataclass(frozen=True)
class WarmupResult:
    """Warmup detection and changepoint statistics."""

    discard_iterations: int
    cusum_changepoint: int
    stable_index: int
    warmup_slope: float
    warmup_ratio: float
    is_warmup_detected: bool


def detect_cusum_changepoint(x: Sequence[float] | np.ndarray) -> int:
    """Detect single changepoint index via Cumulative Sum (CUSUM) of mean deviations.

    Finds index k in [1, n-2] that maximizes |sum_{i=0}^k (x_i - mean(x))|.
    Returns 0 if sample size is too small.
    """
    arr = np.asarray(x, dtype=np.float64)
    n = arr.size
    if n < 4:
        return 0

    deviations = arr - np.mean(arr)
    cusum = np.cumsum(deviations)
    # Search within inner range [1, n-2] to avoid trivial boundary endpoints
    inner_cusum = np.abs(cusum[1:-1])
    if inner_cusum.size == 0:
        return 0
    best_idx = int(np.argmax(inner_cusum)) + 1
    return best_idx


def find_stable_baseline(
    x: Sequence[float] | np.ndarray,
    window_size: int | None = None,
    mad_threshold: float = 1.0,
) -> int:
    """Find first index k where median(x[k:k+W]) and median(x[k+W:k+2W]) differ by <= mad_threshold * MAD.

    Returns k (index of stable start, meaning k iterations should be discarded).
    """
    arr = np.asarray(x, dtype=np.float64)
    n = arr.size
    if n < 8:
        return 0

    w = window_size if window_size is not None else max(4, min(15, n // 10))
    if 2 * w > n:
        w = max(2, n // 4)

    total_mad = scaled_mad(arr)
    # If MAD is near zero, use a small absolute floor based on median
    threshold = max(total_mad * mad_threshold, 1e-4)

    for k in range(0, n - 2 * w + 1):
        w1 = arr[k : k + w]
        w2 = arr[k + w : k + 2 * w]
        diff = abs(float(np.median(w1)) - float(np.median(w2)))
        if diff <= threshold:
            return k

    return 0


def calculate_warmup_slope(x: Sequence[float] | np.ndarray, k: int) -> float:
    """Calculate slope (ms/iteration) over the warmup window [0:k].

    Returns 0.0 if k < 2.
    """
    arr = np.asarray(x, dtype=np.float64)
    if k < 2 or arr.size < 2:
        return 0.0
    warmup_slice = arr[: max(2, min(k + 1, arr.size))]
    iters = np.arange(warmup_slice.size, dtype=np.float64)
    # Simple linear fit slope
    cov = np.cov(iters, warmup_slice, bias=True)
    if cov[0, 0] <= 1e-12:
        return 0.0
    slope = float(cov[0, 1] / cov[0, 0])
    return slope


def analyze_warmup(
    x: Sequence[float] | np.ndarray,
    window_size: int | None = None,
    ratio_threshold: float = 1.25,
) -> WarmupResult:
    """Perform end-to-end warmup analysis."""
    arr = np.asarray(x, dtype=np.float64)
    n = arr.size
    if n < 4:
        return WarmupResult(
            discard_iterations=0,
            cusum_changepoint=0,
            stable_index=0,
            warmup_slope=0.0,
            warmup_ratio=1.0,
            is_warmup_detected=False,
        )

    cp = detect_cusum_changepoint(arr)
    stable_k = find_stable_baseline(arr, window_size=window_size)

    # Use the consensus discard recommendation
    discard_k = stable_k if stable_k > 0 else (cp if cp <= n // 2 else 0)

    # Compute ratio between initial segment and subsequent segment
    if discard_k > 0 and discard_k < n:
        med_init = float(np.median(arr[:discard_k]))
        med_stable = float(np.median(arr[discard_k:]))
        warmup_ratio = med_init / max(med_stable, 1e-9)
    else:
        # Fallback to inspecting first 10% vs remaining 90%
        split = max(1, n // 10)
        med_init = float(np.median(arr[:split]))
        med_stable = float(np.median(arr[split:]))
        warmup_ratio = med_init / max(med_stable, 1e-9)

    slope = calculate_warmup_slope(arr, max(discard_k, cp))
    is_detected = bool(warmup_ratio >= ratio_threshold and discard_k > 0)

    return WarmupResult(
        discard_iterations=discard_k,
        cusum_changepoint=cp,
        stable_index=stable_k,
        warmup_slope=slope,
        warmup_ratio=warmup_ratio,
        is_warmup_detected=is_detected,
    )
