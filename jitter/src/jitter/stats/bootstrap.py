"""Moving-block and stationary bootstrap for time series confidence intervals.

Because latency runs have serial correlation, i.i.d. bootstrap destroys temporal
dependence and produces severely understated confidence intervals.

This module implements:
- Automatic block length selection via Politis & White (2004) spectral method.
- Moving-block bootstrap with configurable resamples (default 10,000).
- Percentile confidence interval estimation.
- Bias-corrected and accelerated (BCa) bootstrap, strictly conditioned on lag-1 |rho_1| < 0.2.
- Deterministic seeding for strict reproducibility.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import scipy.stats as sp_stats

from jitter.stats.autocorr import lag1_autocorrelation, sample_autocorrelation


@dataclass(frozen=True)
class BootstrapResult:
    """Confidence interval and bootstrap estimation result."""

    estimate: float
    ci_lower: float
    ci_upper: float
    confidence_level: float
    method: str
    block_length: int
    n_resamples: int
    bca_used: bool


def politis_white_block_length(x: Sequence[float] | np.ndarray) -> int:
    """Select optimal block length b for moving-block bootstrap (Politis & White 2004).

    For stationary series with autocorrelation, balances variance and bias:
        b* ~ (2 * G^2 / D^2)^(1/3) * n^(1/3)
    Clamped to [1, max(1, int(sqrt(n)))].
    """
    arr = np.asarray(x, dtype=np.float64)
    n = arr.size
    if n < 4:
        return 1

    rho1 = abs(lag1_autocorrelation(arr))
    if rho1 < 0.05:
        # Effectively i.i.d.
        return 1

    # Approximate spectral density parameters via autocovariance sequence
    ac = sample_autocorrelation(arr, max_lag=min(n // 2, 50))
    # Sum of lags weighted by lag distance
    lags = np.arange(len(ac))
    g = float(np.sum(lags * np.abs(ac)))
    d = float(np.sum(np.abs(ac)))

    if d <= 1e-12:
        return 1

    ratio = (2.0 * (g**2)) / max(d**2, 1e-9)
    b_opt = int(np.round((ratio ** (1.0 / 3.0)) * (n ** (1.0 / 3.0))))

    max_b = max(1, int(np.sqrt(n)))
    return int(np.clip(b_opt, 1, max_b))


def generate_moving_blocks(
    arr: np.ndarray,
    block_length: int,
    n_resamples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample n_resamples series of shape (n_resamples, n) via moving-block bootstrap."""
    n = arr.size
    b = max(1, min(block_length, n))
    if b == 1:
        # Standard i.i.d. bootstrap
        indices = rng.integers(0, n, size=(n_resamples, n))
        return arr[indices]

    n_blocks_needed = int(np.ceil(n / b))
    max_start = n - b + 1
    # Random block start indices: shape (n_resamples, n_blocks_needed)
    starts = rng.integers(0, max_start, size=(n_resamples, n_blocks_needed))

    # Construct block resamples
    offsets = np.arange(b)
    # Shape: (n_resamples, n_blocks_needed, b)
    block_indices = starts[:, :, np.newaxis] + offsets[np.newaxis, np.newaxis, :]
    # Flatten to (n_resamples, n_blocks_needed * b) and slice to length n
    flat_indices = block_indices.reshape(n_resamples, -1)[:, :n]
    return np.asarray(arr[flat_indices], dtype=np.float64)


def compute_bca_interval(
    arr: np.ndarray,
    replicates: np.ndarray,
    stat_fn: Callable[[np.ndarray], float],
    point_estimate: float,
    alpha: float,
) -> tuple[float, float]:
    """Compute BCa (Bias-Corrected and Accelerated) confidence interval."""
    n = arr.size

    # Bias correction z0
    prop_less = float(np.mean(replicates < point_estimate))
    prop_less = np.clip(prop_less, 1e-6, 1.0 - 1e-6)
    z0 = float(sp_stats.norm.ppf(prop_less))

    # Acceleration a via jackknife
    jack_stats = np.empty(n, dtype=np.float64)
    for i in range(n):
        jack_sample = np.delete(arr, i)
        jack_stats[i] = stat_fn(jack_sample)

    jack_mean = float(np.mean(jack_stats))
    diff = jack_mean - jack_stats
    num = float(np.sum(diff**3))
    den = float(6.0 * (np.sum(diff**2) ** 1.5))
    a = (num / den) if den > 1e-12 else 0.0
    a = float(np.clip(a, -0.5, 0.5))

    # Adjusted percentiles
    z_alpha_low = float(sp_stats.norm.ppf(alpha / 2.0))
    z_alpha_high = float(sp_stats.norm.ppf(1.0 - alpha / 2.0))

    def bca_pct(z_val: float) -> float:
        denom = 1.0 - a * (z0 + z_val)
        if abs(denom) < 1e-6:
            denom = 1e-6
        arg = z0 + (z0 + z_val) / denom
        return float(sp_stats.norm.cdf(arg) * 100.0)

    pct_low = np.clip(bca_pct(z_alpha_low), 0.01, 99.9)
    pct_high = np.clip(bca_pct(z_alpha_high), 0.1, 99.99)

    if pct_low >= pct_high:
        pct_low = (alpha / 2.0) * 100.0
        pct_high = (1.0 - alpha / 2.0) * 100.0

    ci_low = float(np.percentile(replicates, pct_low))
    ci_high = float(np.percentile(replicates, pct_high))
    return ci_low, ci_high


def bootstrap_ci(
    x: Sequence[float] | np.ndarray,
    stat_fn: Callable[[np.ndarray], float] = np.median,
    confidence_level: float = 0.95,
    n_resamples: int = 10_000,
    seed: int | None = 42,
    block_length: int | None = None,
) -> BootstrapResult:
    """Compute moving-block bootstrap confidence interval for a given statistic."""
    arr = np.asarray(x, dtype=np.float64)
    n = arr.size
    if n == 0:
        raise ValueError("Cannot compute bootstrap on empty sample")
    if n == 1:
        val = float(arr[0])
        return BootstrapResult(
            estimate=val,
            ci_lower=val,
            ci_upper=val,
            confidence_level=confidence_level,
            method="single_sample",
            block_length=1,
            n_resamples=1,
            bca_used=False,
        )

    rng = np.random.default_rng(seed)
    b = block_length if block_length is not None else politis_white_block_length(arr)
    point_est = float(stat_fn(arr))

    # Generate resamples
    resamples = generate_moving_blocks(arr, b, n_resamples, rng)
    replicates = np.apply_along_axis(stat_fn, axis=1, arr=resamples)

    alpha = 1.0 - confidence_level
    lag1 = abs(lag1_autocorrelation(arr))

    # BCa is conditioned on lag-1 |rho_1| < 0.2 and block_length == 1
    # because BCa assumptions break under strong serial correlation
    use_bca = bool(lag1 < 0.2 and b == 1 and n >= 8)

    if use_bca:
        try:
            ci_low, ci_high = compute_bca_interval(arr, replicates, stat_fn, point_est, alpha)
            method = "bca"
        except Exception:
            ci_low = float(np.percentile(replicates, (alpha / 2.0) * 100.0))
            ci_high = float(np.percentile(replicates, (1.0 - alpha / 2.0) * 100.0))
            method = "moving_block_percentile"
            use_bca = False
    else:
        ci_low = float(np.percentile(replicates, (alpha / 2.0) * 100.0))
        ci_high = float(np.percentile(replicates, (1.0 - alpha / 2.0) * 100.0))
        method = "moving_block_percentile"

    return BootstrapResult(
        estimate=point_est,
        ci_lower=ci_low,
        ci_upper=ci_high,
        confidence_level=confidence_level,
        method=method,
        block_length=b,
        n_resamples=n_resamples,
        bca_used=use_bca,
    )
