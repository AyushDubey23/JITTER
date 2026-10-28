"""Autocorrelation and effective sample size (n_eff) calculation.

Benchmarks exhibit strong serial correlation due to OS thread scheduling,
frequency governor hysteresis, memory caching, and progressive heating.
Naive standard error and sample size severely overstate statistical power.

This module implements:
- Sample autocorrelation sequence estimation via FFT / direct computation.
- Lag-1 autocorrelation (rho_1).
- Truncation lag selection using Geyer's Initial Positive Sequence estimator (1992).
- Effective sample size:
    n_eff = n / (1 + 2 * sum_{k=1}^M rho_k)
- Low confidence warning flag when n_eff < 30.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AutocorrResult:
    """Autocorrelation and effective sample size metrics."""

    n: int
    lag1: float
    truncation_lag: int
    integrated_autocorr_time: float
    n_eff: float
    is_low_confidence: bool
    autocorrelations: list[float]


def sample_autocorrelation(
    x: Sequence[float] | np.ndarray, max_lag: int | None = None
) -> np.ndarray:
    """Compute sample autocorrelation sequence rho_k for lags 0 to max_lag.

    Uses zero-padded FFT for O(n log n) efficiency.
    """
    arr = np.asarray(x, dtype=np.float64)
    n = arr.size
    if n < 2:
        return np.array([1.0], dtype=np.float64)

    # Center by sample mean
    centered = arr - np.mean(arr)
    var = np.dot(centered, centered)
    if var <= 1e-15:
        # Constant sequence: autocorrelation is 0 for lag > 0
        res = np.zeros(n if max_lag is None else max_lag + 1, dtype=np.float64)
        res[0] = 1.0
        return res

    # Zero-padded FFT
    n_fft = 2 ** int(np.ceil(np.log2(2 * n - 1)))
    fx = np.fft.fft(centered, n_fft)
    autocov = np.fft.ifft(fx * np.conj(fx)).real[:n]
    autocorr = autocov / var

    if max_lag is not None:
        max_lag = min(max_lag, n - 1)
        return np.asarray(autocorr[: max_lag + 1], dtype=np.float64)
    return np.asarray(autocorr, dtype=np.float64)


def lag1_autocorrelation(x: Sequence[float] | np.ndarray) -> float:
    """Calculate lag-1 autocorrelation rho_1."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.size < 2:
        return 0.0
    ac = sample_autocorrelation(arr, max_lag=1)
    return float(ac[1]) if len(ac) > 1 else 0.0


def geyer_initial_positive_sequence(autocorr: np.ndarray) -> int:
    """Determine truncation lag M using Geyer's (1992) Initial Positive Sequence estimator.

    Pairs Gamma_m = rho_{2m} + rho_{2m+1} are inspected for m = 0, 1, 2, ...
    The sequence is truncated at M = 2K - 1, where K is the first index where Gamma_K <= 0.
    """
    n = len(autocorr)
    if n < 2:
        return 0

    max_pairs = (n - 1) // 2
    for m in range(max_pairs):
        pair_sum = autocorr[2 * m] + autocorr[2 * m + 1]
        if pair_sum <= 0:
            # First non-positive pair at m; truncate at previous pair (2m - 1)
            return max(0, 2 * m - 1)

    # If all inspected pairs were positive, truncate at last valid odd lag
    return max(0, 2 * max_pairs - 1)


def effective_sample_size(
    x: Sequence[float] | np.ndarray,
    min_neff_threshold: float = 30.0,
) -> AutocorrResult:
    """Calculate effective sample size n_eff accounting for autocorrelation.

    Formula:
        tau = 1 + 2 * sum_{k=1}^M rho_k
        n_eff = n / tau
    where M is determined by Geyer's Initial Positive Sequence estimator.
    """
    arr = np.asarray(x, dtype=np.float64)
    n = int(arr.size)
    if n == 0:
        raise ValueError("Cannot compute effective sample size for empty sample")
    if n < 4:
        return AutocorrResult(
            n=n,
            lag1=0.0,
            truncation_lag=0,
            integrated_autocorr_time=1.0,
            n_eff=float(n),
            is_low_confidence=True,
            autocorrelations=[1.0],
        )

    ac = sample_autocorrelation(arr)
    lag1 = float(ac[1])

    # Truncation lag M via Geyer's initial positive sequence
    m_lag = geyer_initial_positive_sequence(ac)

    if m_lag > 0:
        # Sum autocorrelations up to lag M
        sum_rho = float(np.sum(ac[1 : m_lag + 1]))
        tau = 1.0 + 2.0 * sum_rho
    else:
        tau = 1.0

    # tau cannot be physically less than 1.0 for positively autocorrelated benchmarks
    # but in anti-persistent sequences it could be smaller than 1. We floor at 0.1 to avoid div-zero.
    tau = max(0.1, tau)
    n_eff = float(n) / tau

    # n_eff cannot exceed nominal sample size n
    n_eff = float(np.clip(n_eff, 1.0, float(n)))

    return AutocorrResult(
        n=n,
        lag1=lag1,
        truncation_lag=m_lag,
        integrated_autocorr_time=tau,
        n_eff=n_eff,
        is_low_confidence=(n_eff < min_neff_threshold),
        autocorrelations=[float(v) for v in ac[: min(50, len(ac))]],
    )
