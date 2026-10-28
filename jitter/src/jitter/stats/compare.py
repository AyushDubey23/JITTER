"""Robust two-sample comparisons and hypothesis testing for benchmark runs.

Benchmarks often violate standard Gaussian assumptions. Two-sample t-tests
exhibit inflated false positive rates under heavy tails and autocorrelation.

This module provides:
- Block permutation test for difference of medians (preserving autocorrelation).
- Mann-Whitney U test (exact for small n, asymptotic with continuity correction for large).
- Hodges-Lehmann non-parametric estimator for shift magnitude.
- Cliff's delta effect size with Romano et al. (2006) magnitude thresholds:
    negligible: |delta| < 0.147
    small:      0.147 <= |delta| < 0.330
    medium:     0.330 <= |delta| < 0.474
    large:      |delta| >= 0.474
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import scipy.stats as sp_stats

from jitter.stats.bootstrap import politis_white_block_length

CliffsMagnitude = Literal["negligible", "small", "medium", "large"]


@dataclass(frozen=True)
class ComparisonResult:
    """Statistical summary of two-sample comparison."""

    diff_of_medians: float
    hodges_lehmann_shift: float
    cliffs_delta: float
    cliffs_magnitude: CliffsMagnitude
    permutation_p_value: float
    mann_whitney_u: float
    mann_whitney_p_value: float
    ci_lower: float
    ci_upper: float


def cliffs_delta(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
) -> tuple[float, CliffsMagnitude]:
    """Calculate Cliff's delta non-parametric effect size and Romano et al. (2006) qualitative magnitude.

    Delta = [#(b > a) - #(b < a)] / (n_a * n_b)
    Positive delta means b is generally greater than a.
    """
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)

    if arr_a.size == 0 or arr_b.size == 0:
        return 0.0, "negligible"

    # Vectorized outer comparison
    diff = np.subtract.outer(arr_b, arr_a)
    greater = np.count_nonzero(diff > 0)
    lesser = np.count_nonzero(diff < 0)
    total = arr_a.size * arr_b.size

    delta = float((greater - lesser) / total)
    abs_d = abs(delta)

    if abs_d < 0.147:
        magnitude: CliffsMagnitude = "negligible"
    elif abs_d < 0.330:
        magnitude = "small"
    elif abs_d < 0.474:
        magnitude = "medium"
    else:
        magnitude = "large"

    return delta, magnitude


def hodges_lehmann_shift(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
) -> float:
    """Calculate the two-sample Hodges-Lehmann shift estimator (median of all b_j - a_i)."""
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)

    if arr_a.size == 0 or arr_b.size == 0:
        return 0.0

    # If small enough, compute all pairwise differences exactly
    if arr_a.size * arr_b.size <= 2_000_000:
        diffs = np.subtract.outer(arr_b, arr_a).ravel()
        return float(np.median(diffs))

    # For very large samples, use a stratified random sample of 1,000,000 pairs
    rng = np.random.default_rng(42)
    idx_a = rng.integers(0, arr_a.size, size=1_000_000)
    idx_b = rng.integers(0, arr_b.size, size=1_000_000)
    diffs = arr_b[idx_b] - arr_a[idx_a]
    return float(np.median(diffs))


def block_permutation_test(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
    n_permutations: int = 2000,
    seed: int | None = 42,
    block_length: int | None = None,
) -> float:
    """Two-sample block permutation test for difference of medians.

    Partitions sequences into contiguous blocks of length `block_length`
    to account for serial dependence, then randomly permutes block assignments.
    Returns two-sided p-value.
    """
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    n_a = arr_a.size
    n_b = arr_b.size

    if n_a < 2 or n_b < 2:
        return 1.0

    obs_diff = abs(float(np.median(arr_b)) - float(np.median(arr_a)))

    # Determine common block length
    if block_length is None:
        b_a = politis_white_block_length(arr_a)
        b_b = politis_white_block_length(arr_b)
        blen = max(b_a, b_b)
    else:
        blen = max(1, block_length)

    rng = np.random.default_rng(seed)

    if blen == 1:
        # Standard i.i.d. permutation test
        pooled = np.concatenate([arr_a, arr_b])
        count = 0
        for _ in range(n_permutations):
            perm = rng.permutation(pooled)
            perm_a = perm[:n_a]
            perm_b = perm[n_a:]
            diff = abs(float(np.median(perm_b)) - float(np.median(perm_a)))
            if diff >= obs_diff:
                count += 1
        return float((count + 1) / (n_permutations + 1))

    # Divide arr_a and arr_b into non-overlapping blocks
    blocks_a = [arr_a[i : i + blen] for i in range(0, n_a - blen + 1, blen)]
    blocks_b = [arr_b[i : i + blen] for i in range(0, n_b - blen + 1, blen)]

    all_blocks = blocks_a + blocks_b
    total_blocks = len(all_blocks)
    n_blocks_a = len(blocks_a)

    if total_blocks < 2 or n_blocks_a == 0 or n_blocks_a == total_blocks:
        # Fall back to standard permutation if insufficient blocks
        pooled = np.concatenate([arr_a, arr_b])
        count = 0
        for _ in range(n_permutations):
            perm = rng.permutation(pooled)
            diff = abs(float(np.median(perm[n_a:])) - float(np.median(perm[:n_a])))
            if diff >= obs_diff:
                count += 1
        return float((count + 1) / (n_permutations + 1))

    count = 0
    for _ in range(n_permutations):
        perm_idx = rng.permutation(total_blocks)
        perm_blocks_a = [all_blocks[idx] for idx in perm_idx[:n_blocks_a]]
        perm_blocks_b = [all_blocks[idx] for idx in perm_idx[n_blocks_a:]]

        sample_a = np.concatenate(perm_blocks_a)
        sample_b = np.concatenate(perm_blocks_b)
        diff = abs(float(np.median(sample_b)) - float(np.median(sample_a)))
        if diff >= obs_diff:
            count += 1

    return float((count + 1) / (n_permutations + 1))


def compare_runs(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
    confidence_level: float = 0.95,
    n_permutations: int = 2000,
    seed: int | None = 42,
) -> ComparisonResult:
    """Execute complete robust comparison between baseline run 'a' and comparison run 'b'."""
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)

    med_a = float(np.median(arr_a))
    med_b = float(np.median(arr_b))
    diff_med = med_b - med_a

    # Hodges-Lehmann shift
    hl_shift = hodges_lehmann_shift(arr_a, arr_b)

    # Cliff's delta and magnitude
    delta, magnitude = cliffs_delta(arr_a, arr_b)

    # Mann-Whitney U test (handles small sample exact, large sample asymptotic)
    try:
        mw_res = sp_stats.mannwhitneyu(arr_b, arr_a, alternative="two-sided", method="auto")
        mw_u = float(mw_res.statistic)
        mw_p = float(mw_res.pvalue)
    except Exception:
        mw_u = 0.0
        mw_p = 1.0

    # Block permutation test
    perm_p = block_permutation_test(arr_a, arr_b, n_permutations=n_permutations, seed=seed)

    # 95% bootstrap CI for the difference of medians
    rng = np.random.default_rng(seed)
    diff_replicates = np.empty(n_permutations, dtype=np.float64)
    for i in range(n_permutations):
        idx_a = rng.integers(0, arr_a.size, size=arr_a.size)
        idx_b = rng.integers(0, arr_b.size, size=arr_b.size)
        diff_replicates[i] = np.median(arr_b[idx_b]) - np.median(arr_a[idx_a])

    alpha = 1.0 - confidence_level
    ci_low = float(np.percentile(diff_replicates, (alpha / 2.0) * 100.0))
    ci_high = float(np.percentile(diff_replicates, (1.0 - alpha / 2.0) * 100.0))

    return ComparisonResult(
        diff_of_medians=diff_med,
        hodges_lehmann_shift=hl_shift,
        cliffs_delta=delta,
        cliffs_magnitude=magnitude,
        permutation_p_value=perm_p,
        mann_whitney_u=mw_u,
        mann_whitney_p_value=mw_p,
        ci_lower=ci_low,
        ci_upper=ci_high,
    )
