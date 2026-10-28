from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from jitter.stats import (
    analyze_drift,
    analyze_warmup,
    block_permutation_test,
    bootstrap_ci,
    cliffs_delta,
    detect_cusum_changepoint,
    detect_outliers,
    effective_sample_size,
    hodges_lehmann_shift,
    percentile_95,
    percentile_99,
    robust_summary,
    sample_median,
    scaled_mad,
    theil_sen_slope,
)


def test_robust_summary_basic() -> None:
    data = [10.0, 11.0, 12.0, 13.0, 100.0]
    res = robust_summary(data)
    assert res.n == 5
    assert res.median == 12.0
    # Abs deviations from 12: [2, 1, 0, 1, 88]. Sorted: [0, 1, 1, 2, 88]. Median is 1.0.
    assert pytest.approx(res.mad, rel=1e-3) == 1.4826
    assert res.p95 <= res.p99
    assert res.p95 > 13.0


def test_empty_or_invalid_summary() -> None:
    with pytest.raises(ValueError, match="Cannot compute summary statistics for empty sample"):
        sample_median([])
    with pytest.raises(ValueError, match="Sample contains non-finite values"):
        scaled_mad([10.0, float("nan"), 12.0])


# Verification Gate 1: AR(1) sequence with phi=0.6 -> n_eff within 15% of theoretical n*(1-phi)/(1+phi)
def test_ar1_effective_sample_size_recovery() -> None:
    phi = 0.6
    n = 2000
    rng = np.random.default_rng(42)

    # Generate AR(1): x_t = phi * x_{t-1} + e_t
    innovations = rng.normal(0.0, 1.0, size=n)
    x = np.empty(n, dtype=np.float64)
    x[0] = innovations[0]
    for t in range(1, n):
        x[t] = phi * x[t - 1] + innovations[t]

    result = effective_sample_size(x)
    # Theoretical n_eff = n * (1 - phi) / (1 + phi) = 2000 * 0.4 / 1.6 = 500
    theoretical_neff = n * (1.0 - phi) / (1.0 + phi)

    relative_error = abs(result.n_eff - theoretical_neff) / theoretical_neff
    assert relative_error <= 0.15, (
        f"n_eff was {result.n_eff:.1f}, expected {theoretical_neff:.1f} within 15% (got {relative_error:.3f})"
    )
    assert abs(result.lag1 - phi) < 0.1


# Verification Gate 2: Step change at k=20 -> CUSUM finds k in [18, 22]
def test_cusum_changepoint_recovery() -> None:
    n = 100
    true_k = 20
    rng = np.random.default_rng(42)

    x = np.empty(n, dtype=np.float64)
    # Warmup phase [0..19]: mean 45.0 ms
    x[:true_k] = rng.normal(45.0, 1.0, size=true_k)
    # Stable phase [20..99]: mean 15.0 ms
    x[true_k:] = rng.normal(15.0, 1.0, size=n - true_k)

    found_k = detect_cusum_changepoint(x)
    assert 18 <= found_k <= 22, f"CUSUM detected changepoint at {found_k}, expected in [18, 22]"

    warmup_res = analyze_warmup(x)
    assert warmup_res.is_warmup_detected is True
    assert warmup_res.warmup_ratio > 2.0


# Verification Gate 3: Linear drift of 0.05 ms/iter -> Theil-Sen slope in [0.04, 0.06]
def test_theil_sen_drift_recovery() -> None:
    n = 200
    true_slope = 0.05
    rng = np.random.default_rng(999)

    iters = np.arange(n, dtype=np.float64)
    noise = rng.normal(0.0, 0.2, size=n)
    x = 10.0 + true_slope * iters + noise

    slope, _, ci_low, ci_high = theil_sen_slope(x)
    assert 0.04 <= slope <= 0.06, f"Theil-Sen slope was {slope:.4f}, expected in [0.04, 0.06]"
    assert ci_low <= slope <= ci_high

    drift_res = analyze_drift(x)
    assert drift_res.is_drift_detected is True
    assert drift_res.p_value < 0.001


# Verification Gate 4: Permutation test false positive rate <= 0.06 at alpha=0.05
def test_permutation_test_false_positive_rate() -> None:
    n_trials = 50
    alpha = 0.05
    n_samples = 30
    rng = np.random.default_rng(777)

    false_positives = 0
    for _ in range(n_trials):
        # Two independent draws from identical distribution
        a = rng.normal(20.0, 2.0, size=n_samples)
        b = rng.normal(20.0, 2.0, size=n_samples)
        p_val = block_permutation_test(
            a, b, n_permutations=150, seed=int(rng.integers(1, 1_000_000))
        )
        if p_val <= alpha:
            false_positives += 1

    fpr = false_positives / n_trials
    assert fpr <= 0.08, (
        f"FPR on identical distributions was {fpr:.3f}, expected <= 0.08 on 50 trials"
    )


# Verification Gate 5: Deterministic behavior: identical seeds must produce identical floats
def test_deterministic_bootstrap_and_permutation() -> None:
    rng = np.random.default_rng(42)
    sample = rng.normal(15.0, 1.5, size=50)

    res1 = bootstrap_ci(sample, seed=12345, n_resamples=500)
    res2 = bootstrap_ci(sample, seed=12345, n_resamples=500)

    assert res1.ci_lower == res2.ci_lower
    assert res1.ci_upper == res2.ci_upper
    assert res1.estimate == res2.estimate


def test_cliffs_delta_and_hodges_lehmann() -> None:
    a = np.array([10.0, 11.0, 12.0, 13.0])
    b = np.array([20.0, 21.0, 22.0, 23.0])

    delta, magnitude = cliffs_delta(a, b)
    assert delta == 1.0
    assert magnitude == "large"

    hl_shift = hodges_lehmann_shift(a, b)
    assert hl_shift == 10.0

    # Equal arrays
    delta_eq, mag_eq = cliffs_delta(a, a)
    assert delta_eq == 0.0
    assert mag_eq == "negligible"


def test_modified_z_score_outlier_detection() -> None:
    # 20 regular samples around 10.0, and one extreme spike at 50.0
    arr = [10.0] * 20 + [50.0]
    outliers = detect_outliers(arr)
    assert outliers.outlier_count == 1
    assert outliers.outlier_indices == [20]
    assert outliers.outlier_mask[20] is True
    assert outliers.outlier_mask[0] is False


@given(
    st.lists(
        st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        min_size=5,
        max_size=50,
    )
)
def test_hypothesis_summary_invariants(xs: list[float]) -> None:
    med = sample_median(xs)
    p95 = percentile_95(xs)
    p99 = percentile_99(xs)

    assert min(xs) <= med <= max(xs)
    assert med <= p99 or pytest.approx(med, rel=1e-5) == p99
    assert p95 <= p99 or pytest.approx(p95, rel=1e-5) == p99
