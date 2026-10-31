"""Calibrated decision thresholds for the Jitter diagnosis engine.

Threshold values are derived from empirical baseline distributions and verified
through the validation harness to maximize macro-F1 while maintaining calibrated confidence.
"""

from __future__ import annotations

# Minimum effective sample size required for conclusive diagnosis
MIN_EFFECTIVE_SAMPLE_SIZE: float = 10.0

# Warmup drift thresholds
WARMUP_RATIO_THRESHOLD: float = 1.50  # Cold initial median vs stable median ratio
WARMUP_MIN_DISCARD_ITERS: int = 1  # At least 1 discard iteration detected
WARMUP_STABLE_RATIO_CEILING: float = 1.15  # Warm run must not exhibit high warmup ratio

# Thermal throttling thresholds
THERMAL_THEIL_SEN_SLOPE: float = 0.02  # ms / iteration degradation rate
THERMAL_MANN_KENDALL_P: float = 0.01  # Statistically significant monotonic trend
THERMAL_MIN_STATUS: int = 1  # Reached at least LIGHT throttling
THERMAL_COOLDOWN_CLIFFS_DELTA: float = 0.474  # Large effect size between hot tail and rested run

# Delegate fallback thresholds
DELEGATE_CPU_SHIFT_MAX_RATIO: float = 0.10  # |shift| < 10% of CPU median implies parity
DELEGATE_CPU_CLIFFS_DELTA_CEILING: float = 0.147  # Negligible effect size vs CPU baseline

# Background contention noise thresholds
NOISE_MAD_RATIO_THRESHOLD: float = 0.30  # MAD > 30% of median indicates high variance
NOISE_MIN_OUTLIER_COUNT: int = 2  # At least 2 modified Z-score (>3.5) outliers
NOISE_QUIET_P99_REDUCTION_RATIO: float = 0.20  # Quiet mode reduces p99 by >= 20%
NOISE_QUIET_MEDIAN_SHIFT_TOLERANCE: float = 0.10  # Median remains within 10%

# Regression thresholds
REGRESSION_PERMUTATION_P: float = 0.01  # Statistically significant shift
REGRESSION_CLIFFS_DELTA_MIN: float = 0.330  # Medium or large effect size (>= 0.330)
REGRESSION_MIN_SHIFT_MS: float = 1.5  # Meaningful absolute shift
