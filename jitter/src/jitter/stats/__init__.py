"""Statistics core for Jitter."""

from jitter.stats.autocorr import (
    AutocorrResult,
    effective_sample_size,
    geyer_initial_positive_sequence,
    lag1_autocorrelation,
    sample_autocorrelation,
)
from jitter.stats.bootstrap import (
    BootstrapResult,
    bootstrap_ci,
    politis_white_block_length,
)
from jitter.stats.compare import (
    CliffsMagnitude,
    ComparisonResult,
    block_permutation_test,
    cliffs_delta,
    compare_runs,
    hodges_lehmann_shift,
)
from jitter.stats.drift import (
    DriftResult,
    analyze_drift,
    mann_kendall_test,
    theil_sen_slope,
)
from jitter.stats.outliers import (
    OutlierResult,
    compute_modified_z_scores,
    detect_outliers,
)
from jitter.stats.summary import (
    RobustSummary,
    percentile_95,
    percentile_99,
    robust_summary,
    sample_median,
    scaled_mad,
)
from jitter.stats.warmup import (
    WarmupResult,
    analyze_warmup,
    calculate_warmup_slope,
    detect_cusum_changepoint,
    find_stable_baseline,
)

__all__ = [
    "sample_median",
    "scaled_mad",
    "percentile_95",
    "percentile_99",
    "robust_summary",
    "RobustSummary",
    "lag1_autocorrelation",
    "sample_autocorrelation",
    "geyer_initial_positive_sequence",
    "effective_sample_size",
    "AutocorrResult",
    "detect_cusum_changepoint",
    "find_stable_baseline",
    "calculate_warmup_slope",
    "analyze_warmup",
    "WarmupResult",
    "theil_sen_slope",
    "mann_kendall_test",
    "analyze_drift",
    "DriftResult",
    "politis_white_block_length",
    "bootstrap_ci",
    "BootstrapResult",
    "cliffs_delta",
    "CliffsMagnitude",
    "hodges_lehmann_shift",
    "block_permutation_test",
    "compare_runs",
    "ComparisonResult",
    "compute_modified_z_scores",
    "detect_outliers",
    "OutlierResult",
]
