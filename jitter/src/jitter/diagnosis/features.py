"""Diagnostic feature extraction across benchmark reruns and telemetry."""

from __future__ import annotations

from dataclasses import dataclass

from jitter.schema.report import Report
from jitter.schema.runs import ProcessStart, RerunType, RunEntry
from jitter.stats.autocorr import effective_sample_size
from jitter.stats.compare import compare_runs
from jitter.stats.drift import analyze_drift
from jitter.stats.outliers import detect_outliers
from jitter.stats.summary import robust_summary
from jitter.stats.warmup import analyze_warmup


@dataclass(frozen=True)
class DiagnosticFeatures:
    """Aggregated empirical features extracted from rerun series and aligned telemetry."""

    # Sample sizes and sufficiency
    min_n_eff: float
    total_samples: int
    min_run_samples: int

    # Warmup features
    warmup_detected: bool
    warmup_ratio: float
    warmup_discard_iters: int

    # Thermal features
    theil_sen_slope: float
    mann_kendall_p: float
    thermal_drift_detected: bool
    peak_thermal_status: int
    max_battery_temp_c: float
    cooldown_cliffs_delta: float
    cooldown_shift_ms: float

    # Delegate features
    delegate_requested: str
    delegate_executed: str
    delegate_mismatch: bool
    cpu_shift_ms: float
    cpu_cliffs_delta: float
    cpu_cliffs_magnitude: str

    # Background noise features
    cold_mad: float
    cold_median: float
    cold_mad_ratio: float
    cold_outlier_count: int
    quiet_outlier_count: int
    quiet_p99_reduction_ratio: float
    quiet_median_shift_ratio: float

    # Uniform regression features
    warm_shift_ms: float
    warm_permutation_p: float
    warm_cliffs_delta: float
    warm_cliffs_magnitude: str


def extract_features(runs_or_report: Report | list[RunEntry]) -> DiagnosticFeatures:
    """Extract diagnostic features from a Report or list of RunEntry objects."""
    if isinstance(runs_or_report, Report):
        runs = runs_or_report.runs
    else:
        runs = runs_or_report

    if not runs:
        raise ValueError("Cannot extract diagnostic features from an empty list of runs")

    # Group runs by role
    cold_run: RunEntry | None = None
    warm_run: RunEntry | None = None
    cooldown_run: RunEntry | None = None
    delegate_swap_run: RunEntry | None = None
    quiet_run: RunEntry | None = None

    for r in runs:
        if r.rerun == RerunType.COLD_WARM:
            if r.process_start == ProcessStart.COLD and cold_run is None:
                cold_run = r
            elif warm_run is None:
                warm_run = r
        elif r.rerun == RerunType.COOLDOWN and cooldown_run is None:
            cooldown_run = r
        elif r.rerun == RerunType.DELEGATE_SWAP and delegate_swap_run is None:
            delegate_swap_run = r
        elif r.rerun == RerunType.QUIET_DEVICE and quiet_run is None:
            quiet_run = r

    # Fallbacks if rerun types are not explicitly split
    if cold_run is None:
        cold_run = runs[0]
    if warm_run is None:
        warm_run = runs[1] if len(runs) > 1 else cold_run
    if cooldown_run is None:
        cooldown_run = runs[2] if len(runs) > 2 else cold_run
    if delegate_swap_run is None:
        delegate_swap_run = runs[3] if len(runs) > 3 else cold_run
    if quiet_run is None:
        quiet_run = runs[4] if len(runs) > 4 else cold_run

    # Effective sample sizes
    n_eff_vals = [effective_sample_size(r.samples_ms).n_eff for r in runs]
    min_n_eff = float(min(n_eff_vals))
    total_samples = sum(len(r.samples_ms) for r in runs)
    min_run_samples = min(len(r.samples_ms) for r in runs)

    # 1. Warmup analysis
    warmup_res = analyze_warmup(cold_run.samples_ms)

    # 2. Thermal drift and cooldown analysis
    drift_res = analyze_drift(cold_run.samples_ms)
    peak_thermal = (
        max(cold_run.telemetry.thermal_status) if cold_run.telemetry.thermal_status else 0
    )
    max_battery_temp = (
        max(cold_run.telemetry.battery_temp_c) if cold_run.telemetry.battery_temp_c else 25.0
    )

    cooldown_comp = compare_runs(
        cold_run.samples_ms, cooldown_run.samples_ms, seed=42, n_permutations=100
    )

    # 3. Delegate swap analysis
    delegate_mismatch = cold_run.delegate_requested != cold_run.delegate_executed
    cpu_comp = compare_runs(
        cold_run.samples_ms, delegate_swap_run.samples_ms, seed=42, n_permutations=100
    )

    # 4. Background noise & outlier analysis
    summ_cold = robust_summary(cold_run.samples_ms)
    cold_mad_ratio = summ_cold.mad / summ_cold.median if summ_cold.median > 0 else 0.0

    outliers_cold = detect_outliers(cold_run.samples_ms)
    outliers_quiet = detect_outliers(quiet_run.samples_ms)

    summ_quiet = robust_summary(quiet_run.samples_ms)
    quiet_p99_reduction = (
        (summ_cold.p99 - summ_quiet.p99) / summ_cold.p99 if summ_cold.p99 > 0 else 0.0
    )
    quiet_median_shift = (
        abs(summ_quiet.median - summ_cold.median) / summ_cold.median
        if summ_cold.median > 0
        else 0.0
    )

    # 5. Cold vs Warm comparison (for regression analysis)
    comp_warm = compare_runs(cold_run.samples_ms, warm_run.samples_ms, seed=42, n_permutations=100)

    return DiagnosticFeatures(
        min_n_eff=min_n_eff,
        total_samples=total_samples,
        min_run_samples=min_run_samples,
        warmup_detected=warmup_res.is_warmup_detected,
        warmup_ratio=warmup_res.warmup_ratio,
        warmup_discard_iters=warmup_res.discard_iterations,
        theil_sen_slope=drift_res.slope_ms_per_iter,
        mann_kendall_p=drift_res.p_value,
        thermal_drift_detected=drift_res.is_drift_detected,
        peak_thermal_status=peak_thermal,
        max_battery_temp_c=max_battery_temp,
        cooldown_cliffs_delta=cooldown_comp.cliffs_delta,
        cooldown_shift_ms=cooldown_comp.hodges_lehmann_shift,
        delegate_requested=cold_run.delegate_requested,
        delegate_executed=cold_run.delegate_executed,
        delegate_mismatch=delegate_mismatch,
        cpu_shift_ms=cpu_comp.hodges_lehmann_shift,
        cpu_cliffs_delta=cpu_comp.cliffs_delta,
        cpu_cliffs_magnitude=cpu_comp.cliffs_magnitude,
        cold_mad=summ_cold.mad,
        cold_median=summ_cold.median,
        cold_mad_ratio=cold_mad_ratio,
        cold_outlier_count=outliers_cold.outlier_count,
        quiet_outlier_count=outliers_quiet.outlier_count,
        quiet_p99_reduction_ratio=quiet_p99_reduction,
        quiet_median_shift_ratio=quiet_median_shift,
        warm_shift_ms=comp_warm.hodges_lehmann_shift,
        warm_permutation_p=comp_warm.permutation_p_value,
        warm_cliffs_delta=comp_warm.cliffs_delta,
        warm_cliffs_magnitude=comp_warm.cliffs_magnitude,
    )
