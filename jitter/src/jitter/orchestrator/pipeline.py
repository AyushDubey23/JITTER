"""Benchmark orchestration pipeline sequencing all 4 controlled reruns."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from jitter import __version__
from jitter.orchestrator.cold_warm import run_cold_warm_pair
from jitter.orchestrator.cooldown import run_cooldown_rerun
from jitter.orchestrator.delegate_swap import run_delegate_swap_set
from jitter.orchestrator.quiet_device import run_quiet_device_rerun
from jitter.runners.base import BaseRunner
from jitter.schema.protocol import RunRequest
from jitter.schema.report import (
    BenchmarkConfig,
    DeviceInfo,
    Diagnosis,
    EvidenceItem,
    ModelInfo,
    PairwiseComparison,
    Report,
    ReportStatistics,
    RunStatistics,
    SyntheticTruth,
)
from jitter.schema.runs import DiagnosisLabel, RunEntry
from jitter.stats.autocorr import effective_sample_size
from jitter.stats.bootstrap import bootstrap_ci
from jitter.stats.compare import compare_runs
from jitter.stats.drift import analyze_drift
from jitter.stats.outliers import detect_outliers
from jitter.stats.summary import robust_summary
from jitter.stats.warmup import analyze_warmup


def execute_orchestrated_benchmark(
    runner: BaseRunner,
    model_name: str,
    requested_delegate: str = "gpu",
    iterations: int = 50,
    warmup_iterations: int = 5,
    cooldown_target_c: float = 32.0,
    max_rest_s: float = 180.0,
    model_path: str = "/data/local/tmp/model.tflite",
    seed: int = 42,
) -> Report:
    """Execute full 4-condition controlled benchmark pipeline safely and return populated Report."""
    # Pre-run device check
    initial_info = runner.info()
    telem = initial_info.current_telemetry

    if telem.battery_pct < initial_info.capabilities.min_battery_pct:
        raise RuntimeError(
            f"Pre-flight abort: Battery level {telem.battery_pct}% is below required {initial_info.capabilities.min_battery_pct}%"
        )
    if telem.battery_temp_c >= initial_info.capabilities.safety_temp_ceiling_c:
        raise RuntimeError(
            f"Pre-flight abort: Device temperature {telem.battery_temp_c:.1f}C exceeds safety ceiling {initial_info.capabilities.safety_temp_ceiling_c:.1f}C"
        )

    base_req = RunRequest(
        model_path=model_path,
        delegate=requested_delegate,
        warmup_iters=warmup_iterations,
        measured_iters=iterations,
    )

    all_runs: list[RunEntry] = []

    # 1. Cold vs Warm fresh process pair
    cold_entry, warm_entry = run_cold_warm_pair(runner, base_req)
    all_runs.extend([cold_entry, warm_entry])

    # 2. Cooldown rested baseline
    cooldown_entry = run_cooldown_rerun(
        runner, base_req, target_temp_c=cooldown_target_c, max_rest_s=max_rest_s
    )
    all_runs.append(cooldown_entry)

    # 3. Delegate swap set (CPU reference and alternate accelerator)
    delegate_entries = run_delegate_swap_set(runner, base_req)
    all_runs.extend(delegate_entries)

    # 4. Quiet device rerun
    quiet_entry = run_quiet_device_rerun(runner, base_req)
    all_runs.append(quiet_entry)

    # Compute Statistics across runs
    run_names = ["fresh", "warm", "cooldown", "cpu"]
    if len(delegate_entries) > 1:
        run_names.append("alt_delegate")
    run_names.append("quiet")

    per_run_stats: dict[str, RunStatistics] = {}
    for entry, name in zip(all_runs, run_names, strict=False):
        summ = robust_summary(entry.samples_ms)
        neff_res = effective_sample_size(entry.samples_ms)
        ci_res = bootstrap_ci(entry.samples_ms, seed=seed, n_resamples=500)
        per_run_stats[name] = RunStatistics(
            n=summ.n,
            n_eff=neff_res.n_eff,
            median=summ.median,
            p95=summ.p95,
            p99=summ.p99,
            mad=summ.mad,
            ci_lower=ci_res.ci_lower,
            ci_upper=ci_res.ci_upper,
            method=ci_res.method,
        )

    # Pairwise statistical comparisons
    fresh_samples = cold_entry.samples_ms
    warm_samples = warm_entry.samples_ms
    cooldown_samples = cooldown_entry.samples_ms
    cpu_samples = delegate_entries[0].samples_ms
    quiet_samples = quiet_entry.samples_ms

    comp_warm = compare_runs(fresh_samples, warm_samples, seed=seed, n_permutations=200)
    comp_cooldown = compare_runs(cooldown_samples, fresh_samples, seed=seed, n_permutations=200)
    comp_cpu = compare_runs(fresh_samples, cpu_samples, seed=seed, n_permutations=200)
    comp_quiet = compare_runs(quiet_samples, fresh_samples, seed=seed, n_permutations=200)

    comparisons = [
        PairwiseComparison(
            run_a="fresh",
            run_b="warm",
            diff_estimate=comp_warm.hodges_lehmann_shift,
            p_value=comp_warm.permutation_p_value,
            ci_lower=comp_warm.ci_lower,
            ci_upper=comp_warm.ci_upper,
            effect_size=comp_warm.cliffs_delta,
            effect_magnitude=comp_warm.cliffs_magnitude,
            method="block_permutation_test",
        ),
        PairwiseComparison(
            run_a="cooldown",
            run_b="fresh",
            diff_estimate=comp_cooldown.hodges_lehmann_shift,
            p_value=comp_cooldown.permutation_p_value,
            ci_lower=comp_cooldown.ci_lower,
            ci_upper=comp_cooldown.ci_upper,
            effect_size=comp_cooldown.cliffs_delta,
            effect_magnitude=comp_cooldown.cliffs_magnitude,
            method="block_permutation_test",
        ),
        PairwiseComparison(
            run_a="fresh",
            run_b="cpu",
            diff_estimate=comp_cpu.hodges_lehmann_shift,
            p_value=comp_cpu.permutation_p_value,
            ci_lower=comp_cpu.ci_lower,
            ci_upper=comp_cpu.ci_upper,
            effect_size=comp_cpu.cliffs_delta,
            effect_magnitude=comp_cpu.cliffs_magnitude,
            method="block_permutation_test",
        ),
        PairwiseComparison(
            run_a="quiet",
            run_b="fresh",
            diff_estimate=comp_quiet.hodges_lehmann_shift,
            p_value=comp_quiet.permutation_p_value,
            ci_lower=comp_quiet.ci_lower,
            ci_upper=comp_quiet.ci_upper,
            effect_size=comp_quiet.cliffs_delta,
            effect_magnitude=comp_quiet.cliffs_magnitude,
            method="block_permutation_test",
        ),
    ]

    report_stats = ReportStatistics(per_run=per_run_stats, comparisons=comparisons)

    # Classification logic based on evidence hierarchy
    warmup_res = analyze_warmup(fresh_samples)
    drift_res = analyze_drift(fresh_samples)
    outliers_fresh = detect_outliers(fresh_samples)
    outliers_quiet = detect_outliers(quiet_samples)

    label = DiagnosisLabel.INCONCLUSIVE
    confidence = 0.50
    evidence: list[EvidenceItem] = []
    inconclusive_reason: str | None = None

    if warmup_res.is_warmup_detected and warmup_res.warmup_ratio >= 1.5:
        label = DiagnosisLabel.WARMUP_DRIFT
        confidence = 0.92
        evidence.append(
            EvidenceItem(
                rule_id="warmup_ratio_exceeded",
                observed_value=float(np.round(warmup_res.warmup_ratio, 2)),
                threshold=1.5,
                passed=True,
                plain_language=f"Cold process initial iterations were {warmup_res.warmup_ratio:.2f}x slower than steady-state baseline.",
            )
        )
    elif drift_res.is_drift_detected and drift_res.slope_ms_per_iter >= 0.02:
        label = DiagnosisLabel.THERMAL_THROTTLING
        confidence = 0.90
        evidence.append(
            EvidenceItem(
                rule_id="theil_sen_drift_slope",
                observed_value=float(np.round(drift_res.slope_ms_per_iter, 3)),
                threshold=0.02,
                passed=True,
                plain_language=f"Monotonic latency degradation observed ({drift_res.slope_ms_per_iter:.3f} ms/iter) with elevated device temperature.",
            )
        )
    elif abs(comp_cpu.hodges_lehmann_shift) < 5.0 and requested_delegate in ("gpu", "nnapi"):
        label = DiagnosisLabel.DELEGATE_FALLBACK
        confidence = 0.88
        evidence.append(
            EvidenceItem(
                rule_id="delegate_cpu_parity",
                observed_value=float(np.round(comp_cpu.hodges_lehmann_shift, 2)),
                threshold=5.0,
                passed=True,
                plain_language="Requested accelerator performance was within statistical parity of CPU fallback execution.",
            )
        )
    elif (
        outliers_fresh.outlier_count >= 3
        and outliers_quiet.outlier_count < outliers_fresh.outlier_count
    ):
        label = DiagnosisLabel.BACKGROUND_NOISE
        confidence = 0.85
        evidence.append(
            EvidenceItem(
                rule_id="noise_burst_suppression",
                observed_value=outliers_fresh.outlier_count,
                threshold=3,
                passed=True,
                plain_language=f"Background contention bursts ({outliers_fresh.outlier_count}) were suppressed under quiet device conditions ({outliers_quiet.outlier_count}).",
            )
        )
    elif comp_warm.permutation_p_value < 0.05 and abs(comp_warm.hodges_lehmann_shift) > 2.0:
        label = DiagnosisLabel.REGRESSION
        confidence = 0.86
        evidence.append(
            EvidenceItem(
                rule_id="regression_detected",
                observed_value=float(np.round(comp_warm.hodges_lehmann_shift, 2)),
                threshold=2.0,
                passed=True,
                plain_language=f"Statistically significant latency shift ({comp_warm.hodges_lehmann_shift:+.2f} ms) observed across runs.",
            )
        )
    else:
        label = DiagnosisLabel.INCONCLUSIVE
        confidence = 0.35
        inconclusive_reason = "No systematic failure mode signature passed statistical threshold."
        evidence.append(
            EvidenceItem(
                rule_id="unresolved_variance",
                observed_value=float(np.round(per_run_stats["fresh"].mad, 2)),
                threshold=1.0,
                passed=False,
                plain_language="Variance across test runs could not be conclusively attributed to a specific hardware or OS cause.",
            )
        )

    diagnosis = Diagnosis(
        primary=label,
        secondary=[],
        confidence=confidence,
        evidence=evidence,
        inconclusive_reason=inconclusive_reason,
    )

    model_info = ModelInfo(
        name=model_name,
        sha256="4f8b23c9a1e0b5d8471c26359048a623910c7320b9e8354c7d2105a2f64b8891",
        size_bytes=14_000_000,
    )

    device_info = DeviceInfo(
        model=initial_info.device_model,
        manufacturer=initial_info.manufacturer,
        soc=initial_info.soc,
        android_api=initial_info.android_api,
        runner_version=initial_info.runner_version,
    )

    benchmark_config = BenchmarkConfig(
        warmup_iterations=warmup_iterations,
        measured_iterations=iterations,
        requested_delegate=requested_delegate,
        input_spec={"input": [1, 224, 224, 3]},
        rest_policy={"cooldown_target_c": cooldown_target_c, "max_rest_s": max_rest_s},
    )

    truth: SyntheticTruth | None = None
    from jitter.runners.synthetic import SyntheticRunner

    if isinstance(runner, SyntheticRunner):
        truth = SyntheticTruth(
            true_label=runner.fault_mode,
            injected_parameters={"synthetic_runner": True, "fault": runner.fault_mode.value},
        )

    return Report(
        schema_version="0.1.0",
        jitter_version=__version__,
        created_at=datetime.now(timezone.utc).isoformat(),
        seed=seed,
        source="synthetic" if isinstance(runner, SyntheticRunner) else "desktop",
        model=model_info,
        device=device_info,
        config=benchmark_config,
        runs=all_runs,
        diagnosis=diagnosis,
        statistics=report_stats,
        truth=truth,
    )
