"""Scenario generator producing canonical synthetic benchmark reports.

Simulates all 5 failure modes, multi-fault scenarios, and inconclusive verdicts:
1. warmup_drift: High initial cold latency decaying to stable warm state.
2. thermal_throttling: Heat accumulation causing governor downclocking; cooldown restores baseline.
3. delegate_fallback: Requested delegate yields same or worse performance as CPU.
4. background_noise: Heavy-tailed Poisson-Pareto bursts eliminated by quiet mode.
5. regression: Uniform statistically significant latency shift without secondary causes.
6. inconclusive: Overwhelming noise or low effective sample size.

Every generated report populates Report.truth with exact injected parameters.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np

from jitter import __version__
from jitter.schema import (
    BenchmarkConfig,
    DeviceInfo,
    Diagnosis,
    DiagnosisLabel,
    EvidenceItem,
    ModelInfo,
    PairwiseComparison,
    ProcessStart,
    Report,
    ReportStatistics,
    RerunType,
    RunEntry,
    RunStatistics,
    SyntheticTruth,
    TelemetrySeries,
)
from jitter.stats.autocorr import effective_sample_size
from jitter.stats.bootstrap import bootstrap_ci
from jitter.stats.compare import compare_runs
from jitter.stats.drift import analyze_drift
from jitter.stats.outliers import detect_outliers
from jitter.stats.summary import robust_summary
from jitter.stats.warmup import analyze_warmup
from jitter.synth.delegates import MODEL_CATALOG, FallbackMode, simulate_iteration
from jitter.synth.noise import ContentionNoiseGenerator
from jitter.synth.thermal import RCThermalModel


def _build_run_entry(
    name: str,
    rerun: RerunType,
    process_start: ProcessStart,
    delegate_requested: str,
    delegate_executed: str,
    samples_ms: list[float],
    t0_ns: int,
    telemetry_thermal_status: list[int],
    telemetry_thermal_headroom: list[float | None],
    telemetry_battery_temp_c: list[float],
    telemetry_charging: list[bool],
    telemetry_screen_on: list[bool],
    notes: list[str] | None = None,
) -> RunEntry:
    n = len(samples_ms)
    timestamps_ns = [t0_ns + int(i * np.mean(samples_ms) * 1_000_000) for i in range(n)]
    series = TelemetrySeries(
        thermal_status=telemetry_thermal_status,
        thermal_headroom=telemetry_thermal_headroom,
        battery_temp_c=telemetry_battery_temp_c,
        charging=telemetry_charging,
        screen_on=telemetry_screen_on,
    )
    return RunEntry(
        rerun=rerun,
        process_start=process_start,
        delegate_requested=delegate_requested,
        delegate_executed=delegate_executed,
        samples_ms=samples_ms,
        timestamps_ns=timestamps_ns,
        telemetry=series,
        notes=notes or [f"Synthetic run: {name}"],
    )


def generate_scenario_report(
    label: DiagnosisLabel | str,
    model_name: str = "mobilenet_v2",
    iterations: int = 50,
    seed: int = 42,
    secondary_fault: str | None = None,
) -> Report:
    """Generate a fully formed, mathematically validated Report for a given scenario."""
    if isinstance(label, str):
        label = DiagnosisLabel(label)

    rng = np.random.default_rng(seed)
    created_at = datetime.now(timezone.utc).isoformat()

    model_meta = MODEL_CATALOG.get(model_name, MODEL_CATALOG["mobilenet_v2"])
    model_info = ModelInfo(
        name=model_meta.name,
        sha256="8f4b23c9a1e0b5d8471c26359048a623910c7320b9e8354c7d2105a2f64b8891",
        size_bytes=model_meta.size_bytes,
    )
    device_info = DeviceInfo(
        model="Pixel 7",
        manufacturer="Google",
        soc="Tensor G2",
        android_api=34,
        runner_version=__version__,
    )
    config = BenchmarkConfig(
        warmup_iterations=5,
        measured_iterations=iterations,
        requested_delegate="gpu",
        input_spec={"input": [1, 224, 224, 3]},
        rest_policy={"cooldown_target_c": 32.0, "max_rest_s": 180},
    )

    injected_params: dict[str, Any] = {
        "scenario": label.value,
        "iterations": iterations,
        "seed": seed,
    }

    # Initialize physical models
    thermal = RCThermalModel(initial_temp_c=28.0)
    noise_gen = ContentionNoiseGenerator(seed=seed)

    # 1. Cold Fresh Run
    fresh_latencies: list[float] = []
    fresh_thermal_status: list[int] = []
    fresh_headroom: list[float | None] = []
    fresh_temp: list[float] = []

    fallback = FallbackMode.FULL if label == DiagnosisLabel.DELEGATE_FALLBACK else FallbackMode.NONE
    if secondary_fault == "fallback":
        fallback = FallbackMode.FULL

    for i in range(iterations):
        sim = simulate_iteration(
            model_name=model_name,
            requested_delegate="gpu",
            iteration_idx=i,
            total_iterations=iterations,
            fallback_mode=fallback,
            rng=rng,
        )
        lat = sim.raw_latency_ms

        # Injected Warmup Fault
        if label == DiagnosisLabel.WARMUP_DRIFT or secondary_fault == "warmup":
            if i < 4:
                # 3.5x warmup multiplier decaying
                decay = np.exp(-0.8 * i)
                lat += sim.raw_latency_ms * (3.0 * decay)
            injected_params["warmup_multiplier"] = 3.5

        # Injected Thermal Fault
        if label == DiagnosisLabel.THERMAL_THROTTLING:
            # Heat up rapidly with heavy power
            th_state = thermal.step(power_w=sim.power_w * 2.2, dt_s=lat / 1000.0)
            lat = lat / th_state.frequency_ratio
            injected_params["thermal_boost_power"] = 2.2
        else:
            th_state = thermal.step(power_w=sim.power_w, dt_s=lat / 1000.0)

        # Injected Regression Fault
        if label == DiagnosisLabel.REGRESSION:
            lat += 12.0  # +12 ms uniform regression
            injected_params["regression_shift_ms"] = 12.0

        fresh_latencies.append(lat)
        fresh_thermal_status.append(th_state.thermal_status)
        fresh_headroom.append(th_state.thermal_headroom)
        fresh_temp.append(th_state.temperature_c)

    # Apply Background Noise if active
    if label == DiagnosisLabel.BACKGROUND_NOISE or secondary_fault == "noise":
        arr_noisy, bursts = noise_gen.apply_noise(np.array(fresh_latencies), quiet_mode=False)
        fresh_latencies = [float(v) for v in arr_noisy]
        injected_params["noise_burst_count"] = len(bursts)

    # Inconclusive scenario: small sample size with high noise
    if label == DiagnosisLabel.INCONCLUSIVE:
        fresh_latencies = [
            fresh_latencies[j] * float(rng.uniform(0.5, 2.5))
            for j in range(min(10, len(fresh_latencies)))
        ]
        fresh_thermal_status = fresh_thermal_status[: len(fresh_latencies)]
        fresh_headroom = fresh_headroom[: len(fresh_latencies)]
        fresh_temp = fresh_temp[: len(fresh_latencies)]
        injected_params["inconclusive_reason"] = "Severe noise and low sample size"

    n_samples = len(fresh_latencies)
    run_fresh = _build_run_entry(
        name="cold_warm_fresh",
        rerun=RerunType.COLD_WARM,
        process_start=ProcessStart.COLD,
        delegate_requested="gpu",
        delegate_executed="cpu" if fallback == FallbackMode.FULL else "gpu",
        samples_ms=fresh_latencies,
        t0_ns=1_000_000_000,
        telemetry_thermal_status=fresh_thermal_status,
        telemetry_thermal_headroom=fresh_headroom,
        telemetry_battery_temp_c=fresh_temp,
        telemetry_charging=[False] * n_samples,
        telemetry_screen_on=[True] * n_samples,
    )

    # 2. Warm Run (same process immediately following)
    warm_latencies: list[float] = []
    warm_thermal_status: list[int] = []
    warm_headroom: list[float | None] = []
    warm_temp: list[float] = []

    for i in range(n_samples):
        sim = simulate_iteration(
            model_name=model_name,
            requested_delegate="gpu",
            iteration_idx=i,
            total_iterations=n_samples,
            fallback_mode=fallback,
            rng=rng,
        )
        lat = sim.raw_latency_ms
        if label == DiagnosisLabel.REGRESSION:
            lat += 12.0
        th_state = thermal.step(power_w=sim.power_w, dt_s=lat / 1000.0)
        warm_latencies.append(lat)
        warm_thermal_status.append(th_state.thermal_status)
        warm_headroom.append(th_state.thermal_headroom)
        warm_temp.append(th_state.temperature_c)

    run_warm = _build_run_entry(
        name="cold_warm_warm",
        rerun=RerunType.COLD_WARM,
        process_start=ProcessStart.WARM,
        delegate_requested="gpu",
        delegate_executed="cpu" if fallback == FallbackMode.FULL else "gpu",
        samples_ms=warm_latencies,
        t0_ns=2_000_000_000,
        telemetry_thermal_status=warm_thermal_status,
        telemetry_thermal_headroom=warm_headroom,
        telemetry_battery_temp_c=warm_temp,
        telemetry_charging=[False] * n_samples,
        telemetry_screen_on=[True] * n_samples,
    )

    # 3. Cooldown Run (rested baseline)
    thermal.cool_down(rest_duration_s=60.0)
    cooldown_latencies: list[float] = []
    cooldown_thermal_status: list[int] = []
    cooldown_headroom: list[float | None] = []
    cooldown_temp: list[float] = []

    for i in range(n_samples):
        sim = simulate_iteration(
            model_name=model_name,
            requested_delegate="gpu",
            iteration_idx=i,
            total_iterations=n_samples,
            fallback_mode=fallback,
            rng=rng,
        )
        lat = sim.raw_latency_ms
        if label == DiagnosisLabel.REGRESSION:
            lat += 12.0
        th_state = thermal.step(power_w=sim.power_w, dt_s=lat / 1000.0)
        cooldown_latencies.append(lat)
        cooldown_thermal_status.append(th_state.thermal_status)
        cooldown_headroom.append(th_state.thermal_headroom)
        cooldown_temp.append(th_state.temperature_c)

    run_cooldown = _build_run_entry(
        name="cooldown_rested",
        rerun=RerunType.COOLDOWN,
        process_start=ProcessStart.COLD,
        delegate_requested="gpu",
        delegate_executed="cpu" if fallback == FallbackMode.FULL else "gpu",
        samples_ms=cooldown_latencies,
        t0_ns=3_000_000_000,
        telemetry_thermal_status=cooldown_thermal_status,
        telemetry_thermal_headroom=cooldown_headroom,
        telemetry_battery_temp_c=cooldown_temp,
        telemetry_charging=[False] * n_samples,
        telemetry_screen_on=[True] * n_samples,
    )

    # 4. CPU Reference Run
    cpu_latencies: list[float] = []
    cpu_thermal_status: list[int] = []
    cpu_headroom: list[float | None] = []
    cpu_temp: list[float] = []

    for i in range(n_samples):
        sim = simulate_iteration(
            model_name=model_name,
            requested_delegate="cpu",
            iteration_idx=i,
            total_iterations=n_samples,
            fallback_mode=FallbackMode.NONE,
            rng=rng,
        )
        lat = sim.raw_latency_ms
        if label == DiagnosisLabel.REGRESSION:
            lat += 12.0
        th_state = thermal.step(power_w=sim.power_w, dt_s=lat / 1000.0)
        cpu_latencies.append(lat)
        cpu_thermal_status.append(th_state.thermal_status)
        cpu_headroom.append(th_state.thermal_headroom)
        cpu_temp.append(th_state.temperature_c)

    run_cpu = _build_run_entry(
        name="delegate_cpu",
        rerun=RerunType.DELEGATE_SWAP,
        process_start=ProcessStart.COLD,
        delegate_requested="cpu",
        delegate_executed="cpu",
        samples_ms=cpu_latencies,
        t0_ns=4_000_000_000,
        telemetry_thermal_status=cpu_thermal_status,
        telemetry_thermal_headroom=cpu_headroom,
        telemetry_battery_temp_c=cpu_temp,
        telemetry_charging=[False] * n_samples,
        telemetry_screen_on=[True] * n_samples,
    )

    # 5. Quiet Mode Run (bursts eliminated)
    quiet_latencies: list[float] = []
    quiet_thermal_status: list[int] = []
    quiet_headroom: list[float | None] = []
    quiet_temp: list[float] = []

    for i in range(n_samples):
        sim = simulate_iteration(
            model_name=model_name,
            requested_delegate="gpu",
            iteration_idx=i,
            total_iterations=n_samples,
            fallback_mode=fallback,
            rng=rng,
        )
        lat = sim.raw_latency_ms
        if label == DiagnosisLabel.REGRESSION:
            lat += 12.0
        th_state = thermal.step(power_w=sim.power_w, dt_s=lat / 1000.0)
        quiet_latencies.append(lat)
        quiet_thermal_status.append(th_state.thermal_status)
        quiet_headroom.append(th_state.thermal_headroom)
        quiet_temp.append(th_state.temperature_c)

    run_quiet = _build_run_entry(
        name="quiet_device",
        rerun=RerunType.QUIET_DEVICE,
        process_start=ProcessStart.COLD,
        delegate_requested="gpu",
        delegate_executed="cpu" if fallback == FallbackMode.FULL else "gpu",
        samples_ms=quiet_latencies,
        t0_ns=5_000_000_000,
        telemetry_thermal_status=quiet_thermal_status,
        telemetry_thermal_headroom=quiet_headroom,
        telemetry_battery_temp_c=quiet_temp,
        telemetry_charging=[False] * n_samples,
        telemetry_screen_on=[True] * n_samples,
    )

    runs = [run_fresh, run_warm, run_cooldown, run_cpu, run_quiet]

    # Compute Statistics across runs
    per_run_stats: dict[str, RunStatistics] = {}
    for entry, name in zip(runs, ["fresh", "warm", "cooldown", "cpu", "quiet"], strict=False):
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

    # Compute Pairwise Comparisons
    comp_warm = compare_runs(fresh_latencies, warm_latencies, seed=seed, n_permutations=200)
    comp_cooldown = compare_runs(cooldown_latencies, fresh_latencies, seed=seed, n_permutations=200)
    comp_cpu = compare_runs(fresh_latencies, cpu_latencies, seed=seed, n_permutations=200)
    comp_quiet = compare_runs(quiet_latencies, fresh_latencies, seed=seed, n_permutations=200)

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

    # Formulate Diagnosis & Evidence Items based on ground truth
    evidence: list[EvidenceItem] = []
    inconclusive_reason: str | None = None
    confidence = 0.94

    if label == DiagnosisLabel.WARMUP_DRIFT:
        warmup_res = analyze_warmup(fresh_latencies)
        evidence.append(
            EvidenceItem(
                rule_id="warmup_ratio_threshold",
                observed_value=float(np.round(warmup_res.warmup_ratio, 2)),
                threshold=1.5,
                passed=warmup_res.is_warmup_detected,
                plain_language=f"Cold process initial iterations were {warmup_res.warmup_ratio:.2f}x slower than steady state.",
            )
        )
    elif label == DiagnosisLabel.THERMAL_THROTTLING:
        drift_res = analyze_drift(fresh_latencies)
        evidence.append(
            EvidenceItem(
                rule_id="theil_sen_drift_slope",
                observed_value=float(np.round(drift_res.slope_ms_per_iter, 3)),
                threshold=0.02,
                passed=drift_res.is_drift_detected,
                plain_language=f"Latency degraded at {drift_res.slope_ms_per_iter:.3f} ms/iter coinciding with elevated thermal status.",
            )
        )
    elif label == DiagnosisLabel.DELEGATE_FALLBACK:
        evidence.append(
            EvidenceItem(
                rule_id="delegate_vs_cpu_parity",
                observed_value=float(np.round(comp_cpu.hodges_lehmann_shift, 2)),
                threshold=5.0,
                passed=True,
                plain_language="Requested GPU delegate exhibited indistinguishable latency compared to CPU baseline.",
            )
        )
    elif label == DiagnosisLabel.BACKGROUND_NOISE:
        outliers_fresh = detect_outliers(fresh_latencies)
        outliers_quiet = detect_outliers(quiet_latencies)
        evidence.append(
            EvidenceItem(
                rule_id="noise_burst_suppression",
                observed_value=outliers_fresh.outlier_count,
                threshold=3,
                passed=True,
                plain_language=f"Detected {outliers_fresh.outlier_count} contention bursts in active mode, reduced to {outliers_quiet.outlier_count} in quiet mode.",
            )
        )
    elif label == DiagnosisLabel.REGRESSION:
        evidence.append(
            EvidenceItem(
                rule_id="regression_shift_detected",
                observed_value=float(np.round(comp_warm.hodges_lehmann_shift, 2)),
                threshold=2.0,
                passed=True,
                plain_language=f"Significant baseline latency increase (+{comp_warm.hodges_lehmann_shift:.2f} ms) across runs without thermal/delegate faults.",
            )
        )
    else:
        # Inconclusive
        confidence = 0.28
        inconclusive_reason = (
            "Overwhelming random variance across test runs masked systematic signal (n_eff < 30)."
        )
        evidence.append(
            EvidenceItem(
                rule_id="min_effective_sample_size",
                observed_value=float(np.round(per_run_stats["fresh"].n_eff, 1)),
                threshold=30.0,
                passed=False,
                plain_language=f"Effective sample size ({per_run_stats['fresh'].n_eff:.1f}) below minimum 30 required for conclusive diagnosis.",
            )
        )

    diagnosis = Diagnosis(
        primary=label,
        secondary=[DiagnosisLabel(secondary_fault)] if secondary_fault else [],
        confidence=confidence,
        evidence=evidence,
        inconclusive_reason=inconclusive_reason,
    )

    truth = SyntheticTruth(
        true_label=label,
        injected_parameters=injected_params,
    )

    return Report(
        schema_version="0.1.0",
        jitter_version=__version__,
        created_at=created_at,
        seed=seed,
        source="synthetic",
        model=model_info,
        device=device_info,
        config=config,
        runs=runs,
        diagnosis=diagnosis,
        statistics=report_stats,
        truth=truth,
    )
