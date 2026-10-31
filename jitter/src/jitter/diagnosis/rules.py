"""Rule evaluation logic for Jitter root-cause diagnosis engine."""

from __future__ import annotations

import numpy as np

from jitter.diagnosis.features import DiagnosticFeatures
from jitter.diagnosis.thresholds import (
    DELEGATE_CPU_CLIFFS_DELTA_CEILING,
    DELEGATE_CPU_SHIFT_MAX_RATIO,
    MIN_EFFECTIVE_SAMPLE_SIZE,
    NOISE_MAD_RATIO_THRESHOLD,
    NOISE_MIN_OUTLIER_COUNT,
    NOISE_QUIET_MEDIAN_SHIFT_TOLERANCE,
    NOISE_QUIET_P99_REDUCTION_RATIO,
    REGRESSION_CLIFFS_DELTA_MIN,
    REGRESSION_MIN_SHIFT_MS,
    REGRESSION_PERMUTATION_P,
    THERMAL_COOLDOWN_CLIFFS_DELTA,
    THERMAL_MANN_KENDALL_P,
    THERMAL_MIN_STATUS,
    THERMAL_THEIL_SEN_SLOPE,
    WARMUP_MIN_DISCARD_ITERS,
    WARMUP_RATIO_THRESHOLD,
)
from jitter.schema.report import EvidenceItem


def evaluate_inconclusive(
    feat: DiagnosticFeatures,
) -> tuple[bool, float, list[EvidenceItem], str | None]:
    """Check if effective sample size or iteration count is insufficient for conclusive diagnosis."""
    # Insufficient run iterations is unconditionally inconclusive
    if feat.min_run_samples < 15:
        ev = EvidenceItem(
            rule_id="min_run_samples",
            observed_value=feat.min_run_samples,
            threshold=15,
            passed=False,
            plain_language=f"Iteration count per run ({feat.min_run_samples}) is below minimum 15 required for statistical validity.",
        )
        return True, 0.94, [ev], f"Insufficient iterations per run ({feat.min_run_samples} < 15) to evaluate latency distribution."

    # Insufficient total samples across runs
    if feat.total_samples < 50:
        ev = EvidenceItem(
            rule_id="min_sample_count",
            observed_value=feat.total_samples,
            threshold=50,
            passed=False,
            plain_language=f"Total recorded sample count ({feat.total_samples}) is below minimum 50 required across reruns.",
        )
        return True, 0.94, [ev], f"Insufficient total sample count ({feat.total_samples} < 50) to establish reliable baseline."

    # If severe autocorrelation reduces effective sample size below threshold
    # and no deterministic physical mechanism is present:
    has_deterministic_signal = (
        (feat.warmup_detected and feat.warmup_ratio >= WARMUP_RATIO_THRESHOLD)
        or (feat.theil_sen_slope >= THERMAL_THEIL_SEN_SLOPE and (feat.peak_thermal_status >= THERMAL_MIN_STATUS or abs(feat.cooldown_cliffs_delta) >= 0.33))
        or feat.delegate_mismatch
        or (feat.cold_outlier_count >= NOISE_MIN_OUTLIER_COUNT and feat.quiet_p99_reduction_ratio >= NOISE_QUIET_P99_REDUCTION_RATIO)
    )

    if feat.min_n_eff < MIN_EFFECTIVE_SAMPLE_SIZE and not has_deterministic_signal:
        ev = EvidenceItem(
            rule_id="min_effective_sample_size",
            observed_value=float(np.round(feat.min_n_eff, 1)),
            threshold=MIN_EFFECTIVE_SAMPLE_SIZE,
            passed=False,
            plain_language=(
                f"Effective sample size ({feat.min_n_eff:.1f}) is below minimum threshold "
                f"({MIN_EFFECTIVE_SAMPLE_SIZE:.0f}) required for conclusive diagnosis."
            ),
        )
        reason = (
            f"Overwhelming variance or severe autocorrelation (n_eff={feat.min_n_eff:.1f} < {MIN_EFFECTIVE_SAMPLE_SIZE:.0f}). "
            "Data cannot distinguish systematic root causes from random noise."
        )
        confidence = float(np.clip(0.88 + (1.0 - (feat.min_n_eff / MIN_EFFECTIVE_SAMPLE_SIZE)) * 0.08, 0.88, 0.96))
        return True, confidence, [ev], reason

    return False, 0.0, [], None


def evaluate_warmup_drift(
    feat: DiagnosticFeatures,
) -> tuple[bool, float, list[EvidenceItem]]:
    """Evaluate evidence for startup warmup latency drift."""
    evidence: list[EvidenceItem] = []

    passed_ratio = feat.warmup_ratio >= WARMUP_RATIO_THRESHOLD
    evidence.append(
        EvidenceItem(
            rule_id="warmup_ratio_threshold",
            observed_value=float(np.round(feat.warmup_ratio, 2)),
            threshold=WARMUP_RATIO_THRESHOLD,
            passed=passed_ratio,
            plain_language=(
                f"Cold process initial iterations were {feat.warmup_ratio:.2f}x slower than steady state "
                f"(threshold: {WARMUP_RATIO_THRESHOLD:.2f}x)."
            ),
        )
    )

    passed_iters = feat.warmup_discard_iters >= WARMUP_MIN_DISCARD_ITERS
    evidence.append(
        EvidenceItem(
            rule_id="warmup_discard_iterations",
            observed_value=feat.warmup_discard_iters,
            threshold=WARMUP_MIN_DISCARD_ITERS,
            passed=passed_iters,
            plain_language=(
                f"CUSUM change-point detector identified {feat.warmup_discard_iters} initial iterations "
                f"before latency reached steady-state stability."
            ),
        )
    )

    passed_noise_suppression = (
        feat.quiet_p99_reduction_ratio >= NOISE_QUIET_P99_REDUCTION_RATIO
        and feat.cold_outlier_count >= 2
    )
    is_warmup = feat.warmup_detected and passed_ratio and passed_iters and not passed_noise_suppression
    confidence = 0.0
    if is_warmup:
        # Calibrate confidence based on effect magnitude
        confidence = float(
            np.clip(0.85 + (feat.warmup_ratio - WARMUP_RATIO_THRESHOLD) * 0.08, 0.85, 0.99)
        )

    return is_warmup, confidence, evidence


def evaluate_thermal_throttling(
    feat: DiagnosticFeatures,
) -> tuple[bool, float, list[EvidenceItem]]:
    """Evaluate evidence for thermal downclocking and throttling."""
    evidence: list[EvidenceItem] = []

    passed_slope = feat.theil_sen_slope >= THERMAL_THEIL_SEN_SLOPE
    evidence.append(
        EvidenceItem(
            rule_id="theil_sen_drift_slope",
            observed_value=float(np.round(feat.theil_sen_slope, 4)),
            threshold=THERMAL_THEIL_SEN_SLOPE,
            passed=passed_slope,
            plain_language=(
                f"Monotonic latency degradation slope of {feat.theil_sen_slope:.4f} ms/iter "
                f"exceeds threshold {THERMAL_THEIL_SEN_SLOPE:.4f} ms/iter."
            ),
        )
    )

    passed_status = feat.peak_thermal_status >= THERMAL_MIN_STATUS or feat.max_battery_temp_c >= 38.0
    evidence.append(
        EvidenceItem(
            rule_id="thermal_status_elevation",
            observed_value=feat.peak_thermal_status,
            threshold=THERMAL_MIN_STATUS,
            passed=passed_status,
            plain_language=(
                f"Device reached thermal throttling status {feat.peak_thermal_status} "
                f"(battery temperature: {feat.max_battery_temp_c:.1f}°C)."
            ),
        )
    )

    passed_cooldown = abs(feat.cooldown_cliffs_delta) >= 0.330 or abs(feat.cooldown_shift_ms) >= 1.0
    evidence.append(
        EvidenceItem(
            rule_id="cooldown_recovery_effect",
            observed_value=float(np.round(feat.cooldown_cliffs_delta, 3)),
            threshold=THERMAL_COOLDOWN_CLIFFS_DELTA,
            passed=passed_cooldown,
            plain_language=(
                f"Rested cooldown run recovered baseline latency with Cliff's delta {feat.cooldown_cliffs_delta:.3f} "
                f"(Hodges-Lehmann shift: {feat.cooldown_shift_ms:.2f} ms)."
            ),
        )
    )

    is_thermal = (
        (passed_slope and passed_status)
        or (passed_slope and passed_cooldown and feat.mann_kendall_p < 0.05)
    )
    confidence = 0.0
    if is_thermal:
        agreement = sum([passed_slope, passed_status, passed_cooldown])
        confidence = float(np.clip(0.85 + agreement * 0.04, 0.88, 0.98))

    return is_thermal, confidence, evidence


def evaluate_delegate_fallback(
    feat: DiagnosticFeatures,
) -> tuple[bool, float, list[EvidenceItem]]:
    """Evaluate evidence for silent delegate failure or execution fallback to CPU."""
    evidence: list[EvidenceItem] = []

    mismatch = feat.delegate_mismatch
    evidence.append(
        EvidenceItem(
            rule_id="delegate_execution_match",
            observed_value=feat.delegate_executed,
            threshold=feat.delegate_requested,
            passed=not mismatch,
            plain_language=(
                f"Requested delegate '{feat.delegate_requested}' resulted in actual executed delegate '{feat.delegate_executed}'."
            ),
        )
    )

    # Parity check against CPU baseline
    passed_cpu_parity = (
        abs(feat.cpu_cliffs_delta) <= DELEGATE_CPU_CLIFFS_DELTA_CEILING
        or abs(feat.cpu_shift_ms) <= 5.0
    )
    evidence.append(
        EvidenceItem(
            rule_id="delegate_vs_cpu_parity",
            observed_value=float(np.round(feat.cpu_cliffs_delta, 3)),
            threshold=DELEGATE_CPU_CLIFFS_DELTA_CEILING,
            passed=passed_cpu_parity,
            plain_language=(
                f"Comparison against CPU reference showed Cliff's delta {feat.cpu_cliffs_delta:.3f} "
                f"({feat.cpu_cliffs_magnitude}), indicating delegate fallback to CPU performance."
            ),
        )
    )

    is_fallback = mismatch or (
        feat.delegate_requested.lower() != "cpu" and passed_cpu_parity and not feat.thermal_drift_detected
    )

    confidence = 0.0
    if is_fallback:
        confidence = 0.98 if mismatch else 0.91

    return is_fallback, confidence, evidence


def evaluate_background_noise(
    feat: DiagnosticFeatures,
) -> tuple[bool, float, list[EvidenceItem]]:
    """Evaluate evidence for transient OS / background contention spikes."""
    evidence: list[EvidenceItem] = []

    passed_bursts = feat.cold_outlier_count >= NOISE_MIN_OUTLIER_COUNT or feat.cold_mad_ratio >= NOISE_MAD_RATIO_THRESHOLD
    evidence.append(
        EvidenceItem(
            rule_id="outlier_burst_frequency",
            observed_value=feat.cold_outlier_count,
            threshold=NOISE_MIN_OUTLIER_COUNT,
            passed=passed_bursts,
            plain_language=(
                f"Detected {feat.cold_outlier_count} modified Z-score latency outlier bursts in active mode "
                f"(MAD/median ratio: {feat.cold_mad_ratio:.2f})."
            ),
        )
    )

    passed_quiet = feat.quiet_p99_reduction_ratio >= NOISE_QUIET_P99_REDUCTION_RATIO
    evidence.append(
        EvidenceItem(
            rule_id="quiet_mode_p99_suppression",
            observed_value=float(np.round(feat.quiet_p99_reduction_ratio, 2)),
            threshold=NOISE_QUIET_P99_REDUCTION_RATIO,
            passed=passed_quiet,
            plain_language=(
                f"Quiet device mode suppressed tail latency by {feat.quiet_p99_reduction_ratio * 100:.1f}% "
                f"(outliers dropped to {feat.quiet_outlier_count})."
            ),
        )
    )

    is_thermal_present = feat.thermal_drift_detected and (
        feat.peak_thermal_status >= THERMAL_MIN_STATUS or feat.max_battery_temp_c >= 35.0
    )
    is_warmup_present = feat.warmup_detected and feat.warmup_ratio >= WARMUP_RATIO_THRESHOLD

    is_noise = passed_bursts and passed_quiet and not is_thermal_present
    confidence = 0.0
    if is_noise:
        confidence = float(np.clip(0.90 + feat.quiet_p99_reduction_ratio * 0.08, 0.90, 0.98))

    return is_noise, confidence, evidence


def evaluate_regression(
    feat: DiagnosticFeatures,
) -> tuple[bool, float, list[EvidenceItem]]:
    """Evaluate evidence for persistent uniform model or code regression."""
    evidence: list[EvidenceItem] = []

    # 1. Absence of warmup
    is_warmup_free = not feat.warmup_detected or feat.warmup_ratio < 1.25
    evidence.append(
        EvidenceItem(
            rule_id="warmup_absence",
            observed_value=float(np.round(feat.warmup_ratio, 2)),
            threshold=1.25,
            passed=is_warmup_free,
            plain_language=f"Startup warmup ratio is steady ({feat.warmup_ratio:.2f}x < 1.25x).",
        )
    )

    # 2. Absence of thermal throttling
    is_thermal_free = (
        (feat.theil_sen_slope < THERMAL_THEIL_SEN_SLOPE or feat.mann_kendall_p >= 0.05)
        and feat.peak_thermal_status == 0
        and feat.max_battery_temp_c < 36.0
        and abs(feat.cooldown_cliffs_delta) < 0.30
    )
    evidence.append(
        EvidenceItem(
            rule_id="thermal_absence",
            observed_value=float(np.round(feat.theil_sen_slope, 4)),
            threshold=THERMAL_THEIL_SEN_SLOPE,
            passed=is_thermal_free,
            plain_language=f"No thermal degradation trend ({feat.theil_sen_slope:.4f} ms/iter, status {feat.peak_thermal_status}).",
        )
    )

    # 3. Absence of delegate fallback
    is_delegate_healthy = (
        not feat.delegate_mismatch
        and feat.cpu_shift_ms > 3.0
        and abs(feat.cpu_cliffs_delta) >= 0.20
    )
    evidence.append(
        EvidenceItem(
            rule_id="delegate_healthy",
            observed_value=float(np.round(feat.cpu_shift_ms, 2)),
            threshold=3.0,
            passed=is_delegate_healthy,
            plain_language=f"Hardware delegate executed cleanly with +{feat.cpu_shift_ms:.1f}ms speedup over CPU reference.",
        )
    )

    # 4. Absence of background contention noise
    is_noise_free = (
        feat.quiet_p99_reduction_ratio < 0.20
        and feat.cold_outlier_count <= 3
        and feat.cold_mad_ratio < 0.25
    )
    evidence.append(
        EvidenceItem(
            rule_id="noise_absence",
            observed_value=feat.cold_outlier_count,
            threshold=3,
            passed=is_noise_free,
            plain_language=f"Execution is stable with minimal contention ({feat.cold_outlier_count} outliers, MAD ratio {feat.cold_mad_ratio:.2f}).",
        )
    )

    # 5. Sufficient sample size
    is_well_sampled = feat.min_run_samples >= 15 and feat.min_n_eff >= 10.0

    is_regression = (
        is_warmup_free
        and is_thermal_free
        and is_delegate_healthy
        and is_noise_free
        and is_well_sampled
    )

    confidence = 0.0
    if is_regression:
        confidence = 0.95

    return is_regression, confidence, evidence
