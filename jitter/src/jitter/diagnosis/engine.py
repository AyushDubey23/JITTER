"""Root-cause diagnosis engine evaluating multi-condition evidence."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from jitter.diagnosis.features import extract_features
from jitter.diagnosis.rules import (
    evaluate_background_noise,
    evaluate_delegate_fallback,
    evaluate_inconclusive,
    evaluate_regression,
    evaluate_thermal_throttling,
    evaluate_warmup_drift,
)
from jitter.schema.report import Diagnosis, EvidenceItem
from jitter.schema.runs import DiagnosisLabel, RunEntry

if TYPE_CHECKING:
    from jitter.schema.report import Report


class DiagnosisEngine:
    """Deterministic, rule-based diagnosis engine for on-device ML latency root causes."""

    def diagnose(self, runs_or_report: Report | list[RunEntry]) -> Diagnosis:
        """Perform root-cause diagnosis on benchmark reruns."""
        features = extract_features(runs_or_report)

        # 1. Check for inconclusive sample sufficiency
        is_inconclusive, inconc_conf, inconc_ev, inconc_reason = evaluate_inconclusive(features)
        if is_inconclusive:
            return Diagnosis(
                primary=DiagnosisLabel.INCONCLUSIVE,
                secondary=[],
                confidence=inconc_conf,
                evidence=inconc_ev,
                inconclusive_reason=inconc_reason,
            )

        # 2. Evaluate all primary failure modes
        candidates: list[tuple[DiagnosisLabel, float, list[EvidenceItem]]] = []

        # Warmup drift
        is_warmup, warmup_conf, warmup_ev = evaluate_warmup_drift(features)
        if is_warmup:
            candidates.append((DiagnosisLabel.WARMUP_DRIFT, warmup_conf, warmup_ev))

        # Thermal throttling
        is_thermal, thermal_conf, thermal_ev = evaluate_thermal_throttling(features)
        if is_thermal:
            candidates.append((DiagnosisLabel.THERMAL_THROTTLING, thermal_conf, thermal_ev))

        # Delegate fallback
        is_delegate, delegate_conf, delegate_ev = evaluate_delegate_fallback(features)
        if is_delegate:
            candidates.append((DiagnosisLabel.DELEGATE_FALLBACK, delegate_conf, delegate_ev))

        # Background noise
        is_noise, noise_conf, noise_ev = evaluate_background_noise(features)
        if is_noise:
            candidates.append((DiagnosisLabel.BACKGROUND_NOISE, noise_conf, noise_ev))

        # Uniform regression (only if not better explained by thermal or delegate)
        is_regression = False
        reg_conf = 0.0
        reg_ev: list[EvidenceItem] = []
        if not is_thermal and not is_delegate:
            is_regression, reg_conf, reg_ev = evaluate_regression(features)
            if is_regression:
                candidates.append((DiagnosisLabel.REGRESSION, reg_conf, reg_ev))

        # 3. Classify outcome
        if not candidates:
            inconclusive_ev = [
                EvidenceItem(
                    rule_id="no_matching_failure_pattern",
                    observed_value=float(np.round(features.theil_sen_slope, 4)),
                    threshold=0.0,
                    passed=False,
                    plain_language="Signals did not meet confidence criteria for any single deterministic failure mode.",
                )
            ]
            return Diagnosis(
                primary=DiagnosisLabel.INCONCLUSIVE,
                secondary=[],
                confidence=0.90,
                evidence=inconclusive_ev,
                inconclusive_reason="Benchmark results do not match any deterministic failure pattern or exhibit conflicting signals.",
            )

        # Sort candidates by confidence descending
        candidates.sort(key=lambda c: c[1], reverse=True)
        primary_label, primary_conf, primary_ev = candidates[0]
        secondary_labels = [c[0] for c in candidates[1:]]

        # Assemble clean, non-contradictory evidence:
        # Supporting evidence for primary (and secondary) verdict, followed by ruled-out items
        curated_evidence: list[EvidenceItem] = list(primary_ev)

        for _, _, sec_ev in candidates[1:]:
            curated_evidence.extend(sec_ev)

        # Add clear 'ruled out' items for unselected failure modes
        from jitter.diagnosis.thresholds import (
            NOISE_MIN_OUTLIER_COUNT,
            THERMAL_THEIL_SEN_SLOPE,
            WARMUP_RATIO_THRESHOLD,
        )

        all_selected = {primary_label, *secondary_labels}

        if DiagnosisLabel.DELEGATE_FALLBACK not in all_selected and not is_delegate:
            curated_evidence.append(
                EvidenceItem(
                    rule_id="ruled_out_delegate_fallback",
                    observed_value=features.delegate_executed,
                    threshold=features.delegate_requested,
                    passed=True,
                    plain_language=(
                        f"Ruled out delegate fallback: requested '{features.delegate_requested}' executed on "
                        f"'{features.delegate_executed}' without CPU parity (Cliff's delta: {features.cpu_cliffs_delta:.3f})."
                    ),
                )
            )

        if DiagnosisLabel.THERMAL_THROTTLING not in all_selected and not is_thermal:
            curated_evidence.append(
                EvidenceItem(
                    rule_id="ruled_out_thermal_throttling",
                    observed_value=float(np.round(features.theil_sen_slope, 4)),
                    threshold=THERMAL_THEIL_SEN_SLOPE,
                    passed=True,
                    plain_language=(
                        f"Ruled out thermal throttling: degradation slope ({features.theil_sen_slope:.4f} ms/iter) "
                        f"and peak thermal status ({features.peak_thermal_status}) remained within baseline limits."
                    ),
                )
            )

        if DiagnosisLabel.BACKGROUND_NOISE not in all_selected and not is_noise:
            curated_evidence.append(
                EvidenceItem(
                    rule_id="ruled_out_background_noise",
                    observed_value=features.cold_outlier_count,
                    threshold=NOISE_MIN_OUTLIER_COUNT,
                    passed=True,
                    plain_language=(
                        f"Ruled out background noise: transient spikes were absent or not suppressed by quiet mode "
                        f"({features.cold_outlier_count} active outliers, {features.quiet_outlier_count} quiet outliers)."
                    ),
                )
            )

        if DiagnosisLabel.WARMUP_DRIFT not in all_selected and not is_warmup:
            curated_evidence.append(
                EvidenceItem(
                    rule_id="ruled_out_warmup_drift",
                    observed_value=float(np.round(features.warmup_ratio, 2)),
                    threshold=WARMUP_RATIO_THRESHOLD,
                    passed=True,
                    plain_language=(
                        f"Ruled out warmup drift: cold startup ratio ({features.warmup_ratio:.2f}x) "
                        "did not indicate substantial startup delay."
                    ),
                )
            )

        return Diagnosis(
            primary=primary_label,
            secondary=secondary_labels,
            confidence=primary_conf,
            evidence=curated_evidence,
            inconclusive_reason=None,
        )


def diagnose(runs_or_report: Report | list[RunEntry]) -> Diagnosis:
    """Convenience functional interface to DiagnosisEngine."""
    return DiagnosisEngine().diagnose(runs_or_report)
