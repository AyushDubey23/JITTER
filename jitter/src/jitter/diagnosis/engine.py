"""Root-cause diagnosis engine evaluating multi-condition evidence."""

from __future__ import annotations

from typing import TYPE_CHECKING

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
        all_evidence: list[EvidenceItem] = []
        candidates: list[tuple[DiagnosisLabel, float, list[EvidenceItem]]] = []

        # Warmup drift
        is_warmup, warmup_conf, warmup_ev = evaluate_warmup_drift(features)
        all_evidence.extend(warmup_ev)
        if is_warmup:
            candidates.append((DiagnosisLabel.WARMUP_DRIFT, warmup_conf, warmup_ev))

        # Thermal throttling
        is_thermal, thermal_conf, thermal_ev = evaluate_thermal_throttling(features)
        all_evidence.extend(thermal_ev)
        if is_thermal:
            candidates.append((DiagnosisLabel.THERMAL_THROTTLING, thermal_conf, thermal_ev))

        # Delegate fallback
        is_delegate, delegate_conf, delegate_ev = evaluate_delegate_fallback(features)
        all_evidence.extend(delegate_ev)
        if is_delegate:
            candidates.append((DiagnosisLabel.DELEGATE_FALLBACK, delegate_conf, delegate_ev))

        # Background noise
        is_noise, noise_conf, noise_ev = evaluate_background_noise(features)
        all_evidence.extend(noise_ev)
        if is_noise:
            candidates.append((DiagnosisLabel.BACKGROUND_NOISE, noise_conf, noise_ev))

        # Uniform regression (only if not better explained by thermal or delegate)
        if not is_thermal and not is_delegate:
            is_regression, reg_conf, reg_ev = evaluate_regression(features)
            all_evidence.extend(reg_ev)
            if is_regression:
                candidates.append((DiagnosisLabel.REGRESSION, reg_conf, reg_ev))

        # 3. Classify outcome
        if not candidates:
            return Diagnosis(
                primary=DiagnosisLabel.INCONCLUSIVE,
                secondary=[],
                confidence=0.90,
                evidence=all_evidence,
                inconclusive_reason="Benchmark results do not match any deterministic failure pattern or exhibit conflicting signals.",
            )

        # Sort candidates by confidence descending
        candidates.sort(key=lambda c: c[1], reverse=True)
        primary_label, primary_conf, _ = candidates[0]
        secondary_labels = [c[0] for c in candidates[1:]]

        return Diagnosis(
            primary=primary_label,
            secondary=secondary_labels,
            confidence=primary_conf,
            evidence=all_evidence,
            inconclusive_reason=None,
        )


def diagnose(runs_or_report: Report | list[RunEntry]) -> Diagnosis:
    """Convenience functional interface to DiagnosisEngine."""
    return DiagnosisEngine().diagnose(runs_or_report)
