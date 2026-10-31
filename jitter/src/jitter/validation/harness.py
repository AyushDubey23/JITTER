"""Validation harness generating scenarios and verifying diagnosis accuracy."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from jitter.diagnosis.engine import DiagnosisEngine
from jitter.schema.runs import DiagnosisLabel
from jitter.synth.scenarios import generate_scenario_report
from jitter.validation.metrics import (
    compute_brier_score,
    compute_classification_metrics,
    compute_confusion_matrix,
    compute_expected_calibration_error,
    compute_macro_f1,
)

CANONICAL_LABELS: list[str] = [
    DiagnosisLabel.WARMUP_DRIFT.value,
    DiagnosisLabel.THERMAL_THROTTLING.value,
    DiagnosisLabel.DELEGATE_FALLBACK.value,
    DiagnosisLabel.BACKGROUND_NOISE.value,
    DiagnosisLabel.REGRESSION.value,
    DiagnosisLabel.INCONCLUSIVE.value,
]


@dataclass
class ValidationResults:
    """Summary of validation evaluation across benchmark scenarios."""

    total_scenarios: int
    scenarios_per_label: int
    labels: list[str]
    confusion_matrix: list[list[int]]
    per_class_metrics: dict[str, dict[str, float]]
    macro_f1: float
    brier_score: float
    ece: float
    elapsed_seconds: float
    mean_diagnose_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_scenarios": self.total_scenarios,
            "scenarios_per_label": self.scenarios_per_label,
            "labels": self.labels,
            "confusion_matrix": self.confusion_matrix,
            "per_class_metrics": self.per_class_metrics,
            "macro_f1": self.macro_f1,
            "brier_score": self.brier_score,
            "ece": self.ece,
            "elapsed_seconds": self.elapsed_seconds,
            "mean_diagnose_ms": self.mean_diagnose_ms,
        }


def run_validation_suite(
    scenarios_per_label: int = 50,
    base_seed: int = 1000,
) -> ValidationResults:
    """Run full synthetic benchmark validation across all 6 diagnosis labels."""
    engine = DiagnosisEngine()
    y_true: list[str] = []
    y_pred: list[str] = []
    confidences: list[float] = []

    t0 = time.perf_counter()
    diagnose_durations: list[float] = []

    models = ["mobilenet_v2", "resnet50", "bert_tiny"]

    for label_str in CANONICAL_LABELS:
        for i in range(scenarios_per_label):
            seed = base_seed + i * 37 + hash(label_str) % 10000
            model = models[i % len(models)]
            report = generate_scenario_report(
                label=label_str,
                model_name=model,
                iterations=50,
                seed=seed,
            )

            t_diag0 = time.perf_counter()
            diagnosis = engine.diagnose(report)
            diagnose_durations.append((time.perf_counter() - t_diag0) * 1000.0)

            true_label = report.truth.true_label.value if report.truth else label_str
            pred_label = diagnosis.primary.value

            y_true.append(true_label)
            y_pred.append(pred_label)
            confidences.append(diagnosis.confidence)

    total_time = time.perf_counter() - t0
    mean_diag_ms = (
        float(sum(diagnose_durations) / len(diagnose_durations)) if diagnose_durations else 0.0
    )

    cm = compute_confusion_matrix(y_true, y_pred, CANONICAL_LABELS)
    per_class = compute_classification_metrics(y_true, y_pred, CANONICAL_LABELS)
    macro_f1 = compute_macro_f1(y_true, y_pred, CANONICAL_LABELS)
    brier = compute_brier_score(y_true, y_pred, confidences)
    ece = compute_expected_calibration_error(y_true, y_pred, confidences)

    return ValidationResults(
        total_scenarios=len(y_true),
        scenarios_per_label=scenarios_per_label,
        labels=CANONICAL_LABELS,
        confusion_matrix=cm.tolist(),
        per_class_metrics=per_class,
        macro_f1=macro_f1,
        brier_score=brier,
        ece=ece,
        elapsed_seconds=total_time,
        mean_diagnose_ms=mean_diag_ms,
    )
