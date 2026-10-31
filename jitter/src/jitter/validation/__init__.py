"""Validation harness and calibration metrics for Jitter diagnosis engine."""

from jitter.validation.harness import ValidationResults, run_validation_suite
from jitter.validation.metrics import (
    compute_brier_score,
    compute_classification_metrics,
    compute_confusion_matrix,
    compute_expected_calibration_error,
    compute_macro_f1,
)
from jitter.validation.report_md import generate_diagnosis_markdown, write_diagnosis_report

__all__ = [
    "ValidationResults",
    "run_validation_suite",
    "compute_confusion_matrix",
    "compute_classification_metrics",
    "compute_macro_f1",
    "compute_brier_score",
    "compute_expected_calibration_error",
    "generate_diagnosis_markdown",
    "write_diagnosis_report",
]
