"""Tests for diagnosis feature extraction, rule evaluation, engine, and metrics."""

from __future__ import annotations

import pytest

from jitter.diagnosis.engine import DiagnosisEngine, diagnose
from jitter.diagnosis.features import DiagnosticFeatures, extract_features
from jitter.schema.runs import DiagnosisLabel
from jitter.synth.scenarios import generate_scenario_report
from jitter.validation.metrics import (
    compute_brier_score,
    compute_classification_metrics,
    compute_confusion_matrix,
    compute_expected_calibration_error,
    compute_macro_f1,
)


def test_feature_extraction_structure() -> None:
    report = generate_scenario_report("warmup_drift", iterations=30, seed=42)
    feat = extract_features(report)
    assert isinstance(feat, DiagnosticFeatures)
    assert feat.total_samples == 30 * 5
    assert feat.min_n_eff > 0.0
    assert feat.warmup_detected is True
    assert feat.warmup_ratio > 1.2


def test_warmup_drift_classification() -> None:
    report = generate_scenario_report("warmup_drift", iterations=40, seed=123)
    diag = diagnose(report)
    assert diag.primary == DiagnosisLabel.WARMUP_DRIFT
    assert diag.confidence >= 0.85
    rule_ids = [e.rule_id for e in diag.evidence]
    assert "warmup_ratio_threshold" in rule_ids


def test_thermal_throttling_classification() -> None:
    report = generate_scenario_report("thermal_throttling", iterations=40, seed=456)
    diag = diagnose(report)
    assert diag.primary == DiagnosisLabel.THERMAL_THROTTLING
    assert diag.confidence >= 0.80
    rule_ids = [e.rule_id for e in diag.evidence]
    assert "theil_sen_drift_slope" in rule_ids


def test_delegate_fallback_classification() -> None:
    report = generate_scenario_report("delegate_fallback", iterations=40, seed=789)
    diag = diagnose(report)
    assert diag.primary == DiagnosisLabel.DELEGATE_FALLBACK
    assert diag.confidence >= 0.90
    rule_ids = [e.rule_id for e in diag.evidence]
    assert any("delegate" in r for r in rule_ids)


def test_background_noise_classification() -> None:
    report = generate_scenario_report("background_noise", iterations=40, seed=101)
    diag = diagnose(report)
    assert diag.primary == DiagnosisLabel.BACKGROUND_NOISE
    assert diag.confidence >= 0.80
    rule_ids = [e.rule_id for e in diag.evidence]
    assert "quiet_mode_p99_suppression" in rule_ids


def test_regression_classification() -> None:
    report = generate_scenario_report("regression", iterations=40, seed=202)
    diag = diagnose(report)
    assert diag.primary == DiagnosisLabel.REGRESSION
    assert diag.confidence >= 0.85
    rule_ids = [e.rule_id for e in diag.evidence]
    assert any(r in rule_ids for r in ["warmup_absence", "thermal_absence", "delegate_healthy"])


def test_inconclusive_classification() -> None:
    report = generate_scenario_report("inconclusive", iterations=20, seed=303)
    diag = diagnose(report)
    assert diag.primary == DiagnosisLabel.INCONCLUSIVE
    assert diag.confidence >= 0.85
    assert diag.inconclusive_reason is not None
    assert len(diag.inconclusive_reason) > 10


def test_validation_metrics_properties() -> None:
    labels = ["a", "b", "c"]
    y_true = ["a", "a", "b", "c", "c"]
    y_pred = ["a", "b", "b", "c", "c"]
    confs = [0.9, 0.7, 0.8, 0.95, 0.85]

    cm = compute_confusion_matrix(y_true, y_pred, labels)
    assert cm.shape == (3, 3)
    assert cm[0, 0] == 1  # 1 true 'a' predicted as 'a'
    assert cm[0, 1] == 1  # 1 true 'a' predicted as 'b'
    assert cm[1, 1] == 1  # 1 true 'b' predicted as 'b'
    assert cm[2, 2] == 2  # 2 true 'c' predicted as 'c'

    metrics = compute_classification_metrics(y_true, y_pred, labels)
    assert metrics["c"]["precision"] == 1.0
    assert metrics["c"]["recall"] == 1.0
    assert metrics["c"]["f1"] == 1.0

    macro_f1 = compute_macro_f1(y_true, y_pred, labels)
    assert 0.0 < macro_f1 <= 1.0

    brier = compute_brier_score(y_true, y_pred, confs)
    assert 0.0 <= brier <= 1.0

    ece = compute_expected_calibration_error(y_true, y_pred, confs, n_bins=5)
    assert 0.0 <= ece <= 1.0
