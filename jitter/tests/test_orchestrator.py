from __future__ import annotations

import pytest

from jitter.orchestrator import execute_orchestrated_benchmark
from jitter.runners import DesktopOnnxRunner, SyntheticRunner
from jitter.schema.report import Report
from jitter.schema.runs import DiagnosisLabel


@pytest.mark.parametrize(
    "mode",
    [
        DiagnosisLabel.WARMUP_DRIFT,
        DiagnosisLabel.THERMAL_THROTTLING,
        DiagnosisLabel.DELEGATE_FALLBACK,
        DiagnosisLabel.BACKGROUND_NOISE,
        DiagnosisLabel.REGRESSION,
    ],
)
def test_orchestrator_pipeline_synthetic_modes(mode: DiagnosisLabel) -> None:
    runner = SyntheticRunner(fault_mode=mode, model_name="mobilenet_v2", seed=42)
    report = execute_orchestrated_benchmark(
        runner=runner,
        model_name="mobilenet_v2.tflite",
        requested_delegate="gpu",
        iterations=20,
        warmup_iterations=2,
    )

    assert isinstance(report, Report)
    assert len(report.runs) >= 4
    assert report.truth is not None
    assert report.truth.true_label == mode
    assert report.statistics is not None
    assert len(report.statistics.comparisons) >= 4


def test_orchestrator_desktop_onnx_pipeline() -> None:
    runner = DesktopOnnxRunner()
    report = execute_orchestrated_benchmark(
        runner=runner,
        model_name="test_mlp.onnx",
        requested_delegate="cpu",
        iterations=15,
        warmup_iterations=2,
    )

    assert isinstance(report, Report)
    assert report.source == "desktop"
    assert len(report.runs) >= 4
    assert report.statistics is not None


def test_orchestrator_battery_safety_guard() -> None:
    runner = SyntheticRunner(fault_mode=DiagnosisLabel.WARMUP_DRIFT)
    # Drain battery below 15% threshold
    runner.battery_pct = 10.0

    with pytest.raises(RuntimeError, match="Battery level 10.0% is below required 15.0%"):
        execute_orchestrated_benchmark(
            runner=runner,
            model_name="mobilenet_v2.tflite",
            iterations=10,
        )


def test_orchestrator_thermal_safety_guard() -> None:
    runner = SyntheticRunner(fault_mode=DiagnosisLabel.WARMUP_DRIFT)
    # Heat runner up beyond 45 C cutoff
    runner.thermal.current_temp_c = 48.0

    with pytest.raises(RuntimeError, match="Device temperature 48.0C exceeds safety ceiling 45.0C"):
        execute_orchestrated_benchmark(
            runner=runner,
            model_name="mobilenet_v2.tflite",
            iterations=10,
        )
