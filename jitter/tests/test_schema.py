from __future__ import annotations

import json
from pathlib import Path

import pytest

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
    RunnerCapabilities,
    RunnerInfoResponse,
    RunRequest,
    RunStatistics,
    TelemetrySeries,
    TelemetrySnapshot,
)
from scripts.gen_schema import check_schemas


def create_valid_run_entry(
    rerun: RerunType = RerunType.COLD_WARM,
    process_start: ProcessStart = ProcessStart.COLD,
    n: int = 5,
) -> RunEntry:
    t0 = 1_000_000_000
    timestamps_ns = [t0 + i * 20_000_000 for i in range(n)]
    samples_ms = [15.2 + i * 0.1 for i in range(n)]
    series = TelemetrySeries(
        thermal_status=[0] * n,
        thermal_headroom=[0.8] * n,
        battery_temp_c=[32.0 + i * 0.1 for i in range(n)],
        charging=[False] * n,
        screen_on=[True] * n,
    )
    return RunEntry(
        rerun=rerun,
        process_start=process_start,
        delegate_requested="gpu",
        delegate_executed="gpu",
        samples_ms=samples_ms,
        timestamps_ns=timestamps_ns,
        telemetry=series,
        notes=["healthy execution"],
    )


def create_valid_report() -> Report:
    model = ModelInfo(
        name="mobilenet_v2.tflite",
        sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        size_bytes=14200000,
    )
    device = DeviceInfo(
        model="Pixel 7",
        manufacturer="Google",
        soc="Tensor G2",
        android_api=34,
        runner_version="0.1.0",
    )
    config = BenchmarkConfig(
        warmup_iterations=5,
        measured_iterations=50,
        requested_delegate="gpu",
        input_spec={"input": [1, 224, 224, 3]},
        rest_policy={"cooldown_target_c": 32.0, "max_rest_s": 300},
    )
    diag = Diagnosis(
        primary=DiagnosisLabel.WARMUP_DRIFT,
        secondary=[],
        confidence=0.92,
        evidence=[
            EvidenceItem(
                rule_id="warmup_ratio",
                observed_value=4.2,
                threshold=1.5,
                passed=True,
                plain_language="Initial samples exhibit 4.2x higher latency than stable iterations.",
            )
        ],
    )
    runs = [
        create_valid_run_entry(rerun=RerunType.COLD_WARM, process_start=ProcessStart.COLD),
        create_valid_run_entry(rerun=RerunType.COLD_WARM, process_start=ProcessStart.WARM),
    ]
    stats = ReportStatistics(
        per_run={
            "cold": RunStatistics(
                n=5,
                n_eff=4.8,
                median=15.4,
                p95=15.6,
                p99=15.6,
                mad=0.2,
                ci_lower=15.1,
                ci_upper=15.7,
                method="stationary_block_bootstrap",
            )
        },
        comparisons=[
            PairwiseComparison(
                run_a="cold",
                run_b="warm",
                diff_estimate=-0.2,
                p_value=0.42,
                ci_lower=-0.5,
                ci_upper=0.1,
                effect_size=-0.1,
                effect_magnitude="negligible",
                method="block_permutation_test",
            )
        ],
    )
    return Report(
        schema_version="0.1.0",
        jitter_version="0.1.0",
        created_at="2026-10-28T12:00:00Z",
        source="device",
        model=model,
        device=device,
        config=config,
        runs=runs,
        diagnosis=diag,
        statistics=stats,
    )


def test_report_json_roundtrip() -> None:
    report = create_valid_report()
    json_str = report.to_json()
    reloaded = Report.from_json(json_str)

    assert reloaded.schema_version == report.schema_version
    assert reloaded.jitter_version == report.jitter_version
    assert reloaded.created_at == report.created_at
    assert reloaded.model.name == "mobilenet_v2.tflite"
    assert reloaded.diagnosis.primary == DiagnosisLabel.WARMUP_DRIFT
    assert len(reloaded.runs) == 2
    assert reloaded.diagnosis.confidence == 0.92
    assert reloaded.statistics is not None
    assert len(reloaded.statistics.comparisons) == 1


def test_report_file_roundtrip(tmp_path: Path) -> None:
    report = create_valid_report()
    report_file = tmp_path / "test_report.json"
    report.to_file(report_file)

    assert report_file.exists()
    reloaded = Report.from_file(report_file)
    assert reloaded == report


def test_inconclusive_validation() -> None:
    # Inconclusive must require inconclusive_reason
    with pytest.raises(ValueError, match="inconclusive_reason must be provided"):
        Diagnosis(
            primary=DiagnosisLabel.INCONCLUSIVE,
            confidence=0.3,
            evidence=[],
        )

    # Valid inconclusive
    inc_diag = Diagnosis(
        primary=DiagnosisLabel.INCONCLUSIVE,
        confidence=0.3,
        evidence=[],
        inconclusive_reason="High background noise masked delegate signals.",
    )
    assert inc_diag.primary == DiagnosisLabel.INCONCLUSIVE
    assert inc_diag.inconclusive_reason is not None


def test_run_entry_series_length_mismatch() -> None:
    # 2 timestamps vs 1 latency
    with pytest.raises(ValueError, match="Length mismatch"):
        RunEntry(
            rerun=RerunType.COLD_WARM,
            process_start=ProcessStart.COLD,
            delegate_requested="gpu",
            delegate_executed="gpu",
            samples_ms=[15.0],
            timestamps_ns=[1_000_000_000, 1_020_000_000],
            telemetry=TelemetrySeries(
                thermal_status=[0],
                thermal_headroom=[0.8],
                battery_temp_c=[32.0],
                charging=[False],
                screen_on=[True],
            ),
        )

    # Telemetry length mismatch
    with pytest.raises(ValueError, match="Telemetry length mismatch"):
        RunEntry(
            rerun=RerunType.COLD_WARM,
            process_start=ProcessStart.COLD,
            delegate_requested="gpu",
            delegate_executed="gpu",
            samples_ms=[15.0, 15.2],
            timestamps_ns=[1_000_000_000, 1_020_000_000],
            telemetry=TelemetrySeries(
                thermal_status=[0],  # only 1 item instead of 2
                thermal_headroom=[0.8, 0.8],
                battery_temp_c=[32.0, 32.1],
                charging=[False, False],
                screen_on=[True, True],
            ),
        )


def test_telemetry_non_monotonic_timestamps() -> None:
    with pytest.raises(ValueError, match="timestamps_ns must be strictly increasing"):
        RunEntry(
            rerun=RerunType.COLD_WARM,
            process_start=ProcessStart.COLD,
            delegate_requested="gpu",
            delegate_executed="gpu",
            samples_ms=[15.0, 15.2, 15.3],
            timestamps_ns=[1_000_000_000, 950_000_000, 1_040_000_000],
            telemetry=TelemetrySeries(
                thermal_status=[0, 0, 0],
                thermal_headroom=[0.8, 0.8, 0.8],
                battery_temp_c=[32.0, 32.1, 32.2],
                charging=[False, False, False],
                screen_on=[True, True, True],
            ),
        )


def test_telemetry_non_positive_latencies() -> None:
    with pytest.raises(ValueError, match="samples_ms\\[1\\] must be positive and finite"):
        RunEntry(
            rerun=RerunType.COLD_WARM,
            process_start=ProcessStart.COLD,
            delegate_requested="gpu",
            delegate_executed="gpu",
            samples_ms=[15.0, 0.0],
            timestamps_ns=[1_000_000_000, 1_020_000_000],
            telemetry=TelemetrySeries(
                thermal_status=[0, 0],
                thermal_headroom=[0.8, 0.8],
                battery_temp_c=[32.0, 32.1],
                charging=[False, False],
                screen_on=[True, True],
            ),
        )

    with pytest.raises(ValueError, match="samples_ms\\[1\\] must be positive and finite"):
        RunEntry(
            rerun=RerunType.COLD_WARM,
            process_start=ProcessStart.COLD,
            delegate_requested="gpu",
            delegate_executed="gpu",
            samples_ms=[15.0, -2.5],
            timestamps_ns=[1_000_000_000, 1_020_000_000],
            telemetry=TelemetrySeries(
                thermal_status=[0, 0],
                thermal_headroom=[0.8, 0.8],
                battery_temp_c=[32.0, 32.1],
                charging=[False, False],
                screen_on=[True, True],
            ),
        )


def test_report_from_json_field_errors() -> None:
    bad_json = json.dumps({"schema_version": "0.1.0"})
    with pytest.raises(ValueError, match="Report validation failed"):
        Report.from_json(bad_json)


def test_protocol_models_roundtrip() -> None:
    runner_info = RunnerInfoResponse(
        runner_version="0.1.0",
        device_model="Galaxy S23",
        manufacturer="Samsung",
        soc="Snapdragon 8 Gen 2",
        android_api=34,
        capabilities=RunnerCapabilities(
            supported_delegates=["cpu", "gpu", "nnapi"],
            can_measure_headroom=True,
            can_control_screen=False,
            safety_temp_ceiling_c=45.0,
            min_battery_pct=15.0,
        ),
        current_telemetry=TelemetrySnapshot(
            timestamp_ns=1_000_000_000,
            thermal_status=0,
            thermal_headroom=0.9,
            battery_temp_c=31.5,
            battery_pct=85.0,
            charging=False,
            screen_on=True,
        ),
    )
    raw_info = runner_info.model_dump_json()
    reloaded_info = RunnerInfoResponse.model_validate_json(raw_info)
    assert reloaded_info == runner_info

    run_req = RunRequest(
        model_path="/data/local/tmp/test_model.tflite",
        delegate="gpu",
        warmup_iters=5,
        measured_iters=50,
        process_start=ProcessStart.COLD,
    )
    assert run_req.measured_iters == 50
    assert run_req.delegate == "gpu"
    assert run_req.process_start == ProcessStart.COLD


def test_schema_drift_check() -> None:
    # Ensure generated schemas match what is in jitter/schema/
    assert check_schemas() is True
