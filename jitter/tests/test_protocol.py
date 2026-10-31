"""Tests for Android Wire Protocol v1 and AdbRunner integration."""

from __future__ import annotations

import pytest

from jitter.runners.adb import AdbRunner
from jitter.runners.fake_server import FakeRunnerServer
from jitter.schema.protocol import (
    RunnerCapabilities,
    RunnerInfoResponse,
    RunRequest,
    RunSafetyConstraints,
    TelemetrySnapshot,
)
from jitter.schema.runs import ProcessStart


def test_wire_protocol_schema_serialization() -> None:
    telemetry = TelemetrySnapshot(
        timestamp_ns=1_000_000_000,
        thermal_status=1,
        thermal_headroom=0.85,
        battery_temp_c=34.2,
        battery_pct=88.5,
        charging=False,
        screen_on=True,
    )
    caps = RunnerCapabilities(
        supported_delegates=["gpu", "cpu", "nnapi"],
        can_measure_headroom=True,
        can_control_screen=True,
    )
    info = RunnerInfoResponse(
        runner_version="0.1.0",
        device_model="Pixel 7",
        manufacturer="Google",
        soc="Tensor G2",
        android_api=34,
        capabilities=caps,
        current_telemetry=telemetry,
    )
    serialized = info.model_dump_json()
    deserialized = RunnerInfoResponse.model_validate_json(serialized)
    assert deserialized.device_model == "Pixel 7"
    assert deserialized.current_telemetry.battery_temp_c == 34.2
    assert "gpu" in deserialized.capabilities.supported_delegates


def test_fake_runner_server_end_to_end() -> None:
    with FakeRunnerServer() as server:
        runner = AdbRunner(base_url=f"http://127.0.0.1:{server.port}")

        # 1. Info endpoint
        info = runner.info()
        assert info.device_model == "Pixel 7 (Mock)"
        assert info.manufacturer == "Google"
        assert info.current_telemetry.battery_pct > 80.0

        # 2. Run endpoint
        spec = RunRequest(
            model_path="/data/local/tmp/mobilenet_v2.tflite",
            delegate="gpu",
            warmup_iters=5,
            measured_iters=30,
            process_start=ProcessStart.COLD,
            constraints=RunSafetyConstraints(min_battery_pct=20.0, max_battery_temp_c=45.0),
        )
        entry = runner.run(spec)
        assert len(entry.samples_ms) == 30
        assert entry.delegate_requested == "gpu"
        assert len(entry.timestamps_ns) == 30

        # 3. Rest endpoint
        rested = runner.rest(target_temp_c=32.0, max_wait_s=5.0)
        assert rested is True


def test_adb_runner_safety_constraint_rejection() -> None:
    with FakeRunnerServer() as server:
        runner = AdbRunner(base_url=f"http://127.0.0.1:{server.port}")

        # Spec requiring battery level higher than device has (85%)
        spec = RunRequest(
            model_path="/data/local/tmp/model.tflite",
            delegate="gpu",
            measured_iters=10,
            constraints=RunSafetyConstraints(
                min_battery_pct=95.0
            ),  # Will fail because mock has 85%
        )

        with pytest.raises(Exception, match="below minimum safe threshold"):
            runner.run(spec)
