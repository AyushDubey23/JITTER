from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest
from typer.testing import CliRunner

from jitter.cli import app
from jitter.schema.report import Report
from jitter.schema.runs import DiagnosisLabel
from jitter.synth.scenarios import generate_scenario_report
from jitter.synth.thermal import RCThermalModel, temperature_to_thermal_status


def test_thermal_model_physics() -> None:
    model = RCThermalModel(ambient_temp_c=25.0, initial_temp_c=28.0)
    # Under high load (6W for 25s), temperature must rise
    states = [model.step(power_w=6.0, dt_s=0.5) for _ in range(50)]
    assert states[-1].temperature_c > 28.0
    assert states[-1].temperature_c > states[0].temperature_c

    # When cooling down (idle 0.2W for 60s), temperature must fall
    cooling_states = model.cool_down(rest_duration_s=60.0, idle_power_w=0.2)
    assert cooling_states[-1].temperature_c < states[-1].temperature_c


def test_thermal_status_mapping() -> None:
    assert temperature_to_thermal_status(32.0) == 0  # NONE
    assert temperature_to_thermal_status(37.0) == 1  # LIGHT
    assert temperature_to_thermal_status(41.0) == 2  # MODERATE
    assert temperature_to_thermal_status(45.0) == 3  # SEVERE
    assert temperature_to_thermal_status(49.0) == 4  # CRITICAL
    assert temperature_to_thermal_status(53.0) == 5  # EMERGENCY
    assert temperature_to_thermal_status(58.0) == 6  # SHUTDOWN


@pytest.mark.parametrize(
    "mode",
    [
        DiagnosisLabel.WARMUP_DRIFT,
        DiagnosisLabel.THERMAL_THROTTLING,
        DiagnosisLabel.DELEGATE_FALLBACK,
        DiagnosisLabel.BACKGROUND_NOISE,
        DiagnosisLabel.REGRESSION,
        DiagnosisLabel.INCONCLUSIVE,
    ],
)
def test_all_scenarios_generate_valid_reports(mode: DiagnosisLabel) -> None:
    report = generate_scenario_report(mode, iterations=30, seed=123)
    assert report.diagnosis.primary == mode
    assert report.truth is not None
    assert report.truth.true_label == mode
    assert len(report.runs) == 5

    # Check serialization round-trip
    serialized = report.to_json()
    reloaded = Report.from_json(serialized)
    assert reloaded.diagnosis.primary == mode


def test_example_reports_validate_against_json_schema() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    schema_path = repo_root / "schema" / "report.schema.json"
    assert schema_path.exists(), "schema/report.schema.json must exist"

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)

    examples_dir = repo_root / "examples" / "reports"
    example_files = list(examples_dir.glob("*.json"))
    assert len(example_files) == 6, f"Expected 6 example reports, found {len(example_files)}"

    for file_path in example_files:
        report = Report.from_file(file_path)
        assert report.schema_version == "0.1.0"

        # Validate against raw JSON Schema
        raw_data = json.loads(file_path.read_text(encoding="utf-8"))
        errors = list(validator.iter_errors(raw_data))
        assert not errors, f"JSON Schema validation failed for {file_path.name}: {errors}"


def test_cli_simulate(tmp_path: Path) -> None:
    runner = CliRunner()
    out_file = tmp_path / "simulated.json"
    result = runner.invoke(
        app,
        [
            "simulate",
            "--failure-mode",
            "thermal_throttling",
            "--iterations",
            "20",
            "--output",
            str(out_file),
        ],
    )
    assert result.exit_code == 0
    assert out_file.exists()

    rep = Report.from_file(out_file)
    assert rep.diagnosis.primary == DiagnosisLabel.THERMAL_THROTTLING
