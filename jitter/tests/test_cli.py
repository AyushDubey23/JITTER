"""Tests for Typer CLI commands in jitter.cli."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from jitter.cli import app

runner = CliRunner()


def test_cli_version() -> None:
    res = runner.invoke(app, ["version"])
    assert res.exit_code == 0
    assert "jitter" in res.stdout


def test_cli_schema_report() -> None:
    res = runner.invoke(app, ["schema", "report"])
    assert res.exit_code == 0
    data = json.loads(res.stdout)
    assert data["title"] == "Report"


def test_cli_schema_protocol() -> None:
    res = runner.invoke(app, ["schema", "protocol"])
    assert res.exit_code == 0
    data = json.loads(res.stdout)
    assert data["title"] == "RunnerInfoResponse"


def test_cli_devices_json() -> None:
    res = runner.invoke(app, ["devices", "--json"])
    assert res.exit_code == 0
    data = json.loads(res.stdout)
    assert isinstance(data, list)


def test_cli_simulate(tmp_path: Path) -> None:
    out_file = tmp_path / "sim.json"
    res = runner.invoke(
        app,
        [
            "simulate",
            "--failure-mode",
            "thermal_throttling",
            "--model",
            "mobilenet_v2",
            "--iterations",
            "20",
            "--output",
            str(out_file),
        ],
    )
    assert res.exit_code == 0
    assert out_file.exists()
    content = json.loads(out_file.read_text(encoding="utf-8"))
    assert content["source"] == "synthetic"
    assert len(content["runs"]) == 5


def test_cli_diagnose() -> None:
    report_path = Path(__file__).resolve().parent.parent / "examples" / "reports" / "warmup_drift.json"
    res = runner.invoke(app, ["diagnose", str(report_path)])
    assert res.exit_code == 0
    assert "WARMUP_DRIFT" in res.stdout


def test_cli_diagnose_json() -> None:
    report_path = Path(__file__).resolve().parent.parent / "examples" / "reports" / "warmup_drift.json"
    res = runner.invoke(app, ["diagnose", str(report_path), "--json"])
    assert res.exit_code == 0
    diag_data = json.loads(res.stdout)
    assert diag_data["primary"] == "warmup_drift"
    assert diag_data["confidence"] > 0.80


def test_cli_compare() -> None:
    report_path = Path(__file__).resolve().parent.parent / "examples" / "reports" / "warmup_drift.json"
    res = runner.invoke(app, ["compare", str(report_path), "--run-a", "0", "--run-b", "1"])
    assert res.exit_code == 0
    assert "Robust Two-Sample Comparison" in res.stdout
    assert "Hodges-Lehmann Shift" in res.stdout


def test_cli_report_formats(tmp_path: Path) -> None:
    report_path = Path(__file__).resolve().parent.parent / "examples" / "reports" / "warmup_drift.json"

    # Terminal format
    res_term = runner.invoke(app, ["report", str(report_path), "--format", "terminal"])
    assert res_term.exit_code == 0
    assert "Controlled Reruns" in res_term.stdout

    # Markdown format
    md_file = tmp_path / "report.md"
    res_md = runner.invoke(app, ["report", str(report_path), "--format", "markdown", "-o", str(md_file)])
    assert res_md.exit_code == 0
    assert md_file.exists()
    assert "# Jitter Benchmark Report" in md_file.read_text(encoding="utf-8")

    # HTML format
    html_file = tmp_path / "report.html"
    res_html = runner.invoke(app, ["report", str(report_path), "--format", "html", "-o", str(html_file)])
    assert res_html.exit_code == 0
    assert html_file.exists()
    assert "<!DOCTYPE html>" in html_file.read_text(encoding="utf-8")


def test_cli_run_synthetic(tmp_path: Path) -> None:
    out_file = tmp_path / "run_out.json"
    res = runner.invoke(
        app,
        [
            "run",
            "--runner",
            "synthetic",
            "--model",
            "mobilenet_v2",
            "--iterations",
            "20",
            "--output",
            str(out_file),
        ],
    )
    assert res.exit_code == 0
    assert out_file.exists()
    assert "Controlled Reruns" in res.stdout
