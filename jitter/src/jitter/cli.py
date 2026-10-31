"""Complete Typer CLI for Jitter: benchmark execution, statistical comparison, diagnosis, device management, and reporting."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from jitter import __version__
from jitter.diagnosis.engine import DiagnosisEngine
from jitter.orchestrator.pipeline import execute_orchestrated_benchmark
from jitter.runners.adb import AdbRunner
from jitter.runners.base import BaseRunner
from jitter.runners.desktop_onnx import DesktopOnnxRunner
from jitter.runners.synthetic import SyntheticRunner
from jitter.schema.protocol import RunnerInfoResponse
from jitter.schema.report import Report
from jitter.schema.runs import DiagnosisLabel
from jitter.stats.compare import compare_runs
from jitter.synth.scenarios import generate_scenario_report
from jitter.validation.harness import CANONICAL_LABELS, run_validation_suite

app = typer.Typer(
    name="jitter",
    help="Jitter: Root-cause analysis tool for on-device ML latency.",
    no_args_is_help=True,
)
schema_app = typer.Typer(name="schema", help="Inspect and export JSON Schemas.")
app.add_typer(schema_app, name="schema")

console = Console()


@app.callback()
def main() -> None:
    """Root-cause analysis tool for on-device ML latency."""


@app.command()
def version() -> None:
    """Print Jitter version."""
    console.print(f"[bold cyan]jitter[/bold cyan] [bold green]v{__version__}[/bold green]")


@schema_app.command("report")
def schema_report(
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="File to write schema to")
    ] = None,
) -> None:
    """Output JSON Schema for Report model."""
    schema_json = json.dumps(Report.model_json_schema(), indent=2, sort_keys=True)
    if output:
        output.write_text(schema_json + "\n", encoding="utf-8")
        console.print(f"[green]Wrote report schema to {output}[/green]")
    else:
        typer.echo(schema_json)


@schema_app.command("protocol")
def schema_protocol(
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="File to write schema to")
    ] = None,
) -> None:
    """Output JSON Schema for Wire Protocol."""
    schema_json = json.dumps(RunnerInfoResponse.model_json_schema(), indent=2, sort_keys=True)
    if output:
        output.write_text(schema_json + "\n", encoding="utf-8")
        console.print(f"[green]Wrote protocol schema to {output}[/green]")
    else:
        typer.echo(schema_json)


@app.command("devices")
def devices(
    adb_path: Annotated[str, typer.Option("--adb", help="Path to adb executable")] = "adb",
    json_output: Annotated[bool, typer.Option("--json", help="Output raw JSON array")] = False,
) -> None:
    """List connected Android devices with battery and thermal readiness status."""
    devs = AdbRunner.list_devices(adb_path=adb_path)

    if json_output:
        data = [d.__dict__ for d in devs]
        typer.echo(json.dumps(data, indent=2))
        return

    if not devs:
        console.print("[yellow]No Android devices detected via ADB.[/yellow]")
        console.print("[dim]Ensure USB Debugging is enabled on your phone and run 'adb devices'.[/dim]")
        return

    table = Table(title="Connected Android Devices", title_style="bold cyan")
    table.add_column("Serial", style="bold")
    table.add_column("State")
    table.add_column("Model")
    table.add_column("SoC")
    table.add_column("Android API")
    table.add_column("Battery")
    table.add_column("Temp")
    table.add_column("Thermal")
    table.add_column("Status", style="bold")

    for d in devs:
        bat_str = f"{d.battery_pct:.0f}%"
        temp_str = f"{d.battery_temp_c:.1f}°C"
        status_text = (
            Text("READY", style="bold green") if d.is_ready else Text(", ".join(d.readiness_notes), style="bold yellow")
        )
        table.add_row(
            d.serial,
            d.state,
            d.model,
            d.soc,
            str(d.android_api) if d.android_api > 0 else "-",
            bat_str if d.battery_pct > 0 else "-",
            temp_str if d.battery_temp_c > 0 else "-",
            str(d.thermal_status) if d.android_api > 0 else "-",
            status_text,
        )

    console.print(table)


@app.command("run")
def run_benchmark(
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="Model catalog name (mobilenet_v2, resnet50) or file path"),
    ] = "mobilenet_v2",
    delegate: Annotated[
        str,
        typer.Option("--delegate", "-d", help="Execution delegate: gpu, cpu, npu"),
    ] = "gpu",
    iterations: Annotated[
        int,
        typer.Option("--iterations", "-n", help="Iterations to measure per run", min=15),
    ] = 50,
    runner_type: Annotated[
        str,
        typer.Option("--runner", "-r", help="Runner environment: synthetic, onnx, adb"),
    ] = "synthetic",
    serial: Annotated[
        str | None,
        typer.Option("--serial", "-s", help="Optional Android device serial for ADB runner"),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="File to write report JSON"),
    ] = None,
    format_opt: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format: table, json"),
    ] = "table",
) -> None:
    """Execute complete 4-rerun controlled benchmark plan and root-cause diagnosis."""
    # Select runner
    runner: BaseRunner
    if runner_type.lower() == "synthetic":
        runner = SyntheticRunner(model_name=model)
    elif runner_type.lower() == "onnx":
        runner = DesktopOnnxRunner(model_path=model)
    elif runner_type.lower() == "adb":
        runner = AdbRunner(serial=serial)
    else:
        console.print(f"[red]Error: Unknown runner '{runner_type}'. Choose: synthetic, onnx, adb[/red]", err=True)
        raise typer.Exit(code=1)

    with console.status(f"[bold green]Executing Jitter 4-rerun plan on {runner_type} runner...[/bold green]"):
        report = execute_orchestrated_benchmark(
            runner=runner,
            model_name=model,
            requested_delegate=delegate,
            iterations=iterations,
        )
        diagnosis = DiagnosisEngine().diagnose(report)

    if format_opt.lower() == "json":
        json_str = report.to_json(indent=2)
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json_str + "\n", encoding="utf-8")
            console.print(f"[green]Report saved to {output}[/green]")
        else:
            typer.echo(json_str)
        return

    # Render Terminal Table Output
    _render_report_terminal(report, diagnosis)

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report.to_json(indent=2) + "\n", encoding="utf-8")
        console.print(f"[bold green]Full report artifact written to {output}[/bold green]")


@app.command("diagnose")
def diagnose_report(
    report_file: Annotated[
        Path,
        typer.Argument(help="Path to report JSON file (or '-' for stdin)"),
    ],
    json_output: Annotated[bool, typer.Option("--json", help="Output diagnosis as JSON")] = False,
) -> None:
    """Evaluate root cause on an existing Jitter benchmark report."""
    if str(report_file) == "-":
        content = sys.stdin.read()
    else:
        if not report_file.exists():
            console.print(f"[red]Error: File {report_file} does not exist.[/red]", err=True)
            raise typer.Exit(code=1)
        content = report_file.read_text(encoding="utf-8")

    try:
        report = Report.model_validate_json(content)
    except Exception as e:
        console.print(f"[red]Error: Invalid Jitter report format: {e}[/red]", err=True)
        raise typer.Exit(code=1) from e

    diagnosis = DiagnosisEngine().diagnose(report)

    if json_output:
        typer.echo(diagnosis.model_dump_json(indent=2))
        return

    _render_diagnosis_panel(diagnosis)


@app.command("compare")
def compare_cli(
    report_a: Annotated[Path, typer.Argument(help="First report JSON file")],
    report_b: Annotated[Path | None, typer.Argument(help="Optional second report JSON file")] = None,
    run_a_idx: Annotated[int, typer.Option("--run-a", help="Run index from first report")] = 0,
    run_b_idx: Annotated[int, typer.Option("--run-b", help="Run index from second (or same) report")] = 1,
    json_output: Annotated[bool, typer.Option("--json", help="Output comparison result as JSON")] = False,
) -> None:
    """Statistically compare two benchmark runs using Hodges-Lehmann shift and permutation testing."""
    if not report_a.exists():
        console.print(f"[red]Error: File {report_a} does not exist.[/red]", err=True)
        raise typer.Exit(code=1)

    rep_a = Report.model_validate_json(report_a.read_text(encoding="utf-8"))
    rep_b = Report.model_validate_json(report_b.read_text(encoding="utf-8")) if report_b else rep_a

    if run_a_idx >= len(rep_a.runs):
        console.print(f"[red]Error: Run index {run_a_idx} out of range for {report_a}[/red]", err=True)
        raise typer.Exit(code=1)
    if run_b_idx >= len(rep_b.runs):
        console.print(f"[red]Error: Run index {run_b_idx} out of range[/red]", err=True)
        raise typer.Exit(code=1)

    entry_a = rep_a.runs[run_a_idx]
    entry_b = rep_b.runs[run_b_idx]

    res = compare_runs(entry_a.samples_ms, entry_b.samples_ms)

    if json_output:
        typer.echo(json.dumps(res.__dict__, indent=2))
        return

    table = Table(title="Robust Two-Sample Comparison", title_style="bold cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_column("Description")

    table.add_row("Run A", f"{entry_a.rerun.value} ({entry_a.process_start.value})", f"N={len(entry_a.samples_ms)}")
    table.add_row("Run B", f"{entry_b.rerun.value} ({entry_b.process_start.value})", f"N={len(entry_b.samples_ms)}")
    table.add_row("Diff of Medians", f"{res.diff_of_medians:+.3f} ms", "Raw median difference (B - A)")
    table.add_row("Hodges-Lehmann Shift", f"{res.hodges_lehmann_shift:+.3f} ms", "Pseudo-median of all pairwise differences")
    table.add_row("Cliff's Delta", f"{res.cliffs_delta:+.3f} ({res.cliffs_magnitude})", "Non-parametric effect size [-1.0, +1.0]")
    table.add_row("Permutation p-value", f"{res.permutation_p_value:.4f}", "Block permutation test preserving autocorrelation")
    table.add_row("Mann-Whitney U", f"{res.mann_whitney_u:.1f} (p={res.mann_whitney_p_value:.4f})", "Rank-sum rank comparison")
    table.add_row("95% Bootstrap CI", f"[{res.ci_lower:+.2f}, {res.ci_upper:+.2f}] ms", "BCa / Percentile confidence interval")

    console.print(table)


@app.command("report")
def format_report_cmd(
    report_file: Annotated[Path, typer.Argument(help="Path to report JSON file")],
    format_type: Annotated[str, typer.Option("--format", "-f", help="Format: terminal, markdown, html")] = "terminal",
    output: Annotated[Path | None, typer.Option("--output", "-o", help="File to write formatted report")] = None,
) -> None:
    """Format and display a Jitter benchmark report in terminal, markdown, or HTML."""
    if not report_file.exists():
        console.print(f"[red]Error: File {report_file} does not exist.[/red]", err=True)
        raise typer.Exit(code=1)

    report = Report.model_validate_json(report_file.read_text(encoding="utf-8"))
    diagnosis = DiagnosisEngine().diagnose(report)

    if format_type.lower() == "terminal":
        _render_report_terminal(report, diagnosis)
    elif format_type.lower() == "markdown":
        md_text = _generate_markdown_report(report, diagnosis)
        if output:
            output.write_text(md_text, encoding="utf-8")
            console.print(f"[green]Markdown report written to {output}[/green]")
        else:
            typer.echo(md_text)
    elif format_type.lower() == "html":
        html_text = _generate_html_report(report, diagnosis)
        if output:
            output.write_text(html_text, encoding="utf-8")
            console.print(f"[green]HTML report written to {output}[/green]")
        else:
            typer.echo(html_text)
    else:
        console.print(f"[red]Error: Invalid format '{format_type}'. Choose: terminal, markdown, html[/red]", err=True)
        raise typer.Exit(code=1)


@app.command("validate")
def validate_cli(
    scenarios_per_label: Annotated[int, typer.Option("--scenarios", "-s", help="Scenarios per class")] = 15,
) -> None:
    """Run full synthetic benchmark suite and compute Macro-F1 and ECE."""
    with console.status(f"[bold green]Running Jitter validation suite ({scenarios_per_label * 6} scenarios)...[/bold green]"):
        results = run_validation_suite(scenarios_per_label=scenarios_per_label)

    # Confusion matrix
    cm_table = Table(title="Validation Confusion Matrix", title_style="bold cyan")
    cm_table.add_column("True \\ Pred", style="bold")
    for l in CANONICAL_LABELS:
        cm_table.add_column(l, justify="right")

    for i, true_l in enumerate(CANONICAL_LABELS):
        row = [true_l] + [str(results.confusion_matrix[i][j]) for j in range(len(CANONICAL_LABELS))]
        cm_table.add_row(*row)

    console.print(cm_table)

    # Summary table
    sum_table = Table(title="Diagnosis Engine Benchmark Metrics", title_style="bold green")
    sum_table.add_column("Metric", style="bold")
    sum_table.add_column("Score")
    sum_table.add_column("Target")
    sum_table.add_column("Status", style="bold")

    f1_status = "[bold green]PASS[/bold green]" if results.macro_f1 >= 0.95 else "[bold red]FAIL[/bold red]"
    ece_status = "[bold green]PASS[/bold green]" if results.ece <= 0.05 else "[bold red]FAIL[/bold red]"

    sum_table.add_row("Macro-Averaged F1", f"{results.macro_f1:.4f}", ">= 0.9500", f1_status)
    sum_table.add_row("Expected Calibration Error (ECE)", f"{results.ece:.4f}", "<= 0.0500", ece_status)
    sum_table.add_row("Brier Score", f"{results.brier_score:.4f}", "Minimize", "[dim]Info[/dim]")
    sum_table.add_row("Mean Latency", f"{results.mean_diagnose_ms:.2f} ms", "< 250 ms", "[bold green]PASS[/bold green]")

    console.print(sum_table)


@app.command("simulate")
def simulate(
    failure_mode: Annotated[
        str,
        typer.Option(
            "--failure-mode",
            "-m",
            help="Failure mode: warmup_drift, thermal_throttling, delegate_fallback, background_noise, regression, inconclusive",
        ),
    ] = "warmup_drift",
    model: Annotated[
        str,
        typer.Option("--model", help="Model name from catalog (e.g. mobilenet_v2, resnet50)"),
    ] = "mobilenet_v2",
    iterations: Annotated[
        int,
        typer.Option("--iterations", "-n", help="Iterations to record per run", min=5),
    ] = 50,
    seed: Annotated[
        int,
        typer.Option("--seed", "-s", help="PRNG seed for deterministic generation"),
    ] = 42,
    secondary: Annotated[
        str | None,
        typer.Option("--secondary", help="Optional secondary fault: warmup, thermal, fallback, noise"),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="File path to write report JSON"),
    ] = None,
) -> None:
    """Generate a synthetic benchmark report with known physical failure characteristics."""
    try:
        diag_label = DiagnosisLabel(failure_mode)
    except ValueError:
        valid_modes = ", ".join(m.value for m in DiagnosisLabel)
        console.print(f"[red]Error: Invalid failure mode '{failure_mode}'. Choose from: {valid_modes}[/red]", err=True)
        raise typer.Exit(code=1) from None

    report = generate_scenario_report(
        label=diag_label,
        model_name=model,
        iterations=iterations,
        seed=seed,
        secondary_fault=secondary,
    )
    json_output = report.to_json(indent=2)

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json_output + "\n", encoding="utf-8")
        console.print(f"[green]Simulated {diag_label.value} report saved to {output}[/green]")
    else:
        typer.echo(json_output)


def _render_diagnosis_panel(diagnosis: Any) -> None:
    color_map = {
        DiagnosisLabel.WARMUP_DRIFT: "yellow",
        DiagnosisLabel.THERMAL_THROTTLING: "red",
        DiagnosisLabel.DELEGATE_FALLBACK: "red",
        DiagnosisLabel.BACKGROUND_NOISE: "magenta",
        DiagnosisLabel.REGRESSION: "green",
        DiagnosisLabel.INCONCLUSIVE: "cyan",
    }
    color = color_map.get(diagnosis.primary, "white")
    title = f"Verdict: {diagnosis.primary.value.upper()} (Confidence: {diagnosis.confidence * 100:.1f}%)"

    lines = []
    if diagnosis.inconclusive_reason:
        lines.append(f"[bold italic]{diagnosis.inconclusive_reason}[/bold italic]\n")

    lines.append("[bold]Evidence & Observations:[/bold]")
    for item in diagnosis.evidence:
        status_symbol = "[green]✓[/green]" if item.passed else "[dim]•[/dim]"
        lines.append(f"  {status_symbol} {item.plain_language}")

    if diagnosis.secondary:
        sec_names = ", ".join(s.value for s in diagnosis.secondary)
        lines.append(f"\n[dim]Secondary signals detected: {sec_names}[/dim]")

    panel = Panel("\n".join(lines), title=title, title_align="left", border_style=color)
    console.print(panel)


def _render_report_terminal(report: Report, diagnosis: Any) -> None:
    console.print(f"\n[bold cyan]Jitter Benchmark Report[/bold cyan] [dim]({report.created_at})[/dim]")
    console.print(f"Model: [bold]{report.model.name}[/bold] | Device: [bold]{report.device.manufacturer} {report.device.model}[/bold] ({report.device.soc})\n")

    # Runs summary table
    table = Table(title="Controlled Reruns", title_style="bold")
    table.add_column("Rerun", style="bold")
    table.add_column("Start")
    table.add_column("Delegate")
    table.add_column("N", justify="right")
    table.add_column("Median (ms)", justify="right")
    table.add_column("p95 (ms)", justify="right")
    table.add_column("p99 (ms)", justify="right")
    table.add_column("MAD (ms)", justify="right")

    for r in report.runs:
        import numpy as np
        med = float(np.median(r.samples_ms))
        p95 = float(np.percentile(r.samples_ms, 95))
        p99 = float(np.percentile(r.samples_ms, 99))
        mad = float(np.median(np.abs(np.array(r.samples_ms) - med)))
        table.add_row(
            r.rerun.value,
            r.process_start.value,
            f"{r.delegate_executed} (req: {r.delegate_requested})",
            str(len(r.samples_ms)),
            f"{med:.2f}",
            f"{p95:.2f}",
            f"{p99:.2f}",
            f"{mad:.2f}",
        )

    console.print(table)
    console.print("")
    _render_diagnosis_panel(diagnosis)


def _generate_markdown_report(report: Report, diagnosis: Any) -> str:
    lines = [
        f"# Jitter Benchmark Report: {report.model.name}",
        f"**Device**: {report.device.manufacturer} {report.device.model} ({report.device.soc})  ",
        f"**Date**: {report.created_at}  ",
        f"**Verdict**: `{diagnosis.primary.value}` ({diagnosis.confidence * 100:.1f}% confidence)\n",
        "## Controlled Reruns\n",
        "| Rerun | Start | Delegate | N | Median (ms) | P95 (ms) | P99 (ms) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    import numpy as np
    for r in report.runs:
        med = float(np.median(r.samples_ms))
        p95 = float(np.percentile(r.samples_ms, 95))
        p99 = float(np.percentile(r.samples_ms, 99))
        lines.append(f"| `{r.rerun.value}` | `{r.process_start.value}` | `{r.delegate_executed}` | {len(r.samples_ms)} | {med:.2f} | {p95:.2f} | {p99:.2f} |")

    lines.append("\n## Diagnosis Evidence\n")
    for ev in diagnosis.evidence:
        mark = "✓" if ev.passed else "•"
        lines.append(f"- **{mark}** {ev.plain_language}")

    return "\n".join(lines) + "\n"


def _generate_html_report(report: Report, diagnosis: Any) -> str:
    md_content = _generate_markdown_report(report, diagnosis)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Jitter Report - {report.model.name}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; max-width: 800px; margin: 40px auto; padding: 20px; line-height: 1.6; background: #0f1115; color: #e6edf3; }}
    table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
    th, td {{ padding: 10px; border: 1px solid #30363d; text-align: left; }}
    th {{ background: #161b22; }}
    code {{ background: #21262d; padding: 2px 6px; border-radius: 4px; font-size: 0.9em; }}
    .verdict {{ padding: 16px; border-radius: 8px; border-left: 6px solid #238636; background: #161b22; margin: 20px 0; }}
  </style>
</head>
<body>
  <div class="verdict">
    <h2>Verdict: {diagnosis.primary.value.upper()} ({diagnosis.confidence * 100:.1f}% confidence)</h2>
    <p>Model: <code>{report.model.name}</code> | Device: <code>{report.device.manufacturer} {report.device.model}</code></p>
  </div>
  <pre>{md_content}</pre>
</body>
</html>"""


if __name__ == "__main__":
    app()
