"""CLI entrypoint for Jitter."""

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from jitter import __version__
from jitter.schema.protocol import RunnerInfoResponse
from jitter.schema.report import Report

app = typer.Typer(
    name="jitter",
    help="Root-cause analysis tool for on-device ML latency.",
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
    typer.echo(f"jitter v{__version__}")


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
        typer.echo(f"Wrote report schema to {output}")
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
        typer.echo(f"Wrote protocol schema to {output}")
    else:
        typer.echo(schema_json)


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
        typer.Option(
            "--secondary", help="Optional secondary fault: warmup, thermal, fallback, noise"
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="File path to write report JSON"),
    ] = None,
) -> None:
    """Generate a synthetic benchmark report with known physical failure characteristics."""
    from jitter.schema.runs import DiagnosisLabel
    from jitter.synth.scenarios import generate_scenario_report

    try:
        diag_label = DiagnosisLabel(failure_mode)
    except ValueError:
        valid_modes = ", ".join(m.value for m in DiagnosisLabel)
        typer.echo(
            f"Error: Invalid failure mode '{failure_mode}'. Choose from: {valid_modes}", err=True
        )
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
        typer.echo(f"Simulated {diag_label.value} report saved to {output}")
    else:
        typer.echo(json_output)


if __name__ == "__main__":
    app()
