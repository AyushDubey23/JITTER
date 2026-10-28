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


if __name__ == "__main__":
    app()
