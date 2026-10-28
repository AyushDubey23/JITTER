"""CLI entrypoint for Jitter."""

import typer
from rich.console import Console

from jitter import __version__

app = typer.Typer(
    name="jitter",
    help="Root-cause analysis tool for on-device ML latency.",
    no_args_is_help=True,
)
console = Console()


@app.callback()
def main() -> None:
    """Root-cause analysis tool for on-device ML latency."""


@app.command()
def version() -> None:
    """Print Jitter version."""
    typer.echo(f"jitter v{__version__}")


if __name__ == "__main__":
    app()
