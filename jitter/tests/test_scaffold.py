"""Basic scaffold and packaging tests."""

from typer.testing import CliRunner

import jitter
from jitter.cli import app

runner = CliRunner()


def test_version_string() -> None:
    """Package exposes a valid semver version string."""
    assert jitter.__version__ == "0.1.0"


def test_cli_version() -> None:
    """CLI version command runs and outputs package version."""
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "jitter v0.1.0" in result.output
