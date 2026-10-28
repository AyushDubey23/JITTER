"""Generate realistic example reports for all 6 failure modes."""

from __future__ import annotations

import argparse
from pathlib import Path

from jitter.schema.runs import DiagnosisLabel
from jitter.synth.scenarios import generate_scenario_report

EXAMPLE_MODES = [
    DiagnosisLabel.WARMUP_DRIFT,
    DiagnosisLabel.THERMAL_THROTTLING,
    DiagnosisLabel.DELEGATE_FALLBACK,
    DiagnosisLabel.BACKGROUND_NOISE,
    DiagnosisLabel.REGRESSION,
    DiagnosisLabel.INCONCLUSIVE,
]


def generate_examples(output_dir: Path | None = None) -> list[Path]:
    """Generate canonical example reports into target directory."""
    if output_dir is None:
        repo_root = Path(__file__).resolve().parent.parent
        output_dir = repo_root / "examples" / "reports"

    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files: list[Path] = []

    for idx, mode in enumerate(EXAMPLE_MODES):
        report = generate_scenario_report(
            label=mode,
            model_name="mobilenet_v2",
            iterations=60,
            seed=42 + idx * 10,
        )
        file_path = output_dir / f"{mode.value}.json"
        report.to_file(file_path, indent=2)
        print(f"Generated example report: {file_path}")
        generated_files.append(file_path)

    return generated_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Jitter example benchmark reports.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write example reports to.",
    )
    args = parser.parse_args()
    generate_examples(args.output_dir)


if __name__ == "__main__":
    main()
