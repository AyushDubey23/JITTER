"""Validation script running full diagnosis benchmark and writing docs/diagnosis.md."""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to pythonpath
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from jitter.validation.harness import run_validation_suite
from jitter.validation.report_md import write_diagnosis_report


def main() -> None:
    print("Running Jitter Diagnosis Engine validation suite...", flush=True)
    results = run_validation_suite(scenarios_per_label=15, base_seed=42)

    print(f"Total scenarios: {results.total_scenarios}", flush=True)
    print(f"Macro-F1: {results.macro_f1:.4f}")
    print(f"ECE: {results.ece:.4f}")
    print(f"Brier score: {results.brier_score:.4f}")
    print(f"Mean diagnose latency: {results.mean_diagnose_ms:.2f} ms")

    docs_path = Path(__file__).resolve().parent.parent / "docs" / "diagnosis.md"
    write_diagnosis_report(results, docs_path)
    print(f"Diagnosis validation report written to {docs_path}")

    # Verify gates
    if results.macro_f1 < 0.95:
        print(f"FAIL: Macro-F1 {results.macro_f1:.4f} is below 0.95 target gate!", file=sys.stderr)
        sys.exit(1)
    if results.ece > 0.05:
        print(f"FAIL: ECE {results.ece:.4f} is above 0.05 calibration gate!", file=sys.stderr)
        sys.exit(1)

    print("SUCCESS: All diagnosis validation gates passed!")


if __name__ == "__main__":
    main()
