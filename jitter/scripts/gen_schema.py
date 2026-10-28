"""Generate or verify JSON Schema definitions for Report and Protocol models."""

import argparse
import json
import sys
from pathlib import Path

from jitter.schema.protocol import RunnerInfoResponse
from jitter.schema.report import Report


def check_schemas() -> bool:
    """Return True if committed schemas match current models, False otherwise."""
    return generate_schemas(check_only=True) == 0


def generate_schemas(check_only: bool = False) -> int:
    """Generate or verify JSON Schemas against committed files."""
    repo_root = Path(__file__).resolve().parent.parent
    schema_dir = repo_root / "schema"
    schema_dir.mkdir(parents=True, exist_ok=True)

    report_schema_path = schema_dir / "report.schema.json"
    protocol_schema_path = schema_dir / "protocol.schema.json"

    # Export schemas with sorted keys and 2-space indentation
    report_schema = json.dumps(Report.model_json_schema(), indent=2, sort_keys=True) + "\n"
    protocol_schema = (
        json.dumps(RunnerInfoResponse.model_json_schema(), indent=2, sort_keys=True) + "\n"
    )

    diffs = []

    for path, content, name in [
        (report_schema_path, report_schema, "report.schema.json"),
        (protocol_schema_path, protocol_schema, "protocol.schema.json"),
    ]:
        if check_only:
            if not path.exists():
                diffs.append(f"Missing schema file: {name}")
            else:
                existing = path.read_text(encoding="utf-8")
                if existing != content:
                    diffs.append(
                        f"Schema drift detected in {name}. Run `python scripts/gen_schema.py` to update."
                    )
        else:
            path.write_text(content, encoding="utf-8")
            print(f"Generated {path}")

    if check_only and diffs:
        for d in diffs:
            print(f"ERROR: {d}", file=sys.stderr)
        return 1

    if check_only:
        print("All committed JSON Schemas are up to date.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate or verify Jitter JSON Schemas.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check whether committed schemas match generated schemas without writing.",
    )
    args = parser.parse_args()
    sys.exit(generate_schemas(check_only=args.check))


if __name__ == "__main__":
    main()
