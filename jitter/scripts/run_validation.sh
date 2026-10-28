#!/usr/bin/env bash
set -euo pipefail

# Run the 10,000-trial statistical validation harness
python scripts/run_validation.py --trials 10000 --seed 42 --output docs/validation.md
