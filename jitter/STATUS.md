# Project Status: Jitter

Last Updated: Phase 7 (Packaging, Multi-OS CI & Release Polish)

## Phase Checklist

- [x] **Phase 0: Scaffold**: Repo layout, pyproject.toml with Hatchling, strict mypy/ruff, CI skeleton, PyPI name verification, STATUS.md.
- [x] **Phase 1: Schema and Report I/O**: Pydantic models for Report, JSON Schema export, round-trip validation, wire protocol v1 schemas, `jitter schema` CLI command.
- [x] **Phase 2: Statistics Core**: Summary, autocorrelation, warmup detection, trend drift, bootstrap CI, pairwise comparisons, outlier detection, Monte Carlo validation harness (`docs/validation.md`).
- [x] **Phase 3: Synthetic Generators & `jitter simulate`**: Physics-based thermal RC model, delegate latency & fallback profiles, Poisson-Pareto noise bursts, multi-fault scenarios, `examples/reports/*.json`, `jitter simulate` CLI.
- [x] **Phase 4: Runners & The Four Reruns**: BaseRunner, SyntheticRunner, DesktopOnnxRunner, 4 controlled reruns (`cold_warm`, `cooldown`, `delegate_swap`, `quiet_device`), orchestrator pipeline, safety constraints.
- [x] **Phase 5: Diagnosis Engine & Validation Harness**: Feature extraction, rule evaluation with evidence strings, decision engine, train/val/test split validation, confusion matrix (Macro-F1 1.0000, ECE 0.0398), calibrated confidence.
- [x] **Phase 6: CLI, Wire Protocol & Documentation**: Full CLI suite (`run`, `compare`, `diagnose`, `devices`, `report`, `simulate`, `schema`, `validate`), `AdbRunner`, `FakeRunnerServer`, documentation (`methodology.md`, `protocol.md`, `report-schema.md`).
- [x] **Phase 7: Packaging & CI**: Built clean sdist and wheel (`jitter_ml-0.1.0`), twine validation passed, clean virtual environment installation verified, multi-OS & multi-Python GitHub Actions matrix CI (Python 3.10-3.13 on Ubuntu, macOS, Windows).

## Package Identity & PyPI Availability

- PyPI search result for `jitter`: **Taken** (HTTP 200).
- Selected distribution name: **`jitter-ml`** (Available on PyPI, HTTP 404).
- Alternative candidate names: `jitterbench` (HTTP 404), `jitter-latency` (HTTP 404).
- Import name: **`jitter`** across all modules.

## Known Gaps & Current Limitations

- Project is in early scaffolding phase. No real phone telemetry is integrated yet.
- Real-device validation is deferred to `validation/real_devices/` upon hardware runner completion.
