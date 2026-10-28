# Project Status: Jitter

Last Updated: Phase 0 (Scaffold)

## Phase Checklist

- [x] **Phase 0: Scaffold**: Repo layout, pyproject.toml with Hatchling, strict mypy/ruff, CI skeleton, PyPI name verification, STATUS.md.
- [ ] **Phase 1: Schema and Report I/O**: Pydantic models for Report, JSON Schema export, round-trip validation, `jitter schema` CLI command.
- [ ] **Phase 2: Statistics Core**: Summary, autocorrelation, warmup detection, trend drift, bootstrap CI, pairwise comparisons, outlier detection.
- [ ] **Phase 3: Synthetic Generators & `jitter simulate`**: Physics-based thermal RC model, delegate latency & fallback profiles, Poisson-Pareto noise bursts, multi-fault scenarios.
- [ ] **Phase 4: Runners & The Four Reruns**: SyntheticRunner, DesktopOnnxRunner, 4 controlled reruns (`cold_warm`, `cooldown`, `delegate_swap`, `quiet_device`), orchestrator pipeline.
- [ ] **Phase 5: Diagnosis Engine & Validation Harness**: Feature extraction, rule evaluation with evidence strings, decision engine, train/val/test split validation, confusion matrix, calibration.
- [ ] **Phase 6: CLI, Wire Protocol & Documentation**: Full CLI suite (`run`, `compare`, `diagnose`, `devices`, `report`, `simulate`, `schema`, `validate`), `AdbRunner`, `FakeRunnerServer`, documentation.
- [ ] **Phase 7: Packaging & CI**: Build sdist and wheel, twine verification, multi-OS GitHub Actions workflow, nightly validation regressions check.

## Package Identity & PyPI Availability

- PyPI search result for `jitter`: **Taken** (HTTP 200).
- Selected distribution name: **`jitter-ml`** (Available on PyPI, HTTP 404).
- Alternative candidate names: `jitterbench` (HTTP 404), `jitter-latency` (HTTP 404).
- Import name: **`jitter`** across all modules.

## Known Gaps & Current Limitations

- Project is in early scaffolding phase. No real phone telemetry is integrated yet.
- Real-device validation is deferred to `validation/real_devices/` upon hardware runner completion.
