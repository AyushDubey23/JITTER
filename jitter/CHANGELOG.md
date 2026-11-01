# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2] - 2026-11-01

### Added
- Regression test suite `tests/test_regressions.py` enforcing checks for issue #2 (package install name), issue #3 (PyPI metadata and status consistency), and issue #4 (non-contradictory diagnosis evidence generation).

### Fixed
- Corrected marketing site reverse-tunnel port documentation to 8080 (matching `AdbRunner` and wire protocol specification).
- Synchronized site hero version badge and CLI help version outputs to v0.1.2.

## [0.1.1] - 2026-11-01

### Fixed
- Relabeled moving-block bootstrap coverage row in `docs/validation.md` as known gap due to finite-sample undercoverage.
- Fixed package install name in README and documentation from `jitter` to `jitter-ml`.
- Updated PyPI description and README to reflect Alpha release status (v0.1.1) and MIT license.
- Rewrote diagnosis evidence generator to eliminate contradictory evidence statements and nonsense percentages.
- Excluded `.hypothesis/`, `.github/`, and `STATUS.md` from sdist packaging.
- Normalized markdown documentation line endings.

## [0.1.0] - 2026-11-01

### Added
- Initial project scaffolding with Hatchling build system.
- Package layout (`src/jitter`) with PEP 561 `py.typed`.
- Complete CLI suite with Typer (`run`, `diagnose`, `compare`, `simulate`, `report`, `devices`, `schema`).
- Four controlled reruns orchestration pipeline and statistical test core.
- Report schema v0.1.0 with JSON Schema export.
- CI workflows and strict mypy/ruff tooling configurations.
