"""Regression tests for issues #2, #3, and #4.

- Issue #2: README tells people to install the wrong package ('pip install jitter').
  Must be 'pip install jitter-ml' everywhere; 'pip install jitter' without '-ml' must not exist.
- Issue #3: PyPI description contradicts the release ('Nothing released yet', 'design phase', 'MIT planned').
  Must state Alpha release and MIT license consistently without contradictory draft text.
- Issue #4: Diagnosis evidence contains contradictory statements and nonsense negative percentages.
  Evidence must strictly curate supporting and ruled-out evidence with no self-contradictory lines.
"""

from __future__ import annotations

import re
from pathlib import Path

from jitter.diagnosis.engine import diagnose
from jitter.diagnosis.features import extract_features
from jitter.diagnosis.rules import evaluate_background_noise, evaluate_delegate_fallback
from jitter.schema.runs import DiagnosisLabel
from jitter.synth.scenarios import generate_scenario_report


def test_regression_issue_2_package_install_name() -> None:
    """Issue #2: Ensure 'pip install jitter-ml' is used and 'pip install jitter' (un-suffixed) is banned."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    jitter_root = Path(__file__).resolve().parent.parent

    target_files = [
        repo_root / "README.md",
        jitter_root / "README.md",
        jitter_root / "CONTRIBUTING.md",
    ]

    for f in target_files:
        assert f.exists(), f"Expected file {f} to exist"
        content = f.read_text(encoding="utf-8")

        # Must contain 'pip install jitter-ml' (or optional extras like 'pip install jitter-ml[dev]')
        assert "pip install jitter-ml" in content or "pip install -e" in content, (
            f"{f.name} missing 'pip install jitter-ml'"
        )

        # Must NOT contain 'pip install jitter' unless followed immediately by '-ml'
        # Regex matches 'pip install jitter' NOT followed by '-ml' or '-latency' or 'bench'
        banned_pattern = re.compile(r"pip install ([\"'])?jitter(?!-ml)")
        matches = banned_pattern.findall(content)
        assert not matches, f"Found banned unsuffixed 'pip install jitter' in {f.name}: {matches}"


def test_regression_issue_3_pypi_description_no_contradictory_draft_text() -> None:
    """Issue #3: Ensure README doesn't state 'design phase', 'Nothing released yet', or 'MIT (planned)'."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    jitter_root = Path(__file__).resolve().parent.parent

    target_files = [
        repo_root / "README.md",
        jitter_root / "README.md",
    ]

    banned_phrases = [
        "Nothing in this repository is released yet",
        "Status: design phase",
        "MIT (planned)",
    ]

    for f in target_files:
        content = f.read_text(encoding="utf-8")
        for phrase in banned_phrases:
            assert phrase not in content, (
                f"Contradictory draft text '{phrase}' still present in {f.name}"
            )

        # Must explicitly mention Alpha release and MIT license
        assert "Alpha release" in content
        assert "MIT" in content


def test_regression_issue_4_non_contradictory_thermal_evidence() -> None:
    """Issue #4: Verify thermal throttling evidence contains no contradictory statements or negative percentages."""
    report = generate_scenario_report("thermal_throttling", iterations=50, seed=42)
    diag = diagnose(report)

    assert diag.primary == DiagnosisLabel.THERMAL_THROTTLING
    assert diag.confidence >= 0.85

    evidence_texts = [e.plain_language for e in diag.evidence]

    # 1. No negative percentage in tail latency suppression
    for text in evidence_texts:
        assert "suppressed tail latency by -" not in text, (
            f"Found nonsense negative percentage suppression in evidence: {text}"
        )

    # 2. Because requested delegate ('gpu') executed on 'gpu', it must NOT say fallback to CPU
    for text in evidence_texts:
        assert "indicating delegate fallback to CPU" not in text, (
            f"Found false CPU fallback claim in evidence: {text}"
        )

    # 3. Ruled-out items must be cleanly identified as ruled out
    ruled_out = [text for text in evidence_texts if text.startswith("Ruled out")]
    assert len(ruled_out) >= 2, "Expected ruled out hypotheses in curated evidence"


def test_regression_issue_4_no_negative_suppression_when_quiet_worse() -> None:
    """Issue #4: When quiet mode tail latency is worse than cold (p99 increased), ratio is negative and handled cleanly."""
    report = generate_scenario_report("thermal_throttling", iterations=40, seed=42)
    feat = extract_features(report)

    # Test direct evaluate_background_noise when quiet mode does not suppress tail
    _, _, noise_ev = evaluate_background_noise(feat)
    for ev in noise_ev:
        if ev.rule_id == "quiet_mode_p99_suppression":
            # If reduction ratio was <= 0, wording must NOT claim it was 'suppressed by -X%'
            if feat.quiet_p99_reduction_ratio <= 0:
                assert "suppressed tail latency by -" not in ev.plain_language
                assert "did not suppress tail latency" in ev.plain_language


def test_regression_issue_4_delegate_speedup_vs_fallback_wording() -> None:
    """Issue #4: Parity check must distinguish true CPU parity from distinct hardware acceleration."""
    report = generate_scenario_report("warmup_drift", iterations=40, seed=42)
    feat = extract_features(report)

    _, _, delegate_ev = evaluate_delegate_fallback(feat)
    for ev in delegate_ev:
        if ev.rule_id == "delegate_vs_cpu_parity":
            if not ev.passed:
                # If cpu parity was not passed, it means delegate operates distinctly from CPU
                assert "indicating performance parity with CPU fallback" not in ev.plain_language
                assert "distinctly faster" in ev.plain_language or "distinctly" in ev.plain_language
