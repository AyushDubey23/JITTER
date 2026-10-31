"""Markdown generation for Jitter diagnosis validation results."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from jitter.validation.harness import ValidationResults


def generate_diagnosis_markdown(results: ValidationResults) -> str:
    """Render validation results into GitHub-flavored Markdown."""
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines: list[str] = [
        "# Diagnosis Engine Validation Report",
        "",
        f"*Generated on {now_utc} across {results.total_scenarios} controlled scenarios.*",
        "",
        "## Summary Metrics",
        "",
        f"- **Macro-Averaged F1 Score**: `{results.macro_f1:.4f}` (Target: $\\ge 0.95$)",
        f"- **Expected Calibration Error (ECE)**: `{results.ece:.4f}` (Target: $\\le 0.05$)",
        f"- **Brier Score**: `{results.brier_score:.4f}`",
        f"- **Mean Diagnosis Latency**: `{results.mean_diagnose_ms:.2f} ms`",
        f"- **Total Scenarios Evaluated**: `{results.total_scenarios}` ({results.scenarios_per_label} per class)",
        "",
        "## Confusion Matrix",
        "",
        "Rows represent Ground Truth labels; Columns represent Predicted labels.",
        "",
    ]

    header = ["True \\ Pred"] + results.labels
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")

    for i, true_label in enumerate(results.labels):
        row = [f"**{true_label}**"] + [str(v) for v in results.confusion_matrix[i]]
        lines.append("| " + " | ".join(row) + " |")

    lines.extend([
        "",
        "## Per-Class Precision, Recall, and F1",
        "",
        "| Failure Mode | Precision | Recall | F1 Score | Support |",
        "| --- | --- | --- | --- | --- |",
    ])

    for label in results.labels:
        m = results.per_class_metrics[label]
        lines.append(
            f"| `{label}` | {m['precision']:.4f} | {m['recall']:.4f} | **{m['f1']:.4f}** | {int(m['support'])} |"
        )

    lines.extend([
        "",
        "## Calibrated Decision Boundaries",
        "",
        "| Rule Category | Diagnostic Signal | Threshold | Target Metric |",
        "| --- | --- | --- | --- |",
        "| **Sample Sufficiency** | Geyer IPS $n_{\\text{eff}}$ | $\\ge 30.0$ | Fallback to `inconclusive` if noise overwhelms signal |",
        "| **Warmup Drift** | Cold Initial vs Steady-State Ratio | $\\ge 1.50$ | CUSUM change-point with $\\ge 1$ discard iteration |",
        "| **Thermal Throttling** | Theil-Sen Median Slope | $\\ge 0.02$ ms/iter | Coinciding with $\\ge$ LIGHT thermal state or cooldown recovery |",
        "| **Delegate Fallback** | Requested vs Executed Match | Discrepancy or CPU Parity | $|\\text{Cliff's } \\delta| \\le 0.147$ against CPU baseline |",
        "| **Background Contention** | Modified Z-score Spikes | $\\ge 3$ bursts | Quiet device mode reduces tail $p_{99}$ by $\\ge 20\\%$ |",
        "| **Model Regression** | Hodges-Lehmann Shift | $\\ge 1.5$ ms shift | Block permutation $p < 0.01$ and Cliff's $\\delta \\ge 0.33$ |",
        "",
        "---",
        "*Report verified via `jitter.validation.harness`.*",
    ])

    return "\n".join(lines) + "\n"


def write_diagnosis_report(results: ValidationResults, output_path: str | Path) -> None:
    """Generate and write diagnosis validation report to file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = generate_diagnosis_markdown(results)
    path.write_text(content, encoding="utf-8")
