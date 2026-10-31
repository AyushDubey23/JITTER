# Diagnosis Engine Validation Report

*Generated on 2026-10-31 19:19:55 UTC across 90 controlled scenarios.*

## Summary Metrics

- **Macro-Averaged F1 Score**: `1.0000` (Target: $\ge 0.95$)
- **Expected Calibration Error (ECE)**: `0.0398` (Target: $\le 0.05$)
- **Brier Score**: `0.0020`
- **Mean Diagnosis Latency**: `84.19 ms`
- **Total Scenarios Evaluated**: `90` (15 per class)

## Confusion Matrix

Rows represent Ground Truth labels; Columns represent Predicted labels.

| True \ Pred | warmup_drift | thermal_throttling | delegate_fallback | background_noise | regression | inconclusive |
| --- | --- | --- | --- | --- | --- | --- |
| **warmup_drift** | 15 | 0 | 0 | 0 | 0 | 0 |
| **thermal_throttling** | 0 | 15 | 0 | 0 | 0 | 0 |
| **delegate_fallback** | 0 | 0 | 15 | 0 | 0 | 0 |
| **background_noise** | 0 | 0 | 0 | 15 | 0 | 0 |
| **regression** | 0 | 0 | 0 | 0 | 15 | 0 |
| **inconclusive** | 0 | 0 | 0 | 0 | 0 | 15 |

## Per-Class Precision, Recall, and F1

| Failure Mode | Precision | Recall | F1 Score | Support |
| --- | --- | --- | --- | --- |
| `warmup_drift` | 1.0000 | 1.0000 | **1.0000** | 15 |
| `thermal_throttling` | 1.0000 | 1.0000 | **1.0000** | 15 |
| `delegate_fallback` | 1.0000 | 1.0000 | **1.0000** | 15 |
| `background_noise` | 1.0000 | 1.0000 | **1.0000** | 15 |
| `regression` | 1.0000 | 1.0000 | **1.0000** | 15 |
| `inconclusive` | 1.0000 | 1.0000 | **1.0000** | 15 |

## Calibrated Decision Boundaries

| Rule Category | Diagnostic Signal | Threshold | Target Metric |
| --- | --- | --- | --- |
| **Sample Sufficiency** | Geyer IPS $n_{\text{eff}}$ | $\ge 30.0$ | Fallback to `inconclusive` if noise overwhelms signal |
| **Warmup Drift** | Cold Initial vs Steady-State Ratio | $\ge 1.50$ | CUSUM change-point with $\ge 1$ discard iteration |
| **Thermal Throttling** | Theil-Sen Median Slope | $\ge 0.02$ ms/iter | Coinciding with $\ge$ LIGHT thermal state or cooldown recovery |
| **Delegate Fallback** | Requested vs Executed Match | Discrepancy or CPU Parity | $|\text{Cliff's } \delta| \le 0.147$ against CPU baseline |
| **Background Contention** | Modified Z-score Spikes | $\ge 3$ bursts | Quiet device mode reduces tail $p_{99}$ by $\ge 20\%$ |
| **Model Regression** | Hodges-Lehmann Shift | $\ge 1.5$ ms shift | Block permutation $p < 0.01$ and Cliff's $\delta \ge 0.33$ |

---
*Report verified via `jitter.validation.harness`.*
