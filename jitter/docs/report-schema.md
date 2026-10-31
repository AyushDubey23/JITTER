# Jitter Report Schema Specification

Jitter outputs comprehensive benchmark results conforming to a versioned JSON Schema (`v0.1.0`). The schema defines complete operational metadata, per-iteration latency samples, synchronized physical telemetry series, statistical estimators, pairwise comparisons, and root-cause diagnoses.

To export the raw JSON schema:
```bash
jitter schema report -o report.schema.json
```

---

## 1. Top-Level Structure

```json
{
  "schema_version": "0.1.0",
  "jitter_version": "0.1.0",
  "created_at": "2026-10-31T19:19:55.123456+00:00",
  "seed": 42,
  "source": "android_adb",
  "model": { ... },
  "device": { ... },
  "config": { ... },
  "runs": [ ... ],
  "statistics": { ... },
  "comparisons": [ ... ],
  "diagnosis": { ... },
  "truth": null
}
```

### Top-Level Fields

| Field | Type | Description |
| :--- | :--- | :--- |
| `schema_version` | `string` | Version of the schema (`0.1.0`) |
| `jitter_version` | `string` | Version of the Jitter library that produced the report |
| `created_at` | `string` (ISO 8601) | UTC timestamp of benchmark completion |
| `seed` | `integer` (optional) | Random seed used for reproducibility |
| `source` | `string` | Execution environment: `android_adb`, `desktop_onnx`, `synthetic` |
| `model` | `object` | Target model metadata |
| `device` | `object` | Physical execution hardware information |
| `config` | `object` | Experimental rerun configuration |
| `runs` | `array` | The five individual benchmark execution series |
| `statistics` | `object` | Robust summary statistics for each run |
| `comparisons` | `array` | Statistical hypothesis tests between run conditions |
| `diagnosis` | `object` | Classified root cause, confidence, and plain-language evidence |
| `truth` | `object` (optional) | Ground truth injection parameters for synthetic validation |

---

## 2. Component Specifications

### 2.1 Model Metadata (`model`)
```json
{
  "name": "mobilenet_v2.tflite",
  "sha256": "8f4b23c9a1e0b5d8471c26359048a623910c7320b9e8354c7d2105a2f64b8891",
  "size_bytes": 14200000
}
```

### 2.2 Device Information (`device`)
```json
{
  "model": "Pixel 7",
  "manufacturer": "Google",
  "soc": "Tensor G2",
  "android_api": 34,
  "runner_version": "0.1.0"
}
```

### 2.3 Benchmark Configuration (`config`)
```json
{
  "warmup_iterations": 5,
  "measured_iterations": 50,
  "requested_delegate": "gpu",
  "input_spec": {
    "input": [1, 224, 224, 3]
  },
  "rest_policy": {
    "cooldown_target_c": 32.0,
    "max_rest_s": 180.0
  }
}
```

### 2.4 Run Entry (`runs[]`)
Each entry represents one condition of the four controlled reruns (`fresh`, `warm`, `cooldown`, `cpu`, `quiet`):

```json
{
  "rerun": "cold_warm",
  "process_start": "cold",
  "delegate_requested": "gpu",
  "delegate_executed": "gpu",
  "samples_ms": [28.4, 21.2, 18.6, 17.5, 17.4],
  "timestamps_ns": [1000000000, 1028400000, 1049600000, 1068200000, 1085700000],
  "telemetry": {
    "thermal_status": [0, 0, 0, 0, 0],
    "thermal_headroom": [0.42, 0.42, 0.41, 0.41, 0.40],
    "battery_temp_c": [29.8, 29.8, 29.9, 29.9, 30.0],
    "charging": [false, false, false, false, false],
    "screen_on": [true, true, true, true, true]
  },
  "notes": []
}
```

### 2.5 Pairwise Comparisons (`comparisons[]`)
Statistical comparisons calculated by `jitter.stats.compare`:

```json
{
  "run_a": "fresh",
  "run_b": "warm",
  "diff_estimate": -1.24,
  "p_value": 0.0021,
  "ci_lower": -1.82,
  "ci_upper": -0.68,
  "effect_size": -0.58,
  "effect_magnitude": "large",
  "method": "block_permutation_test"
}
```

### 2.6 Root-Cause Diagnosis (`diagnosis`)
```json
{
  "primary": "warmup_drift",
  "secondary": [],
  "confidence": 0.965,
  "inconclusive_reason": null,
  "evidence": [
    {
      "rule_id": "warmup_ratio_threshold",
      "observed_value": 1.62,
      "threshold": 1.50,
      "passed": true,
      "plain_language": "Cold process initial iterations were 1.62x slower than steady state (threshold: 1.50x)."
    },
    {
      "rule_id": "warmup_discard_iterations",
      "observed_value": 3,
      "threshold": 1,
      "passed": true,
      "plain_language": "CUSUM change-point detector identified 3 initial iterations before latency reached steady-state stability."
    }
  ]
}
```
