# Jitter Android Runner HTTP Wire Protocol (v1)

This specification defines the HTTP/1.1 REST wire protocol between the Jitter host CLI/orchestrator (Python) and the on-device benchmark runner APK (Kotlin/Android).

The protocol communicates over an `adb forward` TCP tunnel or direct local WiFi interface:
```bash
adb forward tcp:8080 tcp:8080
```

Base URL: `http://127.0.0.1:8080`

All request and response bodies use UTF-8 JSON with `Content-Type: application/json`.

---

## 1. Endpoints Overview

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/ping` | Liveness and health check |
| `GET` | `/v1/info` | Hardware capabilities and instantaneous telemetry snapshot |
| `POST` | `/v1/runs` | Submit a new controlled benchmark execution job |
| `GET` | `/v1/runs/{id}` | Poll execution status and progress of an active job |
| `GET` | `/v1/runs/{id}/result` | Retrieve full iteration latency series and telemetry of a completed job |
| `POST` | `/v1/runs/{id}/abort` | Immediately abort an active benchmark job and release resources |
| `POST` | `/v1/rest` | Block/poll until device cools down to target temperature |

---

## 2. Endpoint Specifications

### 2.1 `GET /ping`
Verifies that the runner service is active and responsive.

**Response** (`200 OK`):
```json
{
  "status": "ok",
  "version": "0.1.0"
}
```

---

### 2.2 `GET /v1/info`
Queries device model details, hardware delegate support, and current battery and thermal telemetry.

**Response** (`200 OK`):
```json
{
  "runner_version": "0.1.0",
  "device_model": "Pixel 7",
  "manufacturer": "Google",
  "soc": "Tensor G2",
  "android_api": 34,
  "capabilities": {
    "supported_delegates": ["gpu", "cpu", "nnapi"],
    "can_measure_headroom": true,
    "can_control_screen": true,
    "safety_temp_ceiling_c": 45.0,
    "min_battery_pct": 15.0
  },
  "current_telemetry": {
    "timestamp_ns": 1729864200000000000,
    "thermal_status": 0,
    "thermal_headroom": 0.35,
    "battery_temp_c": 29.4,
    "battery_pct": 82.0,
    "charging": false,
    "screen_on": true
  }
}
```

---

### 2.3 `POST /v1/runs`
Initiates a benchmark run on device.

**Request Body**:
```json
{
  "model_path": "/data/local/tmp/mobilenet_v2.tflite",
  "delegate": "gpu",
  "warmup_iters": 5,
  "measured_iters": 50,
  "process_start": "cold",
  "input_spec": {
    "input": [1, 224, 224, 3]
  },
  "constraints": {
    "max_battery_temp_c": 45.0,
    "abort_on_thermal_status": 3,
    "min_battery_pct": 15.0
  }
}
```

**Response** (`201 Created`):
```json
{
  "run_id": "run_982341",
  "status": "running"
}
```

---

### 2.4 `GET /v1/runs/{id}`
Polls job state while the benchmark is in progress.

**Response** (`200 OK`):
```json
{
  "run_id": "run_982341",
  "status": "running",
  "progress": {
    "completed_iters": 25,
    "total_iters": 50,
    "current_median_ms": 14.2,
    "current_temp_c": 31.8
  },
  "abort_reason": null,
  "error_message": null
}
```

Lifecycle Status Values: `queued`, `running`, `done`, `aborted`, `failed`.

---

### 2.5 `GET /v1/runs/{id}/result`
Retrieves the execution record once `status` is `done`.

**Response** (`200 OK`):
```json
{
  "rerun": "cold_warm",
  "process_start": "cold",
  "delegate_requested": "gpu",
  "delegate_executed": "gpu",
  "samples_ms": [24.1, 15.2, 14.8, 14.6, 14.5],
  "timestamps_ns": [
    1729864201000000000,
    1729864201025000000,
    1729864201041000000,
    1729864201056000000,
    1729864201071000000
  ],
  "telemetry": {
    "thermal_status": [0, 0, 0, 0, 0],
    "thermal_headroom": [0.35, 0.35, 0.36, 0.36, 0.37],
    "battery_temp_c": [29.5, 29.5, 29.6, 29.6, 29.7],
    "charging": [false, false, false, false, false],
    "screen_on": [true, true, true, true, true]
  },
  "notes": []
}
```

---

### 2.6 `POST /v1/rest`
Blocks until device returns to rested thermal conditions.

**Request Body**:
```json
{
  "target_thermal_status": 0,
  "target_temp_c": 32.0,
  "max_rest_s": 180.0,
  "poll_interval_s": 2.0
}
```

**Response** (`200 OK`):
```json
{
  "target_met": true,
  "elapsed_s": 42.5,
  "observed_temps_c": [35.2, 34.1, 33.0, 31.8],
  "observed_thermal_status": [1, 0, 0, 0],
  "final_telemetry": {
    "timestamp_ns": 1729864245000000000,
    "thermal_status": 0,
    "thermal_headroom": 0.20,
    "battery_temp_c": 31.8,
    "battery_pct": 81.0,
    "charging": false,
    "screen_on": true
  }
}
```

---

## 3. Kotlin Android Runner Implementation Guide

1. **Service Architecture**:
   - Run benchmark workload inside an Android `ForegroundService` with a high-priority notification to prevent Android LMK (Low Memory Killer) termination.
   - Use `EmbeddedServer` (e.g. Ktor, Netty, or NanoHTTPD) listening on port 8080.
2. **Measurement Timing**:
   - Measure each iteration latency using `System.nanoTime()`:
     ```kotlin
     val t0 = System.nanoTime()
     interpreter.run(inputTensor, outputTensor)
     val dtMs = (System.nanoTime() - t0) / 1_000_000.0
     ```
3. **Telemetry Polling**:
   - Register an `OnThermalStatusChangedListener` via `PowerManager`:
     ```kotlin
     val powerManager = getSystemService(Context.POWER_SERVICE) as PowerManager
     powerManager.addThermalStatusListener { status -> ... }
     ```
   - On API 30+, query `powerManager.getThermalHeadroom(forecastSeconds)`.
   - Read battery temperature from `Intent(Intent.ACTION_BATTERY_CHANGED)`:
     ```kotlin
     val tempC = intent.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, 0) / 10.0
     ```
4. **Safety Enforcement**:
   - If `tempC >= safety_temp_ceiling_c` (45°C) or `thermal_status >= 3` (SEVERE), immediately halt inference, release GPU delegates, and transition state to `aborted`.
