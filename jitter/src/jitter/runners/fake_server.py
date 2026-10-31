"""FakeRunnerServer implementing wire protocol v1 for local integration testing."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from jitter.schema.protocol import (
    AbortResponse,
    RestResponse,
    RunCreationResponse,
    RunnerCapabilities,
    RunnerInfoResponse,
    RunProgress,
    RunRequest,
    RunStateResponse,
    RunStatus,
    TelemetrySnapshot,
)
from jitter.schema.runs import RunEntry, TelemetrySeries
from jitter.synth.scenarios import generate_scenario_report


class _FakeRunnerHTTPHandler(BaseHTTPRequestHandler):
    server: FakeRunnerServerInstance  # type: ignore[assignment]

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress default HTTP server logging to keep test output clean."""
        return

    def _send_json(self, status_code: int, data: dict[str, Any] | BaseModel) -> None:  # type: ignore[name-defined]
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        body = data.model_dump_json() if hasattr(data, "model_dump_json") else json.dumps(data)
        self.wfile.write(body.encode("utf-8"))

    def do_GET(self) -> None:
        path = self.path.split("?")[0]
        if path == "/ping":
            self._send_json(200, {"status": "ok", "version": "0.1.0"})
        elif path == "/v1/info":
            info = RunnerInfoResponse(
                runner_version="0.1.0",
                device_model="Pixel 7 (Mock)",
                manufacturer="Google",
                soc="Tensor G2",
                android_api=34,
                capabilities=RunnerCapabilities(
                    supported_delegates=["gpu", "cpu", "nnapi"],
                    can_measure_headroom=True,
                    can_control_screen=True,
                    safety_temp_ceiling_c=45.0,
                    min_battery_pct=15.0,
                ),
                current_telemetry=TelemetrySnapshot(
                    timestamp_ns=time.monotonic_ns(),
                    thermal_status=0,
                    thermal_headroom=0.25,
                    battery_temp_c=29.5,
                    battery_pct=85.0,
                    charging=False,
                    screen_on=True,
                ),
            )
            self._send_json(200, info)
        elif path.startswith("/v1/runs/") and path.endswith("/result"):
            run_id = path.split("/")[3]
            entry = self.server.runs.get(run_id)
            if entry is None:
                self._send_json(404, {"error": "Run not found"})
            else:
                self._send_json(200, entry)
        elif path.startswith("/v1/runs/"):
            run_id = path.split("/")[3]
            if run_id in self.server.runs:
                state = RunStateResponse(
                    run_id=run_id,
                    status=RunStatus.DONE,
                    progress=RunProgress(completed_iters=50, total_iters=50, current_median_ms=12.4),
                )
                self._send_json(200, state)
            else:
                self._send_json(404, {"error": "Run not found"})
        else:
            self._send_json(404, {"error": "Not found"})

    def do_POST(self) -> None:
        path = self.path.split("?")[0]
        content_len = int(self.headers.get("Content-Length", 0))
        post_body = self.rfile.read(content_len).decode("utf-8") if content_len > 0 else "{}"
        try:
            body_json = json.loads(post_body)
        except Exception:
            body_json = {}

        if path == "/v1/runs":
            spec = RunRequest.model_validate(body_json)
            run_id = f"run_{len(self.server.runs) + 1}"

            # Generate synthetic run entry conforming to spec
            scenario_rep = generate_scenario_report("warmup_drift", iterations=spec.measured_iters, seed=42)
            fresh_run = scenario_rep.runs[0]
            n_iters = spec.measured_iters
            t0 = time.monotonic_ns()
            ts = [t0 + i * 20_000_000 for i in range(n_iters)]
            simulated_entry = RunEntry(
                rerun=fresh_run.rerun,
                process_start=spec.process_start,
                delegate_requested=spec.delegate,
                delegate_executed=spec.delegate,
                samples_ms=fresh_run.samples_ms[:n_iters],
                timestamps_ns=ts,
                telemetry=TelemetrySeries(
                    thermal_status=[0] * n_iters,
                    thermal_headroom=[0.2] * n_iters,
                    battery_temp_c=[30.0] * n_iters,
                    charging=[False] * n_iters,
                    screen_on=[True] * n_iters,
                ),
                notes=[],
            )
            self.server.runs[run_id] = simulated_entry
            self._send_json(201, RunCreationResponse(run_id=run_id, status=RunStatus.DONE))
        elif path == "/v1/rest":
            res = RestResponse(
                target_met=True,
                elapsed_s=2.5,
                observed_temps_c=[31.0, 30.5, 30.0],
                observed_thermal_status=[0, 0, 0],
                final_telemetry=TelemetrySnapshot(
                    timestamp_ns=time.monotonic_ns(),
                    thermal_status=0,
                    thermal_headroom=0.2,
                    battery_temp_c=30.0,
                    battery_pct=84.0,
                    charging=False,
                    screen_on=True,
                ),
            )
            self._send_json(200, res)
        elif path.endswith("/abort"):
            run_id = path.split("/")[3]
            self._send_json(200, AbortResponse(run_id=run_id, aborted=True, status=RunStatus.ABORTED))
        else:
            self._send_json(404, {"error": "Not found"})


class FakeRunnerServerInstance(HTTPServer):
    """Internal server holding state for runs."""

    def __init__(self, server_address: tuple[str, int]) -> None:
        super().__init__(server_address, _FakeRunnerHTTPHandler)
        self.runs: dict[str, RunEntry] = {}


class FakeRunnerServer:
    """Context manager and controller for the mock Wire Protocol HTTP server."""

    def __init__(self, host: str = "127.0.0.1", port: int = 0) -> None:
        self.host = host
        self.requested_port = port
        self.server: FakeRunnerServerInstance | None = None
        self._thread: threading.Thread | None = None
        self.port: int = 0

    def start(self) -> str:
        self.server = FakeRunnerServerInstance((self.host, self.requested_port))
        self.port = self.server.server_port
        self._thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self._thread.start()
        return f"http://{self.host}:{self.port}"

    def stop(self) -> None:
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            self.server = None
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def __enter__(self) -> FakeRunnerServer:
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()
