"""Android ADB Runner for executing benchmark reruns on real hardware or wire-protocol servers."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from jitter.runners.base import BaseRunner
from jitter.schema.protocol import (
    RestRequest,
    RestResponse,
    RunnerCapabilities,
    RunnerInfoResponse,
    RunRequest,
    RunStateResponse,
    RunStatus,
    TelemetrySnapshot,
)
from jitter.schema.runs import RunEntry


@dataclass(frozen=True)
class AdbDevice:
    """Attached Android device metadata and readiness summary."""

    serial: str
    state: str
    model: str
    manufacturer: str
    soc: str
    android_api: int
    battery_pct: float
    battery_temp_c: float
    thermal_status: int
    is_ready: bool
    readiness_notes: list[str]


class AdbRunnerError(RuntimeError):
    """Exception raised for ADB runner failures or device safety violations."""


class AdbRunner(BaseRunner):
    """Controls on-device execution via Android Debug Bridge and Wire Protocol v1."""

    def __init__(
        self,
        serial: str | None = None,
        port: int = 8080,
        base_url: str | None = None,
        adb_path: str | None = None,
    ) -> None:
        self.serial = serial
        self.port = port
        self.adb_path = adb_path or shutil.which("adb") or "adb"
        self._base_url = base_url or f"http://127.0.0.1:{port}"
        self._quiet_mode_active = False

    @property
    def base_url(self) -> str:
        return self._base_url

    def _exec_adb(self, args: list[str], timeout: float = 15.0) -> str:
        """Run adb command with target serial and return stdout."""
        cmd = [self.adb_path]
        if self.serial:
            cmd.extend(["-s", self.serial])
        cmd.extend(args)

        try:
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except FileNotFoundError:
            raise AdbRunnerError(
                f"ADB executable '{self.adb_path}' not found. Please install Android Platform Tools."
            ) from None
        except subprocess.TimeoutExpired:
            raise AdbRunnerError(
                f"ADB command timed out after {timeout}s: {' '.join(cmd)}"
            ) from None

        if res.returncode != 0:
            err = res.stderr.strip() or res.stdout.strip()
            raise AdbRunnerError(f"ADB command failed ({res.returncode}): {err}")

        return res.stdout.strip()

    def _http_request(
        self, method: str, path: str, payload: dict[str, Any] | None = None, timeout: float = 30.0
    ) -> dict[str, Any]:
        """Send HTTP wire protocol v1 request."""
        url = f"{self._base_url}{path}"
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {"Content-Type": "application/json"} if payload is not None else {}
        req = urllib.request.Request(url, data=data, headers=headers, method=method)

        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                body = response.read().decode("utf-8")
                return json.loads(body) if body else {}
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            raise AdbRunnerError(f"HTTP {e.code} from runner at {path}: {err_body}") from e
        except urllib.error.URLError as e:
            raise AdbRunnerError(f"Failed to connect to runner at {url}: {e.reason}") from e

    def info(self) -> RunnerInfoResponse:
        """Query hardware capabilities and current telemetry snapshot from runner."""
        try:
            raw = self._http_request("GET", "/v1/info")
            return RunnerInfoResponse.model_validate(raw)
        except Exception:
            # Fallback to direct ADB telemetry queries if wire server is not responding directly
            return self._query_device_info_via_adb()

    def _query_device_info_via_adb(self) -> RunnerInfoResponse:
        """Extract device hardware properties and battery/thermal state using direct ADB commands."""
        model = self._exec_adb(["shell", "getprop", "ro.product.model"]) or "Android Device"
        mfr = self._exec_adb(["shell", "getprop", "ro.product.manufacturer"]) or "Generic"
        soc = self._exec_adb(["shell", "getprop", "ro.board.platform"]) or "Unknown SoC"
        api_str = self._exec_adb(["shell", "getprop", "ro.build.version.sdk"]) or "30"
        android_api = int(api_str) if api_str.isdigit() else 30

        # Parse battery dump
        bat_dump = self._exec_adb(["shell", "dumpsys", "battery"])
        level_m = re.search(r"level:\s*(\d+)", bat_dump)
        temp_m = re.search(r"temperature:\s*(\d+)", bat_dump)
        charging_m = re.search(r"status:\s*(\d+)", bat_dump)

        battery_pct = float(level_m.group(1)) if level_m else 100.0
        # Android battery temperature is reported in tenths of a degree Celsius (e.g. 315 -> 31.5 C)
        battery_temp_c = (float(temp_m.group(1)) / 10.0) if temp_m else 30.0
        charging = charging_m.group(1) in ("2", "5") if charging_m else False

        # Parse thermal status (API 29+)
        thermal_status = 0
        try:
            th_out = self._exec_adb(["shell", "cmd", "thermalservice", "getthermalstatus"])
            th_m = re.search(r"\b([0-6])\b", th_out)
            if th_m:
                thermal_status = int(th_m.group(1))
        except Exception:
            thermal_status = 0

        return RunnerInfoResponse(
            runner_version="0.1.0",
            device_model=model,
            manufacturer=mfr,
            soc=soc,
            android_api=android_api,
            capabilities=RunnerCapabilities(
                supported_delegates=["gpu", "cpu", "nnapi"],
                can_measure_headroom=android_api >= 30,
                can_control_screen=True,
                safety_temp_ceiling_c=45.0,
                min_battery_pct=15.0,
            ),
            current_telemetry=TelemetrySnapshot(
                timestamp_ns=time.monotonic_ns(),
                thermal_status=thermal_status,
                thermal_headroom=None,
                battery_temp_c=battery_temp_c,
                battery_pct=battery_pct,
                charging=charging,
                screen_on=True,
            ),
        )

    def run(self, spec: RunRequest) -> RunEntry:
        """Initiate and await completion of a controlled benchmark run."""
        # 1. Enforce Safety Constraints
        telemetry = self.info().current_telemetry
        if telemetry.battery_pct < spec.constraints.min_battery_pct:
            raise AdbRunnerError(
                f"Battery level {telemetry.battery_pct:.0f}% is below minimum safe threshold "
                f"({spec.constraints.min_battery_pct:.0f}%)."
            )
        if telemetry.battery_temp_c >= spec.constraints.max_battery_temp_c:
            raise AdbRunnerError(
                f"Battery temperature {telemetry.battery_temp_c:.1f}°C exceeds safety ceiling "
                f"({spec.constraints.max_battery_temp_c:.1f}°C)."
            )
        if telemetry.thermal_status >= spec.constraints.abort_on_thermal_status:
            raise AdbRunnerError(
                f"Device is in thermal status {telemetry.thermal_status} "
                f"(cutoff: {spec.constraints.abort_on_thermal_status}). Rest device before running."
            )

        # 2. Submit run job to wire protocol server
        res = self._http_request("POST", "/v1/runs", spec.model_dump())
        run_id = res.get("run_id")
        if not run_id:
            raise AdbRunnerError(f"Runner did not return run_id: {res}")

        # 3. Poll for run completion
        deadline = time.time() + (spec.duration_s or (spec.measured_iters * 2.0) + 60.0)
        while time.time() < deadline:
            state_raw = self._http_request("GET", f"/v1/runs/{run_id}")
            state = RunStateResponse.model_validate(state_raw)
            if state.status == RunStatus.DONE:
                result_raw = self._http_request("GET", f"/v1/runs/{run_id}/result")
                return RunEntry.model_validate(result_raw)
            if state.status in (RunStatus.FAILED, RunStatus.ABORTED):
                raise AdbRunnerError(
                    f"Job {run_id} terminated with status {state.status.value}: {state.error_message or state.abort_reason}"
                )
            time.sleep(0.5)

        self.abort()
        raise AdbRunnerError(f"Job {run_id} timed out waiting for completion.")

    def rest(self, target_temp_c: float, max_wait_s: float = 300.0) -> bool:
        """Wait until device cools down to target_temp_c or timeout expires."""
        try:
            req = RestRequest(target_temp_c=target_temp_c, max_rest_s=max_wait_s)
            raw = self._http_request("POST", "/v1/rest", req.model_dump(), timeout=max_wait_s + 5.0)
            res = RestResponse.model_validate(raw)
            return res.target_met
        except Exception:
            # Fallback to direct polling
            t0 = time.time()
            while time.time() - t0 < max_wait_s:
                info = self.info()
                if info.current_telemetry.battery_temp_c <= target_temp_c:
                    return True
                time.sleep(2.0)
            return False

    def abort(self) -> None:
        """Immediately abort any active job and reset device settings."""
        try:
            self._http_request("POST", "/v1/runs/current/abort", {"reason": "abort_requested"})
        except Exception:
            pass
        if self._quiet_mode_active:
            self.set_quiet_mode(False)

    def set_quiet_mode(self, enabled: bool) -> None:
        """Toggle quiet device mode (DND, airplane mode, stay awake)."""
        val = "1" if enabled else "0"
        try:
            self._exec_adb(["shell", "settings", "put", "global", "zen_mode", val])
            self._exec_adb(["shell", "settings", "put", "global", "airplane_mode_on", val])
            self._exec_adb(
                [
                    "shell",
                    "am",
                    "broadcast",
                    "-a",
                    "android.intent.action.AIRPLANE_MODE",
                    "--ez",
                    "state",
                    "true" if enabled else "false",
                ]
            )
            self._exec_adb(["shell", "svc", "power", "stayon", "true" if enabled else "false"])
            self._quiet_mode_active = enabled
        except Exception:
            # Non-fatal if device permissions or adb doesn't allow setting zen_mode
            pass

    @classmethod
    def list_devices(cls, adb_path: str = "adb") -> list[AdbDevice]:
        """Enumerate attached ADB devices with battery and thermal readiness checks."""
        resolved_adb = shutil.which(adb_path) or adb_path
        try:
            res = subprocess.run(
                [resolved_adb, "devices", "-l"],
                capture_output=True,
                text=True,
                timeout=5.0,
                check=False,
            )
        except Exception:
            return []

        devices: list[AdbDevice] = []
        for line in res.stdout.splitlines():
            line = line.strip()
            if not line or line.startswith("List of devices") or line.startswith("*"):
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            serial = parts[0]
            state = parts[1]

            if state != "device":
                devices.append(
                    AdbDevice(
                        serial=serial,
                        state=state,
                        model="Unknown",
                        manufacturer="Unknown",
                        soc="Unknown",
                        android_api=0,
                        battery_pct=0.0,
                        battery_temp_c=0.0,
                        thermal_status=0,
                        is_ready=False,
                        readiness_notes=[f"Device in state '{state}'"],
                    )
                )
                continue

            # Query device properties
            runner = cls(serial=serial, adb_path=resolved_adb)
            try:
                info = runner.info()
                notes: list[str] = []
                ready = True

                if info.current_telemetry.battery_pct < 20.0:
                    ready = False
                    notes.append(f"Low battery ({info.current_telemetry.battery_pct:.0f}% < 20%)")
                if info.current_telemetry.battery_temp_c >= 40.0:
                    ready = False
                    notes.append(
                        f"Hot battery ({info.current_telemetry.battery_temp_c:.1f}°C >= 40°C)"
                    )
                if info.current_telemetry.thermal_status >= 2:
                    ready = False
                    notes.append(
                        f"Thermal throttling active (status {info.current_telemetry.thermal_status})"
                    )

                if ready:
                    notes.append("Ready for benchmarking")

                devices.append(
                    AdbDevice(
                        serial=serial,
                        state=state,
                        model=info.device_model,
                        manufacturer=info.manufacturer,
                        soc=info.soc or "Unknown",
                        android_api=info.android_api,
                        battery_pct=info.current_telemetry.battery_pct,
                        battery_temp_c=info.current_telemetry.battery_temp_c,
                        thermal_status=info.current_telemetry.thermal_status,
                        is_ready=ready,
                        readiness_notes=notes,
                    )
                )
            except Exception as e:
                devices.append(
                    AdbDevice(
                        serial=serial,
                        state=state,
                        model="Error",
                        manufacturer="Error",
                        soc="Error",
                        android_api=0,
                        battery_pct=0.0,
                        battery_temp_c=0.0,
                        thermal_status=0,
                        is_ready=False,
                        readiness_notes=[f"Query failed: {e}"],
                    )
                )

        return devices
