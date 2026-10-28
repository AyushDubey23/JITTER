"""Pydantic models for the Android Runner HTTP/1.1 Wire Protocol (v1)."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from jitter.schema.runs import ProcessStart


class RunStatus(str, Enum):
    """Execution lifecycle state of a runner job."""

    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    ABORTED = "aborted"
    FAILED = "failed"


class TelemetrySnapshot(BaseModel):
    """Instantaneous snapshot of device physical metrics."""

    model_config = ConfigDict(extra="forbid")

    timestamp_ns: int = Field(..., description="Monotonic timestamp in nanoseconds")
    thermal_status: int = Field(..., ge=0, le=6, description="PowerManager thermal status")
    thermal_headroom: float | None = Field(default=None, description="Thermal headroom forecast")
    battery_temp_c: float = Field(..., description="Battery temperature in degrees Celsius")
    battery_pct: float = Field(
        ..., ge=0.0, le=100.0, description="Battery state of charge (percentage)"
    )
    charging: bool = Field(..., description="Whether device is actively charging")
    screen_on: bool = Field(..., description="Whether screen is currently on")


class RunnerCapabilities(BaseModel):
    """Hardware and engine features exposed by the runner."""

    model_config = ConfigDict(extra="forbid")

    supported_delegates: list[str] = Field(
        ..., min_length=1, description="List of supported delegate names"
    )
    can_measure_headroom: bool = Field(
        default=False, description="Whether getThermalHeadroom is available"
    )
    can_control_screen: bool = Field(
        default=False, description="Whether runner can toggle display state"
    )
    safety_temp_ceiling_c: float = Field(default=45.0, description="Hard thermal cutoff threshold")
    min_battery_pct: float = Field(
        default=15.0, description="Minimum battery level required to initiate runs"
    )


class RunnerInfoResponse(BaseModel):
    """Response payload for GET /v1/info."""

    model_config = ConfigDict(extra="forbid")

    runner_version: str = Field(..., min_length=1, description="Runner software version")
    device_model: str = Field(..., min_length=1, description="Device marketing model")
    manufacturer: str = Field(..., min_length=1, description="Device manufacturer")
    soc: str | None = Field(default=None, description="SoC name if identifiable")
    android_api: int = Field(..., ge=21, description="Android SDK level")
    capabilities: RunnerCapabilities
    current_telemetry: TelemetrySnapshot


class RunSafetyConstraints(BaseModel):
    """Safety limits enforced during benchmark runs."""

    model_config = ConfigDict(extra="forbid")

    max_battery_temp_c: float = Field(
        default=45.0, description="Abort run if battery temp reaches this level"
    )
    abort_on_thermal_status: int = Field(
        default=3, ge=0, le=6, description="Abort run if thermal status reaches this (3=SEVERE)"
    )
    min_battery_pct: float = Field(
        default=15.0, ge=0.0, le=100.0, description="Refuse to start if battery is below this"
    )


class RunRequest(BaseModel):
    """Request payload for POST /v1/runs."""

    model_config = ConfigDict(extra="forbid")

    model_path: str = Field(..., min_length=1, description="Path to model file on device storage")
    delegate: str = Field(..., min_length=1, description="Requested hardware execution delegate")
    warmup_iters: int = Field(
        default=10, ge=0, description="Unmeasured iterations prior to measurement"
    )
    measured_iters: int = Field(default=100, ge=1, description="Number of iterations to record")
    duration_s: float | None = Field(
        default=None, gt=0.0, description="Optional target duration for stress runs"
    )
    input_spec: dict[str, Any] = Field(
        default_factory=dict, description="Input tensor descriptions"
    )
    process_start: ProcessStart = Field(
        default=ProcessStart.WARM, description="Cold vs warm process execution"
    )
    constraints: RunSafetyConstraints = Field(default_factory=RunSafetyConstraints)


class RunCreationResponse(BaseModel):
    """Response payload for POST /v1/runs."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(..., min_length=1, description="Unique job token assigned to run")
    status: RunStatus = Field(default=RunStatus.QUEUED)


class RunProgress(BaseModel):
    """Current execution metrics for an active run."""

    model_config = ConfigDict(extra="forbid")

    completed_iters: int = Field(default=0, ge=0)
    total_iters: int = Field(..., ge=1)
    current_median_ms: float | None = None
    current_temp_c: float | None = None


class RunStateResponse(BaseModel):
    """Response payload for GET /v1/runs/{id}."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    status: RunStatus
    progress: RunProgress
    abort_reason: str | None = None
    error_message: str | None = None


class AbortRequest(BaseModel):
    """Optional payload for POST /v1/runs/{id}/abort."""

    model_config = ConfigDict(extra="forbid")

    reason: str = Field(default="user_requested", description="Reason for aborting")


class AbortResponse(BaseModel):
    """Response payload for POST /v1/runs/{id}/abort."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    aborted: bool
    status: RunStatus


class RestRequest(BaseModel):
    """Request payload for POST /v1/rest."""

    model_config = ConfigDict(extra="forbid")

    target_thermal_status: int = Field(
        default=0, ge=0, le=6, description="Wait until thermal_status <= target"
    )
    target_temp_c: float | None = Field(
        default=None, description="Wait until battery_temp_c <= target_temp_c"
    )
    max_rest_s: float = Field(default=300.0, gt=0.0, description="Maximum rest timeout in seconds")
    poll_interval_s: float = Field(
        default=1.0, gt=0.1, description="Interval between telemetry checks"
    )


class RestResponse(BaseModel):
    """Response payload for POST /v1/rest."""

    model_config = ConfigDict(extra="forbid")

    target_met: bool
    elapsed_s: float
    observed_temps_c: list[float]
    observed_thermal_status: list[int]
    final_telemetry: TelemetrySnapshot
