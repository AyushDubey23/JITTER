"""Pydantic models for rerun definitions, telemetry, and run execution records."""

import math
from collections.abc import Sized
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class DiagnosisLabel(str, Enum):
    """Closed set of diagnosis outcomes."""

    WARMUP_DRIFT = "warmup_drift"
    THERMAL_THROTTLING = "thermal_throttling"
    DELEGATE_FALLBACK = "delegate_fallback"
    BACKGROUND_NOISE = "background_noise"
    REGRESSION = "regression"
    INCONCLUSIVE = "inconclusive"


class RerunType(str, Enum):
    """The four controlled rerun conditions."""

    COLD_WARM = "cold_warm"
    COOLDOWN = "cooldown"
    DELEGATE_SWAP = "delegate_swap"
    QUIET_DEVICE = "quiet_device"


class ProcessStart(str, Enum):
    """Process execution mode."""

    COLD = "cold"
    WARM = "warm"


class TelemetrySeries(BaseModel):
    """Synchronized time series of device state during a run."""

    model_config = ConfigDict(extra="forbid")

    thermal_status: list[Annotated[int, Field(ge=0, le=6)]] = Field(
        ...,
        description="PowerManager thermal status (0=NONE, 1=LIGHT, 2=MODERATE, 3=SEVERE, 4=CRITICAL, 5=EMERGENCY, 6=SHUTDOWN)",
    )
    thermal_headroom: list[float | None] = Field(
        ...,
        description="Thermal headroom forecast (floating point or null if unsupported)",
    )
    battery_temp_c: list[float] = Field(
        ...,
        description="Battery temperature in degrees Celsius",
    )
    charging: list[bool] = Field(
        ...,
        description="Whether device is actively charging",
    )
    screen_on: list[bool] = Field(
        ...,
        description="Whether screen is on",
    )

    @field_validator("battery_temp_c")
    @classmethod
    def validate_battery_temp(cls, v: list[float]) -> list[float]:
        for idx, temp in enumerate(v):
            if not math.isfinite(temp):
                raise ValueError(f"battery_temp_c[{idx}] must be finite, got {temp}")
            if temp < -20.0 or temp > 100.0:
                raise ValueError(f"battery_temp_c[{idx}] out of plausible range: {temp}C")
        return v


class RunEntry(BaseModel):
    """A single execution record of a controlled benchmark rerun."""

    model_config = ConfigDict(extra="forbid")

    rerun: RerunType = Field(..., description="The controlled rerun category")
    process_start: ProcessStart = Field(
        ..., description="Whether run was executed in a fresh process"
    )
    delegate_requested: str = Field(..., min_length=1, description="Delegate requested by caller")
    delegate_executed: str = Field(
        ..., min_length=1, description="Delegate actually executed by runner"
    )
    samples_ms: list[float] = Field(
        ..., min_length=1, description="Per-iteration latency in milliseconds"
    )
    timestamps_ns: list[int] = Field(
        ..., min_length=1, description="Monotonic timestamp in nanoseconds for each sample"
    )
    telemetry: TelemetrySeries = Field(..., description="Aligned device telemetry series")
    notes: list[str] = Field(default_factory=list, description="Descriptive notes or warnings")

    @field_validator("samples_ms")
    @classmethod
    def validate_samples(cls, v: list[float]) -> list[float]:
        for idx, s in enumerate(v):
            if not math.isfinite(s) or s <= 0.0:
                raise ValueError(f"samples_ms[{idx}] must be positive and finite, got {s}")
        return v

    @field_validator("timestamps_ns")
    @classmethod
    def validate_timestamps(cls, v: list[int]) -> list[int]:
        for i in range(1, len(v)):
            if v[i] <= v[i - 1]:
                raise ValueError(
                    f"timestamps_ns must be strictly increasing: timestamps_ns[{i}] ({v[i]}) <= timestamps_ns[{i - 1}] ({v[i - 1]})"
                )
        return v

    @model_validator(mode="after")
    def validate_series_lengths(self) -> "RunEntry":
        n = len(self.samples_ms)
        if len(self.timestamps_ns) != n:
            raise ValueError(
                f"Length mismatch: samples_ms has {n} items but timestamps_ns has {len(self.timestamps_ns)}"
            )
        telem = self.telemetry
        series_checks: list[tuple[str, Sized]] = [
            ("thermal_status", telem.thermal_status),
            ("thermal_headroom", telem.thermal_headroom),
            ("battery_temp_c", telem.battery_temp_c),
            ("charging", telem.charging),
            ("screen_on", telem.screen_on),
        ]
        for name, series in series_checks:
            if len(series) != n:
                raise ValueError(
                    f"Telemetry length mismatch: samples_ms has {n} items but telemetry.{name} has {len(series)}"
                )
        return self
