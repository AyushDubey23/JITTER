"""Pure-logic synthetic runner simulating mobile hardware dynamics."""

from __future__ import annotations

import time

import numpy as np

from jitter import __version__
from jitter.runners.base import BaseRunner
from jitter.schema.protocol import (
    RunnerCapabilities,
    RunnerInfoResponse,
    RunRequest,
    TelemetrySnapshot,
)
from jitter.schema.runs import DiagnosisLabel, ProcessStart, RerunType, RunEntry, TelemetrySeries
from jitter.synth.delegates import FallbackMode, simulate_iteration
from jitter.synth.noise import ContentionNoiseGenerator
from jitter.synth.thermal import RCThermalModel


class SyntheticRunner(BaseRunner):
    """Executes controlled runs against synthetic physical simulation models."""

    def __init__(
        self,
        fault_mode: DiagnosisLabel = DiagnosisLabel.WARMUP_DRIFT,
        model_name: str = "mobilenet_v2",
        seed: int = 42,
        quiet_mode: bool = False,
        secondary_fault: str | None = None,
    ) -> None:
        self.fault_mode = fault_mode
        self.model_name = model_name
        self.seed = seed
        self.quiet_mode = quiet_mode
        self.secondary_fault = secondary_fault
        self.rng = np.random.default_rng(seed)

        self.thermal = RCThermalModel(initial_temp_c=28.5)
        self.noise_gen = ContentionNoiseGenerator(seed=seed)
        self.is_aborted = False
        self.battery_pct = 92.0

    def info(self) -> RunnerInfoResponse:
        """Query synthetic device capabilities and current physical status."""
        curr_temp = self.thermal.current_temp_c
        return RunnerInfoResponse(
            runner_version=__version__,
            device_model="Synthetic Pixel 7",
            manufacturer="Google (Simulated)",
            soc="Tensor G2",
            android_api=34,
            capabilities=RunnerCapabilities(
                supported_delegates=["cpu", "gpu", "nnapi"],
                can_measure_headroom=True,
                can_control_screen=True,
                safety_temp_ceiling_c=45.0,
                min_battery_pct=15.0,
            ),
            current_telemetry=TelemetrySnapshot(
                timestamp_ns=int(time.time() * 1_000_000_000),
                thermal_status=self.thermal.step(0.0, 0.0).thermal_status,
                thermal_headroom=self.thermal.step(0.0, 0.0).thermal_headroom,
                battery_temp_c=curr_temp,
                battery_pct=self.battery_pct,
                charging=False,
                screen_on=True,
            ),
        )

    def run(self, spec: RunRequest) -> RunEntry:
        """Execute a controlled synthetic run conforming to RunRequest specification."""
        if self.is_aborted:
            raise RuntimeError("SyntheticRunner is in aborted state")

        # Check safety constraints
        if self.battery_pct < spec.constraints.min_battery_pct:
            raise RuntimeError(
                f"Safety abort: Battery {self.battery_pct}% is below required {spec.constraints.min_battery_pct}%"
            )

        n = spec.measured_iters
        latencies: list[float] = []
        thermal_statuses: list[int] = []
        thermal_headrooms: list[float | None] = []
        battery_temps: list[float] = []

        # Determine effective fallback
        fallback = FallbackMode.NONE
        if self.fault_mode == DiagnosisLabel.DELEGATE_FALLBACK and spec.delegate in (
            "gpu",
            "nnapi",
        ):
            fallback = FallbackMode.FULL
        elif self.secondary_fault == "fallback" and spec.delegate in ("gpu", "nnapi"):
            fallback = FallbackMode.FULL

        executed_delegate = spec.delegate

        for i in range(n):
            sim = simulate_iteration(
                model_name=self.model_name,
                requested_delegate=spec.delegate,
                iteration_idx=i,
                total_iterations=n,
                fallback_mode=fallback,
                rng=self.rng,
            )
            lat = sim.raw_latency_ms
            executed_delegate = sim.delegate_executed

            # Warmup drift fault (only manifests in COLD fresh process)
            if (
                self.fault_mode == DiagnosisLabel.WARMUP_DRIFT or self.secondary_fault == "warmup"
            ) and spec.process_start == ProcessStart.COLD:
                if i < 4:
                    decay = np.exp(-0.8 * i)
                    lat += sim.raw_latency_ms * (3.2 * decay)

            # Thermal throttling fault
            if self.fault_mode == DiagnosisLabel.THERMAL_THROTTLING:
                th_state = self.thermal.step(power_w=sim.power_w * 2.3, dt_s=lat / 1000.0)
                lat = lat / th_state.frequency_ratio
            else:
                th_state = self.thermal.step(power_w=sim.power_w, dt_s=lat / 1000.0)

            # Safety enforcement during run
            if th_state.temperature_c >= spec.constraints.max_battery_temp_c:
                self.abort()
                raise RuntimeError(
                    f"Thermal safety cutoff triggered: temperature reached {th_state.temperature_c:.1f}C (limit: {spec.constraints.max_battery_temp_c}C)"
                )
            if th_state.thermal_status >= spec.constraints.abort_on_thermal_status:
                self.abort()
                raise RuntimeError(
                    f"Thermal status cutoff triggered: reached level {th_state.thermal_status} (limit: {spec.constraints.abort_on_thermal_status})"
                )

            latencies.append(lat)
            thermal_statuses.append(th_state.thermal_status)
            thermal_headrooms.append(th_state.thermal_headroom)
            battery_temps.append(th_state.temperature_c)

        # Contention noise application (suppressed under quiet_mode)
        if (
            self.fault_mode == DiagnosisLabel.BACKGROUND_NOISE or self.secondary_fault == "noise"
        ) and not self.quiet_mode:
            arr_noisy, _ = self.noise_gen.apply_noise(np.array(latencies), quiet_mode=False)
            latencies = [float(v) for v in arr_noisy]

        t0 = int(time.time() * 1_000_000_000)
        avg_step = int(np.mean(latencies) * 1_000_000) if latencies else 20_000_000
        timestamps_ns = [t0 + i * avg_step for i in range(n)]

        # Map rerun type based on spec
        rerun_type = RerunType.COLD_WARM
        if spec.delegate == "cpu":
            rerun_type = RerunType.DELEGATE_SWAP

        return RunEntry(
            rerun=rerun_type,
            process_start=spec.process_start,
            delegate_requested=spec.delegate,
            delegate_executed=executed_delegate,
            samples_ms=latencies,
            timestamps_ns=timestamps_ns,
            telemetry=TelemetrySeries(
                thermal_status=thermal_statuses,
                thermal_headroom=thermal_headrooms,
                battery_temp_c=battery_temps,
                charging=[False] * n,
                screen_on=[True] * n,
            ),
            notes=[f"Synthetic run: mode={self.fault_mode.value}, delegate={executed_delegate}"],
        )

    def rest(self, target_temp_c: float, max_wait_s: float = 300.0) -> bool:
        """Simulate thermal recovery until temperature reaches target_temp_c."""
        if self.thermal.current_temp_c <= target_temp_c:
            return True
        states = self.thermal.cool_down(rest_duration_s=min(max_wait_s, 60.0), idle_power_w=0.15)
        return bool(states[-1].temperature_c <= target_temp_c)

    def abort(self) -> None:
        """Abort execution."""
        self.is_aborted = True
