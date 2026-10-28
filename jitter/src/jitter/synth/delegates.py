"""Hardware execution delegate profiles and fallback simulation models.

Simulates execution characteristics for CPU, GPU, and NNAPI hardware acceleration
across standard vision and language model architectures.

Supports realistic fallback failure modes:
1. FULL: Whole model routed to CPU despite GPU/NNAPI requested.
2. PARTIAL: Graph partitioning due to unsupported ops; 20% on CPU + buffer marshaling.
3. MID_RUN: Execution begins on GPU/NNAPI, then crashes or spills to CPU mid-run.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class FallbackMode(str, Enum):
    """Failure mode for hardware delegate execution."""

    NONE = "none"
    FULL = "full"
    PARTIAL = "partial"
    MID_RUN = "mid_run"


@dataclass(frozen=True)
class DelegateProfile:
    """Latency and power profile for a specific hardware delegate."""

    delegate_name: str
    base_latency_ms: float
    latency_noise_std_ms: float
    power_w: float
    thermal_sensitivity: float  # sensitivity to clock downclocking


@dataclass(frozen=True)
class ModelArchitectureProfile:
    """Baseline characteristics across delegates for a specific model."""

    name: str
    size_bytes: int
    profiles: dict[str, DelegateProfile]


MODEL_CATALOG: dict[str, ModelArchitectureProfile] = {
    "mobilenet_v2": ModelArchitectureProfile(
        name="mobilenet_v2.tflite",
        size_bytes=14_200_000,
        profiles={
            "cpu": DelegateProfile(
                delegate_name="cpu",
                base_latency_ms=38.0,
                latency_noise_std_ms=0.6,
                power_w=2.8,
                thermal_sensitivity=0.8,
            ),
            "gpu": DelegateProfile(
                delegate_name="gpu",
                base_latency_ms=13.5,
                latency_noise_std_ms=1.4,
                power_w=4.5,
                thermal_sensitivity=1.2,
            ),
            "nnapi": DelegateProfile(
                delegate_name="nnapi",
                base_latency_ms=8.5,
                latency_noise_std_ms=0.9,
                power_w=3.4,
                thermal_sensitivity=1.0,
            ),
        },
    ),
    "mobilenet_v3": ModelArchitectureProfile(
        name="mobilenet_v3.tflite",
        size_bytes=11_800_000,
        profiles={
            "cpu": DelegateProfile(
                delegate_name="cpu",
                base_latency_ms=29.0,
                latency_noise_std_ms=0.5,
                power_w=2.5,
                thermal_sensitivity=0.8,
            ),
            "gpu": DelegateProfile(
                delegate_name="gpu",
                base_latency_ms=10.8,
                latency_noise_std_ms=1.2,
                power_w=4.2,
                thermal_sensitivity=1.2,
            ),
            "nnapi": DelegateProfile(
                delegate_name="nnapi",
                base_latency_ms=7.0,
                latency_noise_std_ms=0.8,
                power_w=3.1,
                thermal_sensitivity=1.0,
            ),
        },
    ),
    "resnet50": ModelArchitectureProfile(
        name="resnet50.tflite",
        size_bytes=98_000_000,
        profiles={
            "cpu": DelegateProfile(
                delegate_name="cpu",
                base_latency_ms=145.0,
                latency_noise_std_ms=1.5,
                power_w=3.5,
                thermal_sensitivity=0.9,
            ),
            "gpu": DelegateProfile(
                delegate_name="gpu",
                base_latency_ms=44.0,
                latency_noise_std_ms=3.2,
                power_w=5.8,
                thermal_sensitivity=1.3,
            ),
            "nnapi": DelegateProfile(
                delegate_name="nnapi",
                base_latency_ms=29.0,
                latency_noise_std_ms=2.0,
                power_w=4.6,
                thermal_sensitivity=1.1,
            ),
        },
    ),
    "bert_tiny": ModelArchitectureProfile(
        name="bert_tiny.tflite",
        size_bytes=17_500_000,
        profiles={
            "cpu": DelegateProfile(
                delegate_name="cpu",
                base_latency_ms=62.0,
                latency_noise_std_ms=1.0,
                power_w=3.0,
                thermal_sensitivity=0.85,
            ),
            "gpu": DelegateProfile(
                delegate_name="gpu",
                base_latency_ms=23.0,
                latency_noise_std_ms=2.2,
                power_w=5.0,
                thermal_sensitivity=1.25,
            ),
            "nnapi": DelegateProfile(
                delegate_name="nnapi",
                base_latency_ms=16.5,
                latency_noise_std_ms=1.3,
                power_w=3.8,
                thermal_sensitivity=1.05,
            ),
        },
    ),
}


@dataclass(frozen=True)
class SimulatedIteration:
    """Result of a single simulated iteration."""

    raw_latency_ms: float
    power_w: float
    delegate_executed: str


def get_delegate_profile(model_name: str, delegate: str) -> DelegateProfile:
    """Look up profile from catalog or construct synthetic default."""
    model = MODEL_CATALOG.get(model_name, MODEL_CATALOG["mobilenet_v2"])
    return model.profiles.get(delegate, model.profiles["cpu"])


def simulate_iteration(
    model_name: str,
    requested_delegate: str,
    iteration_idx: int,
    total_iterations: int,
    fallback_mode: FallbackMode = FallbackMode.NONE,
    rng: np.random.Generator | None = None,
) -> SimulatedIteration:
    """Simulate a single iteration with specified delegate and fallback behavior."""
    gen = rng if rng is not None else np.random.default_rng()
    req_profile = get_delegate_profile(model_name, requested_delegate)
    cpu_profile = get_delegate_profile(model_name, "cpu")

    if fallback_mode == FallbackMode.FULL:
        # Whole model executes on CPU
        executed_delegate = "cpu"
        noise = gen.normal(0.0, cpu_profile.latency_noise_std_ms)
        latency = max(1.0, cpu_profile.base_latency_ms + noise)
        power = cpu_profile.power_w

    elif fallback_mode == FallbackMode.PARTIAL:
        # 80% on requested delegate, 20% on CPU + graph partitioning marshaling cost
        executed_delegate = "cpu+gpu" if requested_delegate == "gpu" else "cpu+nnapi"
        # 0.8 * GPU + 0.2 * CPU + 8ms memory copy & synchronization overhead
        marshaling_cost = 8.0
        combined_base = (
            0.8 * req_profile.base_latency_ms + 0.2 * cpu_profile.base_latency_ms + marshaling_cost
        )
        noise = gen.normal(0.0, req_profile.latency_noise_std_ms + 0.5)
        latency = max(1.0, combined_base + noise)
        power = 0.8 * req_profile.power_w + 0.2 * cpu_profile.power_w

    elif fallback_mode == FallbackMode.MID_RUN:
        # Crashes/falls back at 35% of total iterations
        cutoff = max(5, int(0.35 * total_iterations))
        if iteration_idx < cutoff:
            executed_delegate = requested_delegate
            noise = gen.normal(0.0, req_profile.latency_noise_std_ms)
            latency = max(1.0, req_profile.base_latency_ms + noise)
            power = req_profile.power_w
        elif iteration_idx == cutoff:
            # Crash/context-switch penalty on the exact boundary iteration
            executed_delegate = "cpu"
            latency = cpu_profile.base_latency_ms * 2.5
            power = cpu_profile.power_w
        else:
            executed_delegate = "cpu"
            noise = gen.normal(0.0, cpu_profile.latency_noise_std_ms)
            latency = max(1.0, cpu_profile.base_latency_ms + noise)
            power = cpu_profile.power_w

    else:
        # Normal execution on requested delegate
        executed_delegate = requested_delegate
        noise = gen.normal(0.0, req_profile.latency_noise_std_ms)
        latency = max(1.0, req_profile.base_latency_ms + noise)
        power = req_profile.power_w

    return SimulatedIteration(
        raw_latency_ms=float(latency),
        power_w=float(power),
        delegate_executed=executed_delegate,
    )
