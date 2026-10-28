"""Synthetic generation package for Jitter."""

from jitter.synth.delegates import (
    MODEL_CATALOG,
    DelegateProfile,
    FallbackMode,
    ModelArchitectureProfile,
    SimulatedIteration,
    get_delegate_profile,
    simulate_iteration,
)
from jitter.synth.noise import (
    ContentionBurst,
    ContentionNoiseGenerator,
)
from jitter.synth.scenarios import (
    generate_scenario_report,
)
from jitter.synth.thermal import (
    RCThermalModel,
    ThermalState,
    calculate_thermal_headroom,
    temperature_to_thermal_status,
    thermal_status_to_frequency_ratio,
)

__all__ = [
    "RCThermalModel",
    "ThermalState",
    "temperature_to_thermal_status",
    "thermal_status_to_frequency_ratio",
    "calculate_thermal_headroom",
    "FallbackMode",
    "DelegateProfile",
    "ModelArchitectureProfile",
    "MODEL_CATALOG",
    "SimulatedIteration",
    "get_delegate_profile",
    "simulate_iteration",
    "ContentionBurst",
    "ContentionNoiseGenerator",
    "generate_scenario_report",
]
