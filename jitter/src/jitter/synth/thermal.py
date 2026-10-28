"""First-order RC thermal simulation model for mobile SoC physical dynamics.

Models temperature rise under computational workload, heat dissipation to ambient,
frequency governor throttling steps, and cooling recovery curves during rest periods.

Governing differential equation (Euler discretization):
    T_{t+1} = T_t + (dt / (R * C)) * (T_ambient - T_t) + (P_t * dt / C)
where:
    R = Thermal resistance to ambient (K / W)
    C = Lumped thermal capacitance (J / K)
    P_t = Active power dissipation (W)
    dt = Elapsed duration (seconds)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ThermalState:
    """Instantaneous thermal state of the device."""

    temperature_c: float
    thermal_status: int
    thermal_headroom: float
    frequency_ratio: float


def temperature_to_thermal_status(temp_c: float) -> int:
    """Map temperature in Celsius to Android PowerManager thermal status (0 - 6).

    Status levels:
        0: THERMAL_STATUS_NONE      (< 35 C)
        1: THERMAL_STATUS_LIGHT     (35 - 39 C)
        2: THERMAL_STATUS_MODERATE  (39 - 43 C)
        3: THERMAL_STATUS_SEVERE    (43 - 47 C)
        4: THERMAL_STATUS_CRITICAL  (47 - 51 C)
        5: THERMAL_STATUS_EMERGENCY (51 - 55 C)
        6: THERMAL_STATUS_SHUTDOWN  (>= 55 C)
    """
    if temp_c < 35.0:
        return 0
    if temp_c < 39.0:
        return 1
    if temp_c < 43.0:
        return 2
    if temp_c < 47.0:
        return 3
    if temp_c < 51.0:
        return 4
    if temp_c < 55.0:
        return 5
    return 6


def thermal_status_to_frequency_ratio(status: int) -> float:
    """Determine processor frequency scaling ratio f(T) / f_max given thermal status."""
    scaling_table = {
        0: 1.00,  # NONE: full turbo / nominal clock
        1: 0.95,  # LIGHT: minor governor downclock
        2: 0.82,  # MODERATE: sustained throttle
        3: 0.65,  # SEVERE: aggressive thermal throttle
        4: 0.48,  # CRITICAL: minimum sustained frequency
        5: 0.30,  # EMERGENCY: hard safety clamp
        6: 0.15,  # SHUTDOWN: near stall
    }
    return scaling_table.get(status, 1.00)


def calculate_thermal_headroom(temp_c: float, ceiling_c: float = 43.0) -> float:
    """Calculate PowerManager.getThermalHeadroom forecast (1.0 = throttle boundary)."""
    # 0.0 at 25 C ambient, 1.0 at 43 C ceiling
    headroom = (ceiling_c - temp_c) / (ceiling_c - 25.0)
    return float(np.clip(headroom, 0.0, 1.5))


class RCThermalModel:
    """Simulates first-order RC thermal dynamics across benchmark iterations and cooldowns."""

    def __init__(
        self,
        r_ambient: float = 10.0,  # K / W thermal resistance
        c_thermal: float = 50.0,  # J / K heat capacity
        ambient_temp_c: float = 25.0,
        initial_temp_c: float = 28.0,
    ) -> None:
        self.r = r_ambient
        self.c = c_thermal
        self.ambient_temp_c = ambient_temp_c
        self.current_temp_c = initial_temp_c

    def step(self, power_w: float, dt_s: float) -> ThermalState:
        """Advance thermal model by dt_s seconds under power dissipation power_w."""
        # T_{t+1} = T_t + (dt / (R * C)) * (T_amb - T_t) + (P * dt / C)
        cooling = (dt_s / (self.r * self.c)) * (self.ambient_temp_c - self.current_temp_c)
        heating = (power_w * dt_s) / self.c
        self.current_temp_c += cooling + heating

        status = temperature_to_thermal_status(self.current_temp_c)
        freq_ratio = thermal_status_to_frequency_ratio(status)
        headroom = calculate_thermal_headroom(self.current_temp_c)

        return ThermalState(
            temperature_c=float(self.current_temp_c),
            thermal_status=status,
            thermal_headroom=headroom,
            frequency_ratio=freq_ratio,
        )

    def cool_down(self, rest_duration_s: float, idle_power_w: float = 0.2) -> list[ThermalState]:
        """Simulate cooldown recovery over rest_duration_s."""
        dt = 0.5
        n_steps = int(np.ceil(rest_duration_s / dt))
        states = []
        for _ in range(n_steps):
            states.append(self.step(idle_power_w, dt))
        return states
