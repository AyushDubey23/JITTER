"""Background contention noise model with Poisson-Pareto burst arrivals.

Mobile operating systems schedule background tasks (syncing, indexing, telemetry, GC,
push notifications) that create bursty, heavy-tailed contention without shifting the baseline.

Features:
- Poisson arrival process for background contention bursts.
- Heavy-tailed Pareto distribution for burst duration (many 1-3 iteration spikes, rare long stalls).
- Contention multipliers (1.5x to 5.0x) applied exclusively during active bursts.
- Clean unperturbed baseline between bursts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ContentionBurst:
    """An instance of a background resource contention burst."""

    start_iteration: int
    duration_iterations: int
    multiplier: float
    additive_delay_ms: float


class ContentionNoiseGenerator:
    """Generates realistic Poisson-Pareto background contention bursts."""

    def __init__(
        self,
        arrival_rate_per_iter: float = 0.05,  # Average 1 burst per 20 iterations
        pareto_alpha: float = 1.5,  # Heavy-tail shape (lower = heavier tails)
        min_duration_iters: int = 1,
        multiplier_range: tuple[float, float] = (1.5, 4.5),
        seed: int | None = None,
    ) -> None:
        self.arrival_rate = arrival_rate_per_iter
        self.pareto_alpha = pareto_alpha
        self.min_duration = min_duration_iters
        self.multiplier_range = multiplier_range
        self.rng = np.random.default_rng(seed)

    def generate_burst_schedule(self, total_iterations: int) -> list[ContentionBurst]:
        """Pre-generate schedule of contention bursts across total_iterations."""
        bursts: list[ContentionBurst] = []
        curr = 0

        while curr < total_iterations:
            # Geometric / Exponential inter-arrival distance
            wait = int(self.rng.geometric(p=max(0.01, min(0.99, self.arrival_rate))))
            start = curr + wait
            if start >= total_iterations:
                break

            # Pareto-distributed duration: (U^(-1/alpha) - 1) * xm + xm
            u = self.rng.uniform(1e-4, 1.0)
            duration = int(np.floor(self.min_duration / (u ** (1.0 / self.pareto_alpha))))
            # Cap maximum single burst duration to avoid infinite stall
            duration = max(1, min(duration, max(5, total_iterations // 3)))

            # Multiplier
            mult = float(self.rng.uniform(self.multiplier_range[0], self.multiplier_range[1]))
            # Optional additive delay for I/O spikes
            additive = float(self.rng.exponential(scale=3.0))

            bursts.append(
                ContentionBurst(
                    start_iteration=start,
                    duration_iterations=duration,
                    multiplier=mult,
                    additive_delay_ms=additive,
                )
            )
            curr = start + duration

        return bursts

    def apply_noise(
        self,
        latencies_ms: np.ndarray,
        quiet_mode: bool = False,
    ) -> tuple[np.ndarray, list[ContentionBurst]]:
        """Apply contention bursts to a sequence of base latencies.

        If quiet_mode is True, bursts are suppressed (simulating flight mode / quiet device).
        """
        arr = np.copy(latencies_ms)
        n = arr.size
        if n == 0 or quiet_mode:
            return arr, []

        bursts = self.generate_burst_schedule(n)

        for b in bursts:
            end = min(n, b.start_iteration + b.duration_iterations)
            for i in range(b.start_iteration, end):
                arr[i] = arr[i] * b.multiplier + b.additive_delay_ms

        return arr, bursts
