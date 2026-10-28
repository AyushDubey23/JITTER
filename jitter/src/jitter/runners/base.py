"""Abstract base class for benchmark execution runners."""

from __future__ import annotations

from abc import ABC, abstractmethod

from jitter.schema.protocol import RunnerInfoResponse, RunRequest
from jitter.schema.runs import RunEntry


class BaseRunner(ABC):
    """Abstract interface for all benchmark execution environments (Android, Desktop, Synthetic)."""

    @abstractmethod
    def info(self) -> RunnerInfoResponse:
        """Query hardware capabilities and current telemetry snapshot."""
        ...

    @abstractmethod
    def run(self, spec: RunRequest) -> RunEntry:
        """Execute a controlled benchmark run according to spec."""
        ...

    @abstractmethod
    def rest(self, target_temp_c: float, max_wait_s: float = 300.0) -> bool:
        """Wait until device cools down to target_temp_c or timeout expires."""
        ...

    @abstractmethod
    def abort(self) -> None:
        """Immediately abort any running benchmark and restore safe device state."""
        ...
