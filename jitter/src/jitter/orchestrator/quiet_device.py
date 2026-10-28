"""Quiet device controlled rerun module.

Evaluates latency variance under quieted system conditions (e.g., flight mode,
background services suppressed) to isolate OS background contention noise.
"""

from __future__ import annotations

from jitter.runners.base import BaseRunner
from jitter.schema.protocol import RunRequest
from jitter.schema.runs import ProcessStart, RerunType, RunEntry


def run_quiet_device_rerun(
    runner: BaseRunner,
    base_request: RunRequest,
) -> RunEntry:
    """Execute benchmark under quiet conditions to test noise suppression."""
    spec = base_request.model_copy(
        update={
            "process_start": ProcessStart.COLD,
        }
    )
    run_entry = runner.run(spec)
    return run_entry.model_copy(update={"rerun": RerunType.QUIET_DEVICE})
