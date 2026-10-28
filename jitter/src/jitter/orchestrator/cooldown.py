"""Cooldown and thermal stabilization controlled rerun module.

Monitors device thermal state, rests device until baseline temperature is reached,
and executes a rested cold rerun to isolate thermal throttling effects.
"""

from __future__ import annotations

from jitter.runners.base import BaseRunner
from jitter.schema.protocol import RunRequest
from jitter.schema.runs import ProcessStart, RerunType, RunEntry


def run_cooldown_rerun(
    runner: BaseRunner,
    base_request: RunRequest,
    target_temp_c: float = 32.0,
    max_rest_s: float = 180.0,
) -> RunEntry:
    """Rest runner until cooled down to target temperature, then execute benchmark."""
    # Rest to target thermal baseline
    runner.rest(target_temp_c=target_temp_c, max_wait_s=max_rest_s)

    # Execute from rested state
    spec = base_request.model_copy(update={"process_start": ProcessStart.COLD})
    entry = runner.run(spec)
    return entry.model_copy(update={"rerun": RerunType.COOLDOWN})
