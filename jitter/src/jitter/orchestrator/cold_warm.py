"""Cold vs Warm controlled rerun module.

Isolates warmup drift (JIT compilation, GPU shader compilation, weight loading)
by comparing a fresh cold process against immediate subsequent warm runs.
"""

from __future__ import annotations

from jitter.runners.base import BaseRunner
from jitter.schema.protocol import RunRequest
from jitter.schema.runs import ProcessStart, RerunType, RunEntry


def run_cold_warm_pair(
    runner: BaseRunner,
    base_request: RunRequest,
) -> tuple[RunEntry, RunEntry]:
    """Execute cold fresh-process run followed immediately by in-process warm run."""
    # 1. Fresh cold process
    cold_spec = base_request.model_copy(update={"process_start": ProcessStart.COLD})
    cold_run = runner.run(cold_spec)
    cold_run = cold_run.model_copy(update={"rerun": RerunType.COLD_WARM})

    # 2. Warm in-process execution
    warm_spec = base_request.model_copy(
        update={"process_start": ProcessStart.WARM, "warmup_iters": 0}
    )
    warm_run = runner.run(warm_spec)
    warm_run = warm_run.model_copy(update={"rerun": RerunType.COLD_WARM})

    return cold_run, warm_run
