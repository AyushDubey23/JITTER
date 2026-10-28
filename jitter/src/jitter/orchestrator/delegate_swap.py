"""Delegate swap controlled rerun module.

Compares requested target accelerator against CPU baseline and available
alternate execution delegates to isolate hardware delegate fallbacks.
"""

from __future__ import annotations

from jitter.runners.base import BaseRunner
from jitter.schema.protocol import RunRequest
from jitter.schema.runs import ProcessStart, RerunType, RunEntry


def run_delegate_swap_set(
    runner: BaseRunner,
    base_request: RunRequest,
) -> list[RunEntry]:
    """Execute reference CPU run and alternate accelerator runs."""
    entries: list[RunEntry] = []

    # 1. Mandatory reference CPU run
    cpu_spec = base_request.model_copy(
        update={
            "delegate": "cpu",
            "process_start": ProcessStart.COLD,
        }
    )
    cpu_run = runner.run(cpu_spec)
    entries.append(cpu_run.model_copy(update={"rerun": RerunType.DELEGATE_SWAP}))

    # 2. Check for alternate accelerator
    info = runner.info()
    avail = info.capabilities.supported_delegates
    target = base_request.delegate

    # If requested is GPU and NNAPI is available, run NNAPI
    alt_delegate: str | None = None
    if target == "gpu" and "nnapi" in avail:
        alt_delegate = "nnapi"
    elif target == "nnapi" and "gpu" in avail:
        alt_delegate = "gpu"

    if alt_delegate:
        alt_spec = base_request.model_copy(
            update={
                "delegate": alt_delegate,
                "process_start": ProcessStart.COLD,
            }
        )
        alt_run = runner.run(alt_spec)
        entries.append(alt_run.model_copy(update={"rerun": RerunType.DELEGATE_SWAP}))

    return entries
