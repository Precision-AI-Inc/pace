# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Show that only rank 0 writes progress in a distributed run, with two simulated ranks.

Situation: a multi-GPU job started by ``torchrun``, Accelerate or SLURM runs the same script
on every rank. A scheduler needs one set of progress records per job, not one per GPU.
Pace reads the rank from the first of ``RANK``, ``LOCAL_RANK`` and ``SLURM_PROCID`` that
holds an integer. Rank 0 writes progress; on every other rank ``track``, ``progress``,
``timed``, ``phase`` and ``plan`` still run your code and write nothing. So the training
code is written exactly as for one process, with no ``if rank == 0`` around progress calls.

Log lines are different: ``logger.info`` and the rest are written by every rank. Keep the
rank guard around messages that should appear once (``run_rank`` shows both kinds).

To run without GPUs, this script starts itself twice as a child process with ``RANK=0`` and
``RANK=1``, the way a launcher would, and prints each child's output with a ``[rank N]``
prefix. In a real job the launcher does this and you write only ``run_rank``. The rank is
read each time a line is written, so a launcher sets it before the script starts. Setting
``PACE_ALL_RANKS=1`` writes progress from every rank, which is only for debugging: scope ids
are not qualified by rank.

Run: ``python examples/distributed.py``. Add ``PACE_MIN_INTERVAL=0`` to see every progress
line.

Expect: both ranks log that they started, and only the ``[rank 0]`` lines include ``Train``
progress. Rank 1 prints its own log line and no progress.
"""

import os
import subprocess
import sys
import time

from precisionai.pace import logger, progress

WORLD_SIZE = 2
CHILD_MARKER = "PACE_EXAMPLE_CHILD"
TOTAL_STEPS = 50


def run_rank() -> None:
    """Train as one rank of a distributed job: the code every rank runs."""
    rank = int(os.environ.get("RANK", "0"))

    if rank == 0:
        logger.info(f"Starting on {WORLD_SIZE} ranks")  # one copy, guarded by hand
    logger.info(f"Rank {rank} ready")  # every rank writes this one

    # No rank guard: rank 0 writes the records, the other ranks run the loop silently.
    with progress(total=TOTAL_STEPS, description="Train", unit="step", root=True) as run:
        for _ in range(TOTAL_STEPS):
            time.sleep(0.005)
            run.advance()


def launch() -> None:
    """Start one child per simulated rank and print what each wrote."""
    children = []
    for rank in range(WORLD_SIZE):
        env = {**os.environ, "RANK": str(rank), "WORLD_SIZE": str(WORLD_SIZE), CHILD_MARKER: "1"}
        child = subprocess.Popen(
            [sys.executable, os.path.abspath(__file__)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            encoding="utf-8",
        )
        children.append((rank, child))

    for rank, child in children:
        output, _ = child.communicate()
        for line in output.splitlines():
            print(f"[rank {rank}] {line}", flush=True)
        if child.returncode:
            raise SystemExit(child.returncode)


if __name__ == "__main__":
    if os.environ.get(CHILD_MARKER):
        run_rank()
    else:
        launch()
