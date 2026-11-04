# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Resume from a checkpoint without the estimate counting the earlier work.

Situation: a 100-step training job that was stopped after 70 steps and is started again
from its checkpoint. If the scope started at 0 and then reported 75, a reader would think
75 steps took the few seconds since the restart and give a far too optimistic estimate. The
scope starts at the checkpoint's count instead, and the total stays the full total (not the
remaining work).

Shows: ``progress(total=..., completed=start_step, resumed=start_step > 0, root=True)``.
``completed=`` sets the count the scope starts from, and ``resumed=True`` writes the
``res`` flag on the ``start`` record to say the first count is not zero because of earlier work.
``track`` takes the same two arguments; iterate only the items the checkpoint does not
cover.

Run: ``python examples/resume.py``. Add ``PACE_MIN_INTERVAL=0`` to see every progress line.

Expect: first a ``Checkpoint`` log line saying the run resumes at step 70, then a ``Train``
scope whose ``start`` line reads ``70/100 step`` and carries ``res``, and finishes at
``100/100``.
"""

import json
import tempfile
import time
from pathlib import Path

from precisionai.pace import logger, progress

TOTAL_STEPS = 100


def train_step() -> None:
    """Stand in for one optimizer step."""
    time.sleep(0.01)


def load_checkpoint_step(path: Path) -> int:
    """Return the global step stored in the checkpoint, or 0 when there is none."""
    if not path.exists():
        return 0
    return int(json.loads(path.read_text(encoding="utf-8"))["step"])


def main() -> None:
    """Resume training from a checkpoint written by an earlier run."""
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "checkpoint.json"
        checkpoint.write_text(json.dumps({"step": 70}), encoding="utf-8")  # the earlier run's last save

        start_step = load_checkpoint_step(checkpoint)
        logger.info(f"Checkpoint found, resuming at step {start_step}")

        with progress(
            total=TOTAL_STEPS,
            description="Train",
            unit="step",
            root=True,
            completed=start_step,
            resumed=start_step > 0,
        ) as run:
            for _ in range(start_step, TOTAL_STEPS):
                train_step()
                run.advance()

    logger.success("Training finished")


if __name__ == "__main__":
    main()
