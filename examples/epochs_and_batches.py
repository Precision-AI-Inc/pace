# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Report epochs times batches as one counted scope, not one bar per epoch.

Situation: a training loop of 4 epochs with 25 batches each. The scheduler needs one smooth
rate over the whole job, so the root scope counts every batch of every epoch (100 steps)
and each epoch is only a log line. One scope per epoch would restart the count 4 times and
give the estimate 25 points to learn from instead of 100.

Shows: ``progress(total=epochs * len(loader), root=True)`` advanced once per batch, and a
``logger.info`` line per epoch. Validation stays inside the root as plain work: the
estimate is a little optimistic by the evaluation time still to come (a known gap, not a
reason to give it a root of its own). To see each epoch as a scope with its own bar, see
``nested_scopes.py``.

Run: ``python examples/epochs_and_batches.py``. Add ``PACE_MIN_INTERVAL=0`` to see every
progress line.

Expect: one ``Train`` scope counting ``0/100 step`` to ``100/100 step``, with an ``INFO`` line
at the end of each epoch.
"""

from __future__ import annotations

import time

from precisionai.pace import logger, progress

EPOCHS = 4
BATCHES_PER_EPOCH = 25


def train_step(batch: int) -> float:
    """Stand in for one optimizer step and return its loss."""
    time.sleep(0.004)
    return 1.0 / (1 + batch)


def main() -> None:
    """Train for a few epochs, counting steps across all of them."""
    loader = list(range(BATCHES_PER_EPOCH))
    total_steps = EPOCHS * len(loader)

    with progress(total=total_steps, description="Train", unit="step", root=True) as run:
        for epoch in range(EPOCHS):
            loss = 0.0
            for batch in loader:
                loss = train_step(batch)
                run.advance()
            logger.info(f"epoch {epoch + 1}/{EPOCHS} done, last loss {loss:.3f}")

    logger.success("Training finished")


if __name__ == "__main__":
    main()
