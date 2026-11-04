# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Open a child scope inside a parent scope, and see what nesting means for the estimate.

Situation: 3 epochs, each a training loop followed by a validation loop. The epochs are the
parent and each loop is a child. A scope opened while another is open names it as its
parent (``train-0<epochs`` in the trailer), following the Python context, so it works across
function calls.

What nesting means for estimates: a reader never adds a child's remaining time on top of its
parent's. The parent's own estimate already includes the time its children take, so the
job's estimate is the root's. Here the root counts epochs, so it has a rate of its own only
after three updates. Until then the reader builds the root's estimate from the children:
after the first epoch from the time an epoch took (training and validation together),
during the first from the time so far plus what the running loop has left. When the epoch
count is small, one root over total steps gives a steadier estimate sooner (see
``epochs_and_batches.py``). Children also give people the detail: which loop is running,
and how far it is.

Each pass creates new scopes, so give them distinct ids with ``scope=`` (``train-0``,
``val-0``, ...). Without ``scope=``, repeated descriptions are numbered for you (``train``,
``train-2``, ...). A scope opened in a new thread does not see its parent.

Run: ``python examples/nested_scopes.py``. Add ``PACE_MIN_INTERVAL=0`` to see every progress
line.

Expect: an ``Epochs`` scope marked ``root`` and, indented inside it, ``Train`` and
``Validate`` scopes whose trailers carry ``<epochs``.
"""

import time

from precisionai.pace import logger, progress, track

EPOCHS = 3


def train_step(batch: int) -> float:
    """Stand in for one optimizer step and return its loss."""
    time.sleep(0.01)
    return 1.0 / (1 + batch)


def validate_step(batch: int) -> float:
    """Stand in for scoring one validation batch."""
    time.sleep(0.01)
    return 0.8 + batch * 0.01


def run_epoch(epoch: int) -> None:
    """Train and validate once; both scopes are children of the open ``Epochs`` scope."""
    for batch in track(range(10), description="Train", unit="step", scope=f"train-{epoch}"):
        train_step(batch)
    scores = [
        validate_step(batch) for batch in track(range(4), description="Validate", unit="batch", scope=f"val-{epoch}")
    ]
    logger.info(f"epoch {epoch + 1}/{EPOCHS} validation score {max(scores):.3f}")


def main() -> None:
    """Run every epoch under one parent scope."""
    with progress(total=EPOCHS, description="Epochs", unit="epoch", scope="epochs", root=True) as epochs:
        for epoch in range(EPOCHS):
            run_epoch(epoch)
            epochs.advance()

    logger.success("Training finished")


if __name__ == "__main__":
    main()
