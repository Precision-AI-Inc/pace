# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Declare preprocess, train and evaluate up front so the whole job can be estimated.

Situation: a job with three stages of very different size. A scheduler that only saw the
training loop could not know about the stages after it, so ``plan`` lists them with their
rough share of the runtime, and each stage runs under the name the plan gave it.

Shows: ``plan([...])`` and ``phase(...)``, in the three shapes a stage can take:

* ``phase("Preprocess", weight=0.10)``: a short stage with no count of its own, created
  first so that the plan can take the phase itself, with its weight next to it.
* ``with phase("Train"): for ... in track(...)``: a long counted stage. A phase has no rate
  of its own, so the reader uses the estimate of the loop inside it. While it runs the job
  estimate is marked ``partial``: the reader cannot know whether more loops will follow in
  the same phase. To avoid that, make the counted loop the planned stage itself, as
  ``Evaluate`` is.
* ``track(..., description="Evaluate")``: the counted scope itself is the planned stage.

There is no ``root=True`` here: the plan covers the whole job, which counts as complete
once every planned stage has ended. Plan weights are a guess at each stage's share of the
**runtime** (not of the item counts) and only matter for stages that have not started.
Each plan entry must match the id of the scope that does that stage's work: ``"Train"``
becomes ``train``, which is the phase. The loop inside it has its own id and a ``p=train``
parent.

Run: ``python examples/phases_and_plan.py``. Add ``PACE_MIN_INTERVAL=0`` to see every
progress line.

Expect: a ``Plan`` line listing ``preprocess 10%, train 80%, evaluate 10%``, then each stage
in order. The ``Train steps`` scope carries ``p=train``.
"""

from __future__ import annotations

import time

from precisionai.pace import logger, phase, plan, track


def preprocess() -> None:
    """Stand in for tokenizing and caching the data."""
    time.sleep(0.1)


def train_step(step: int) -> float:
    """Stand in for one optimizer step and return its loss."""
    time.sleep(0.004)
    return 1.0 / (1 + step)


def evaluate_batch(batch: int) -> float:
    """Stand in for scoring one validation batch."""
    time.sleep(0.01)
    return 0.9 + batch * 0.001


def main() -> None:
    """Run the three stages, declared by a plan."""
    # Write the plan before the first stage starts. A plan entry is a phase or a
    # (name, weight) pair.
    prep = phase("Preprocess", weight=0.10)
    plan([prep, ("Train", 0.80), ("Evaluate", 0.10)])

    with prep:
        preprocess()

    with phase("Train"):
        for step in track(range(100), description="Train steps", unit="step"):
            train_step(step)

    scores = [evaluate_batch(batch) for batch in track(range(10), description="Evaluate", unit="batch")]
    logger.success(f"Best validation score {max(scores):.3f}")


if __name__ == "__main__":
    main()
