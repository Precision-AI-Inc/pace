# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Time one-off steps with ``@timed`` around a counted loop.

Situation: a job that loads a model, trains for a while and saves results after every epoch.
Loading and saving are single steps with nothing to count, but they take time that people
and schedulers want to see. ``@timed`` turns each call into a scope of one unit that starts
when the call begins and ends when it returns (or is written as ``fail`` if it raises).

Shows: ``@timed("Load model")``, a bare ``@timed`` (the function's name is the label), and
``@timed`` on a step called repeatedly: each call is a new scope (``save_results``,
``save_results-2``, ...). The saves run inside the root loop, so they are children of it
(``p=train``) and never added on top of its estimate. Work that runs after the root ends is
invisible to a scheduler: if a final step were a large share of the runtime, use a plan
(see ``phases_and_plan.py``).

Run: ``python examples/timed_steps.py``. Add ``PACE_MIN_INTERVAL=0`` to see every progress
line.

Expect: ``Load model`` start and end lines, the ``Train`` scope, and a start and end line
for each of the three ``Save results`` calls, nested in it.
"""

from __future__ import annotations

import time

from precisionai.pace import logger, progress, timed

EPOCHS = 3
STEPS_PER_EPOCH = 20


@timed("Load model")
def load_model() -> dict:
    """Stand in for reading weights from disk."""
    time.sleep(0.1)
    return {"weights": [0.0] * 8}


@timed
def save_results(epoch: int) -> None:
    """Stand in for writing metrics and a checkpoint. With no label, the function name is used."""
    time.sleep(0.03)
    logger.info(f"Saved results for epoch {epoch}")


def train_step(model: dict) -> None:
    """Stand in for one optimizer step."""
    model["weights"][0] += 0.01
    time.sleep(0.004)


def main() -> None:
    """Load the model, train, and save after each epoch."""
    model = load_model()

    with progress(total=EPOCHS * STEPS_PER_EPOCH, description="Train", unit="step", root=True) as run:
        for epoch in range(1, EPOCHS + 1):
            for _ in range(STEPS_PER_EPOCH):
                train_step(model)
                run.advance()
            save_results(epoch)

    logger.success("Training finished")


if __name__ == "__main__":
    main()
