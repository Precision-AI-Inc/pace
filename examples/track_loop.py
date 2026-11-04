# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Report a plain loop with ``track`` and mark it as the job's root.

Situation: an inference job that scores 100 samples one at a time. The loop is nearly the
whole job, so it is the root: the scheduler's estimate for the job is this loop's, and the
job counts as complete when the loop ends.

Shows: ``track`` with keyword arguments (``description=``, ``unit=``) and ``root=True``.
The total comes from ``len(samples)``; pass ``total=`` for an iterable with no length (see
``unknown_total.py``).

Run: ``python examples/track_loop.py``. Add ``PACE_MIN_INTERVAL=0`` to see every progress
line (a real job writes one at most every 15 seconds).

Expect: a ``start`` line, update lines when ``PACE_MIN_INTERVAL=0``, an ``end`` line, then
``SUCCESS``. On a terminal the lines are replaced by one bar that redraws in place. Every
progress line ends with the ``[pace inference ... root]`` trailer a scheduler reads.
"""

import time

from precisionai.pace import logger, track


def score(sample: int) -> float:
    """Stand in for one inference step."""
    time.sleep(0.005)
    return sample * 0.01


def main() -> None:
    """Score every sample, reporting progress."""
    logger.info("Loading samples")
    samples = list(range(100))

    scores = [score(sample) for sample in track(samples, description="Inference", unit="sample", root=True)]

    logger.success(f"Scored {len(scores)} samples")


if __name__ == "__main__":
    main()
