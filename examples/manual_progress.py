# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Count results that arrive out of order with ``progress`` and ``advance``.

Situation: an evaluation harness that sends 40 requests through a pool of 4 workers and
collects the answers as they finish. There is no loop to wrap, so the code advances a
counter itself, from the thread that collects the results. A scope opened inside a worker
thread would not see its parent, and counting results (not submissions) keeps the count
honest.

Shows: ``progress(total=..., root=True)``, ``run.advance()``, and why the ``progress`` block
is entered together with the pool but after it: it exits first, so an exception from
``future.result()`` is written as ``fail`` and the exception still propagates.

Run: ``python examples/manual_progress.py``. Add ``PACE_MIN_INTERVAL=0`` to see every
progress line.

Expect: an ``Evaluation`` scope counting ``0/40 request`` to ``40/40 request``, then ``SUCCESS``
with the mean score.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from precisionai.pace import logger, progress


def evaluate(request: int) -> float:
    """Stand in for one request to a model server."""
    time.sleep(0.02)
    return request % 5 / 4


def main() -> None:
    """Evaluate every request, advancing as each one finishes."""
    requests = list(range(40))
    scores = []

    with (
        ThreadPoolExecutor(max_workers=4) as pool,
        progress(total=len(requests), description="Evaluation", unit="request", root=True) as run,
    ):
        futures = [pool.submit(evaluate, request) for request in requests]
        for future in as_completed(futures):
            scores.append(future.result())
            run.advance()

    logger.success(f"Mean score {sum(scores) / len(scores):.3f} over {len(scores)} requests")


if __name__ == "__main__":
    main()
