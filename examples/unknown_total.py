# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Report a stream with no known length, and set the total when it becomes known.

Situation: records are read from a paged source. At the start nobody knows how many there
are, but the first page says (``total_count``). Until then the scope has no total, so there
is no percentage and no estimate, only a count, a rate and a heartbeat that shows the job is
alive. As soon as the total is known, ``set_total`` gives the reader what it needs to
estimate. Do not invent a total to make a loop estimable: an honest "unknown" is better
than a wrong number. If the total is never known, leave it out and report that gap.

Shows: ``progress(description=..., unit=..., root=True)`` with no ``total``, ``run.advance(n)``
and ``run.set_total(n)``. When the total is known up front, ``track(stream, total=n, ...)``
is enough.

Run: ``python examples/unknown_total.py``. Add ``PACE_MIN_INTERVAL=0`` to see every progress
line.

Expect: a ``start`` line that reads ``0 record · no estimate`` and carries ``t=?``, then lines
with a percentage once the first page has been read, then ``end``.
"""

from __future__ import annotations

import time
from typing import Any, Iterator

from precisionai.pace import logger, progress

PAGE_SIZE = 10
RECORDS = 60


def fetch_pages() -> Iterator[dict[str, Any]]:
    """Yield pages of records, the way a paged API would; each page says how many there are in all."""
    for first in range(0, RECORDS, PAGE_SIZE):
        time.sleep(0.05)
        yield {"records": list(range(first, first + PAGE_SIZE)), "total_count": RECORDS}


def main() -> None:
    """Read every page, learning the total from the first one."""
    stored = 0
    with progress(description="Reading records", unit="record", root=True) as run:
        for page in fetch_pages():
            if run.total is None:
                run.set_total(page["total_count"])
            stored += len(page["records"])
            run.advance(len(page["records"]))

    logger.success(f"Stored {stored} records")


if __name__ == "__main__":
    main()
