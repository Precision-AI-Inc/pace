# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Bring a library's standard ``logging`` output into Pace's format with ``capture_logging``.

Situation: your code uses Pace, but the libraries it calls (PyTorch, Transformers, a data
loader) log through the standard ``logging`` module. Their lines would otherwise come out
in another format, on stderr, interleaved with yours. ``ShardStore`` below stands in for
such a library: it knows nothing about Pace.

Shows: ``capture_logging(level="INFO")``, which adds one handler to the root logger and is
opt-in (importing Pace changes no other library's logging), and
``logging.captureWarnings(True)``, which sends ``warnings.warn`` messages to the same place.
Records below the level (the ``debug`` call here) are not routed. If your program already
called ``logging.basicConfig(...)``, remove that call when you adopt Pace: every record
would otherwise be written twice. A library that prints its own lines or sets
``propagate = False`` (Lightning and Hugging Face Transformers do) is named in
``capture_logging(take_over=[...])`` after it is imported; see ``docs/logging.md``.

Run: ``python examples/stdlib_logging.py``. Add ``PACE_MIN_INTERVAL=0`` to see every
progress line.

Expect: the library's ``INFO``, ``WARNING`` and ``ERROR`` records in Pace's line format on
stdout, with the traceback of the logged exception, the warning from ``warnings.warn``
(followed, as ``captureWarnings`` always does, by the source line that raised it), and a
``Load shards`` progress scope. The ``debug`` record does not appear.
"""

import logging
import time
import warnings

from precisionai.pace import capture_logging, logger, track


class ShardStore:
    """Stand in for a third-party library that logs with the standard library."""

    log = logging.getLogger("shardstore")

    def open(self, shards: int) -> None:
        """Open the store, logging what it found."""
        self.log.info("Found %d shards", shards)
        self.log.debug("Not routed: below the INFO level")
        self.log.warning("Slow disk: %.1f MB/s", 41.7)

    def read(self, shard: int) -> list:
        """Read one shard."""
        time.sleep(0.01)
        if shard == 3:
            try:
                raise OSError("checksum mismatch")
            except OSError:
                self.log.exception("Shard %d is corrupt, skipping it", shard)
                return []
        return [shard] * 4


def main() -> None:
    """Capture the library's logging, then use it."""
    capture_logging(level="INFO")
    logging.captureWarnings(True)

    store = ShardStore()
    store.open(shards=6)
    warnings.warn("tf32 is disabled", UserWarning, stacklevel=1)

    rows = 0
    for shard in track(range(6), description="Load shards", unit="shard", root=True):
        rows += len(store.read(shard))

    logger.success(f"Loaded {rows} rows")


if __name__ == "__main__":
    main()
