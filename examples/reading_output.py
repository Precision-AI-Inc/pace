# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Read another program's Pace output and print the completion estimate as it changes.

Situation: the consumer side, what a scheduler or a dashboard does. It starts a job as a
child process, reads its output as it arrives, and turns it into an estimate. Pace ships
every step so a Python consumer and the terminal's own "time left" run the same code:

    bytes -> LineSplitter -> parse_line -> Record -> JobForecast -> JobEstimate

Shows: ``LineSplitter`` (a pipe delivers bytes in arbitrary chunks, so it is the one that
finds line ends), ``parse_line`` (returns ``None`` for an ordinary log line and never
raises), ``JobForecast.observe`` and ``snapshot``. ``received_at`` and ``now`` come from the
reader's own monotonic clock, not from the log, because the reader has to know when it
received a line to tell a stalled job from a slow one.

A real reader keeps the default ``ForecastConfig``, which waits for 60 seconds of
observations before it calls an estimate ``live``, and calls ``snapshot`` on a timer as
well as after each record, so a silent job turns ``stale``. This demo watches a job of a
second, so it lowers ``min_observed_seconds`` to be able to show a live estimate at all.
Only the trailer at the end of a ``PROGRESS`` line is a contract: never parse the readable
text in front of it.

Run: ``python examples/reading_output.py``. The child is ``examples/track_loop.py`` with
``PACE_MIN_INTERVAL=0`` so that it writes every update, and ``PACE_BAR=never`` so that it
writes lines even on a terminal.

Expect: a few lines, each a state change or a second-ish update: ``learning`` first, then
``live`` with the seconds remaining and ``source=root``, and finally ``complete`` when the
root scope ends. The last line is the child's exit code.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from precisionai.pace import ForecastConfig, JobEstimate, JobForecast, LineSplitter, Record, parse_line

CHILD = Path(__file__).resolve().with_name("track_loop.py")
PRINT_EVERY = 0.2  # seconds between lines while the state does not change


def describe(record: Record, estimate: JobEstimate) -> str:
    """Return one line saying what the record was and what the estimate is now."""
    remaining = "-" if estimate.remaining_seconds is None else f"{estimate.remaining_seconds:.1f}s"
    return (
        f"{record.scope} {record.event.value:<5} {estimate.state.value:<8} "
        f"remaining={remaining:<6} source={estimate.source}"
    )


def main() -> None:
    """Run the child and print its estimate whenever the state changes, and now and then."""
    child = subprocess.Popen(
        [sys.executable, str(CHILD)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # one pipe for both streams
        env={**os.environ, "PACE_MIN_INTERVAL": "0", "PACE_BAR": "never"},
    )
    if child.stdout is None:
        raise RuntimeError("the child was started without a stdout pipe")
    splitter = LineSplitter()
    job = JobForecast(config=ForecastConfig(min_observed_seconds=0.2))

    last_state = None
    last_printed = 0.0
    while True:
        chunk = os.read(child.stdout.fileno(), 65536)  # whatever has arrived, split anywhere
        if not chunk:
            break
        for line in splitter.feed(chunk):
            record = parse_line(line)
            if record is None:
                continue  # an ordinary log line
            now = time.monotonic()
            job.observe(record, received_at=now)
            estimate = job.snapshot(now)
            if estimate.state is not last_state or now - last_printed >= PRINT_EVERY:
                print(describe(record, estimate), flush=True)
                last_state, last_printed = estimate.state, now

    splitter.flush()  # a last line with no trailing newline, if any
    print(f"child exited with {child.wait()}", flush=True)


if __name__ == "__main__":
    main()
