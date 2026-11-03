# Getting started

Pace is a logger with progress reporting built in. You use it the way you would
use any logger. Each progress line also ends with a short machine-readable trailer
(`[pace:1 ...]`) that a scheduler can read to estimate when the job will finish.
You do not have to do anything extra for that.

## Install

```bash
pip install precisionai-pace
```

Python 3.8 or later. The core package has no dependencies, so it adds nothing to
your training image. Framework callbacks are optional extras, see
[Framework integrations](frameworks.md).

## First log lines

```python
from precisionai.pace import logger

logger.info("Loading model")
logger.success("Model ready")
logger.warning("Falling back to cached metadata")
```

```text
2026-11-03T17:02:23Z INFO     Loading model
2026-11-03T17:02:23Z SUCCESS  Model ready
2026-11-03T17:02:23Z WARNING  Falling back to cached metadata
```

Every line is a UTC timestamp (ISO-8601, so it means the same thing in any
container), the level padded to eight characters, and the message. The levels are
`debug`, `info`, `success`, `warning`, `error` and `critical`. The default minimum
level is `INFO`, so `debug` lines are dropped until you ask for them. See
[Logging](logging.md).

## First progress loop

Wrap any iterable in `track`:

```python
import time

from precisionai.pace import logger, track

logger.info("Loading model")
for step in track(range(600), description="Train", unit="step"):
    time.sleep(0.25)  # your training step
logger.success("Training finished")
```

`track` yields the same items and reports progress as the loop consumes them. The
total is taken from `len()` when the iterable has one; pass `total=` otherwise.
Everything you can report, and the other forms (`progress`, `timed`, `phase`,
`plan`), are in [Progress](progress.md).

## What you see in a terminal

When stdout is a terminal, progress is one bar that redraws in place. This is the
same loop with 300 steps. For the first minute the bar reads `estimating…`; then it shows the rate and the time left:

```text
2026-11-03T17:14:05Z INFO     Loading model
Train  ━━━━━━━━────────────  42.7%  128/300 step · estimating…
Train  ━━━━━━━━━━━━━━━━━───  88.3%  265/300 step · 3.98 step/s · ~9s
```

(Those are two frames of the same line, not two lines.) When the loop finishes, the
bar is replaced by a normal line that includes the trailer, so the result stays in
your scrollback:

```text
2026-11-03T17:15:13Z PROGRESS Train  100.0%  300/300 step · 3.97 step/s · done in 1m16s  [pace:1 s=train k=end c=300 t=300 e=75.6 q=302]
2026-11-03T17:15:13Z SUCCESS  Training finished
```

The bar is drawn about ten times a second and is not written to a file.

## What you see in a container or a log file

When stdout is not a terminal, Pace writes ordinary lines, one at a time, and
rate-limits them. This is the 600-step loop above at about four steps per second, trimmed in the
middle:

```text
2026-11-03T17:03:59Z INFO     Loading model
2026-11-03T17:03:59Z PROGRESS Train  0.0%  0/600 step · estimating…  [pace:1 s=train k=start c=0 t=600 u=step e=0 q=1 run=29b7e7e2 n=Train]
2026-11-03T17:04:11Z PROGRESS Train  9.8%  59/600 step · estimating…  [pace:1 s=train k=upd c=59 t=600 e=15.1 q=2]
2026-11-03T17:04:27Z PROGRESS Train  19.8%  119/600 step · estimating…  [pace:1 s=train k=upd c=119 t=600 e=30.2 q=3]
2026-11-03T17:04:39Z PROGRESS Train  29.8%  179/600 step · estimating…  [pace:1 s=train k=upd c=179 t=600 e=45.4 q=4]
2026-11-03T17:04:54Z PROGRESS Train  39.8%  239/600 step · 3.95 step/s · ~1m31s  [pace:1 s=train k=upd c=239 t=600 e=60.6 q=5]
...
2026-11-03T17:06:17Z PROGRESS Train  100.0%  600/600 step · 3.97 step/s · done in 2m31s  [pace:1 s=train k=end c=600 t=600 e=151.5 q=12]
2026-11-03T17:06:17Z SUCCESS  Training finished
```

Things to notice:

- The text before `[pace:1 ...]` is for people. The bracketed trailer is the
  stable, versioned part a program reads. Its format is in
  [Design: line format](design.md#line-format-version-1).
- A line is written at most every 15 seconds per scope, and only when the work
  moved at least 1% or two minutes passed. A ten-hour run writes a few hundred
  lines, not one per step. `start`, `end` and `fail` are always written at once.
  See [Progress: how often lines are written](progress.md#how-often-lines-are-written).
- The first minute reads `estimating…`. The estimate needs 60 seconds of
  observations, three measured intervals and 1% of the work done before it shows a
  time. See the [FAQ](faq.md).
- Colour is only used on a terminal and never inside the trailer.

## Where output goes

Everything goes to **stdout**, flushed after every line. Progress and status are a
job's normal output, and Python block-buffers stdout when it is not a terminal, so
without the flush a container would hold your progress back for minutes.

If your program writes data to stdout that another program reads, send Pace to
stderr instead:

```bash
PACE_STREAM=stderr python train.py
```

or, in code, `logger.configure(stream="stderr")`. A reader that sees both streams
removes the duplicates; see [Reading Pace output](reading-output.md).

## Next

- [Examples](https://github.com/Precision-AI-Inc/pace/blob/main/examples/README.md): one
  runnable script per style of logging and progress. Copy the one that matches your job.
- [Logging](logging.md): levels, `exception`, configuration, capturing the
  standard `logging` module.
- [Progress](progress.md): `track`, `progress`, `timed`, `phase`, `plan`,
  resuming, and what makes the estimate good.
- [Migrating](migrating.md): from `print`, `tqdm`, Loguru and `logging`.
- [Configuration](configuration.md): every environment variable.
