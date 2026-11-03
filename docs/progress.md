# Progress

Progress reporting is what makes Pace more than a logger. You describe the work
(how much there is, what a unit is) and Pace writes the progress lines, draws the
terminal bar and works out the time remaining.

## The five tools

| Tool | Use it for |
|---|---|
| `track(iterable, ...)` | A loop over something you can iterate. The common case. |
| `progress(total, ...)` | A count you advance yourself: concurrent work, callbacks, loops with irregular steps. |
| `timed` | A function that is one step of the job (`@timed("Load model")`). |
| `phase(name)` | A named stage of a multi-stage job. Scopes opened inside it name it as their parent. |
| `plan([...])` | Declares the stages and their weights up front, so the whole job can be estimated. |

All of them create a **scope**: one piece of work with a start, a completed amount,
an optional total, a unit, and an end. Each scope has an id (`s=` in the trailer),
derived from its description (`"Fine-tune 7B"` becomes `fine-tune-7b`) or given
with `scope="..."`. Ids are unique within a process: if you open a second scope
called `Train`, it is `train-2`.

## A plain loop

```python
from precisionai.pace import track

for batch in track(loader, description="Inference", unit="batch"):
    infer(batch)
```

- `track` yields your items unchanged.
- The total is `len(loader)` when the iterable has a length. Pass `total=` for
  one that does not (a generator, an `IterableDataset`).
- An item counts as completed when the loop asks for the next one, so the count
  never runs ahead of the work.
- `description` is the label people see and `unit` the word after the count and
  rate (`step`, `batch`, `sample`, `file`, `byte`).

## A manual counter

When the work is not a loop you can wrap, open a scope with `progress` and advance
it yourself. This one counts requests as they finish, in any order:

```python
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from precisionai.pace import progress


def evaluate(request: int) -> int:
    time.sleep(0.05)
    return request * 2


requests = list(range(40))

with ThreadPoolExecutor(max_workers=4) as pool:
    futures = [pool.submit(evaluate, r) for r in requests]
    with progress(total=len(futures), description="Evaluation", unit="request") as run:
        for future in as_completed(futures):
            future.result()
            run.advance()
```

```text
2026-11-03T17:06:30Z PROGRESS Evaluation  0.0%  0/40 request · estimating…  [pace:1 s=evaluation k=start c=0 t=40 u=request e=0 q=1 run=6bfc3dd5 n=Evaluation]
2026-11-03T17:06:26Z PROGRESS Evaluation  42.5%  17/40 request · estimating…  [pace:1 s=evaluation k=upd c=17 t=40 e=0.3 q=2]
2026-11-03T17:06:26Z PROGRESS Evaluation  82.5%  33/40 request · estimating…  [pace:1 s=evaluation k=upd c=33 t=40 e=0.5 q=3]
2026-11-03T17:06:26Z PROGRESS Evaluation  100.0%  40/40 request · 72.7 request/s · done in 1s  [pace:1 s=evaluation k=end c=40 t=40 e=0.5 q=4]
```

(Run with `PACE_MIN_INTERVAL=0.2` so the short example writes more than start and end.)

`progress` returns a handle with three methods:

| Method | Effect |
|---|---|
| `run.advance(n=1)` | Add `n` to the completed amount. |
| `run.update(completed)` | Set the completed amount. Use it when you know the absolute count, such as `global_step`. A value that is not a non-negative number is ignored. |
| `run.set_total(total)` | Change the total. `None` or a non-positive number makes it unknown. |

The scope starts when the `with` block is entered. It ends when the block exits
normally. If an exception leaves the block, it is written as `fail` (with the count it
reached) and the exception continues to propagate; Pace never swallows it.

The handle is safe to call from several threads. A scope opened *in* a worker thread
does not see the `with` block of the main thread as its parent, because threads
start with the context they were created in; see [Nested scopes](#nested-scopes).

If you cannot use a `with` block, call `scope = progress(...)`, then `scope.start()`,
and finish with `scope.end()` or `scope.fail()`. Each does nothing if the scope has
already ended, and an `advance` before `start` starts the scope.

## When you do not know the total

Leave `total` out. The line shows a count with no percentage and no bar:

```python
from precisionai.pace import progress

with progress(description="Reading shards", unit="shard") as run:
    for shard in shards:
        read(shard)
        run.advance()
```

```text
2026-11-03T17:07:42Z PROGRESS Reading shards  0 shard · no estimate  [pace:1 s=reading-shards k=start c=0 t=? u=shard e=0 q=1 run=6a68257f n=Reading%20shards]
2026-11-03T17:07:43Z PROGRESS Reading shards  17 shard · no estimate  [pace:1 s=reading-shards k=upd c=17 t=? e=0.6 q=2]
```

`t=?` tells a reader the total is unknown. Nothing can be estimated, but the
count, the rate and the heartbeat still show that the job is alive. If you learn
the total later, call `run.set_total(40)`. The next line carries the new total and
the estimate begins from there; the rate seen so far is kept.

```text
2026-11-03T17:07:43Z PROGRESS Reading shards  80.0%  32/40 shard · estimating…  [pace:1 s=reading-shards k=upd c=32 t=40 e=1 q=3]
```

With an unknown total, lines are written on the heartbeat only (there is no 1%
to measure); see [How often lines are written](#how-often-lines-are-written).

## Timing a function

`@timed` turns each call of a function into a scope that starts when the call begins
and ends when it returns:

```python
from precisionai.pace import logger, timed


@timed("Load model")
def load_model(): ...


@timed
def tokenize(): ...
```

```text
2026-11-03T17:08:16Z PROGRESS Load model  0.0%  0/1 call · estimating…  [pace:1 s=load-model k=start c=0 t=1 u=call e=0 q=1 run=a22feb8b n=Load%20model]
2026-11-03T17:08:16Z PROGRESS Load model  100.0%  1/1 call · done in 0s  [pace:1 s=load-model k=end c=1 t=1 e=0.1 q=2]
```

With no description the function's name is used. If the function raises, the scope
is written as `fail` and the exception propagates. A generator or coroutine function
is timed only until the call returns the generator or coroutine object, not until
it is exhausted.

## Phases and a plan

A job often has stages of very different size: preprocess for a minute, train for
nine hours, evaluate for ten minutes. A scheduler that only sees the training loop
cannot know about the stages after it. `plan` declares them up front with weights:

```python
from precisionai.pace import phase, plan, track

plan([("Preprocess", 0.05), ("Train", 0.9), ("Evaluate", 0.05)])

with phase("Preprocess"):
    preprocess()

for batch in track(train_loader, description="Train", unit="step"):
    train_step(batch)

for batch in track(val_loader, description="Evaluate", unit="batch"):
    evaluate_step(batch)
```

`plan` writes one record that lists the phases and weights:

```text
2026-11-03T17:12:54Z PROGRESS Plan  preprocess 5%, train 90%, evaluate 5%  [pace:1 s=plan k=plan e=0 q=1 ph=preprocess:0.05,train:0.9,evaluate:0.05]
```

How a reader uses it (see [Design: forecasting rules](design.md#several-scopes)):
finished phases count zero, the running phase uses its own estimate, and each phase
not yet started is estimated as `weight x seconds-per-weight`, where seconds-per-weight is
taken from the phases that have finished or are running.

Rules that make this work:

- **Each plan entry names the scope that will do that stage's work.** A name is turned into
  a scope id the same way a description is, so `("Train", 0.9)` matches
  `track(..., description="Train")` and `phase("Train")`. If you pass
  `scope="train-steps"` to the loop, the plan must list `train-steps`.
  Two entries with the same name become `train` and `train-2`, like the scopes
  they will match.
- **The running stage needs a count somewhere.** A `phase(...)` is a scope with total 1
  that only reports 0 of 1 at the start and 1 of 1 at the end, so it has no rate of its
  own. When the long stage is `with phase("Train"): for ... in track(...)`, the reader
  uses the estimate of the loop inside the phase. Putting the count on the stage the
  plan names (`track(..., description="Train")`, as above) works too. A phase with no
  counted scope inside it has no estimate until it ends.
- Weights are relative. Entries without a weight share what the others leave of 1.0 equally.
  Pass `Phase` objects (`prep = phase("Preprocess", weight=0.05)`, then
  `plan([prep, ("Train", 0.9)])`) if you prefer the weight next to the stage.
- Write the plan before the first stage starts. A later plan replaces an earlier one.
- Weights are your estimate of relative **duration**, not of work units. Only stages
  that have not started use them; a finished or running stage is measured. A rough guess
  from one earlier run costs accuracy only for the stages still to come.

Phases with `phase` and `timed` are also useful on their own, to mark where time
goes without a plan:

```python
with phase("Export"):
    export_onnx()
    upload_artifacts()
```

## Nested scopes

A scope opened while another is open names it as its parent (`p=` in the trailer).
Validation inside an epoch loop is the standard case:

```python
from precisionai.pace import progress, track

with progress(total=3, description="Epochs", unit="epoch", root=True) as epochs:
    for epoch in range(3):
        for batch in track(train_loader, description="Train", unit="step", scope=f"train-{epoch}"):
            train_step(batch)
        for batch in track(val_loader, description="Validate", unit="batch", scope=f"val-{epoch}"):
            validate_step(batch)
        epochs.advance()
```

```text
2026-11-03T17:07:40Z PROGRESS Epochs  0.0%  0/3 epoch · estimating…  [pace:1 s=epochs root=1 k=start c=0 t=3 u=epoch e=0 q=1 run=648e2e0d n=Epochs]
2026-11-03T17:07:40Z PROGRESS Train  0.0%  0/20 step · estimating…  [pace:1 s=train-0 p=epochs k=start c=0 t=20 u=step e=0 q=1 run=648e2e0d n=Train]
2026-11-03T17:07:40Z PROGRESS Train  100.0%  20/20 step · 45 step/s · done in 0s  [pace:1 s=train-0 p=epochs k=end c=20 t=20 e=0.4 q=3]
2026-11-03T17:07:40Z PROGRESS Validate  0.0%  0/5 batch · estimating…  [pace:1 s=val-0 p=epochs k=start c=0 t=5 u=batch e=0 q=1 run=648e2e0d n=Validate]
2026-11-03T17:07:40Z PROGRESS Epochs  33.3%  1/3 epoch · estimating…  [pace:1 s=epochs root=1 k=upd c=1 t=3 e=0.6 q=2]
```

A nested scope is never added on top of its parent's remaining time: the parent's
own estimate already includes the time the children take.

Nesting follows the Python context. It works across function calls, but a scope
opened in a new thread (or an `asyncio` task created before the parent opened) does
not see its parent. If you need it, run the thread's body inside a copied context
(`contextvars.copy_context().run(...)`).

Each pass through the loop above creates new scopes (`train-0`, `train-1`, ...). That
is fine. If you leave out `scope=`, repeated scopes of the same description are numbered
for you (`train`, `train-2`, `train-3`).

## Resuming from a checkpoint

If a job continues from a checkpoint, start the scope at the checkpoint's count. A
scope that starts at 0 and then reports 3140 after a few seconds looks, to a reader, as
if 3140 steps took those few seconds, and the estimate is far too optimistic.

```python
start_step = 3100  # from the checkpoint you loaded
total_steps = 3200

with progress(total=total_steps, description="Train", unit="step",
              completed=start_step, resumed=True) as run:
    for _ in range(start_step, total_steps):
        train_step()
        run.advance()
```

```text
2026-11-03T17:07:44Z PROGRESS Train  96.9%  3100/3200 step · estimating…  [pace:1 s=train k=start c=3100 t=3200 u=step e=0 q=1 res=1 run=6cbe4833 n=Train]
2026-11-03T17:07:45Z PROGRESS Train  98.1%  3140/3200 step · estimating…  [pace:1 s=train k=upd c=3140 t=3200 e=0.4 q=2]
```

`completed=` is what matters: it sets the count the scope starts from, and the reader
measures rates from the work done after that first record. `resumed=True` additionally
writes `res=1` on the start record, to say that the first count is not zero because of
earlier work. With the reference estimator in this package the estimate is the same
with or without `res=1`; set it anyway, so that a reader that wants to show "resumed
from 3100" can. A job that uses `update(global_step)` instead of `advance()` should
still start the scope with `completed=global_step`. `track` takes the same arguments:

```python
for batch in track(loader, description="Train", unit="step",
                   total=total_steps, completed=start_step, resumed=True):
    ...
```

`track` counts from `completed`, so iterate only the remaining items (skip the ones
the checkpoint already covers).

## Choosing a root scope

When a job has several scopes, a scheduler has to decide which one stands for the
whole job. Mark it with `root=True`:

```python
with progress(total=total_steps, description="Train", unit="step", root=True) as run:
    ...
```

The effect, from a reader's side (this is real output from `JobForecast`, 400 of
1000 steps done at one step per second):

| | State | Remaining | `source` | `partial` |
|---|---|---|---|---|
| No root, no plan | live | 600 s | `concurrent` | `True` |
| `root=True` | live | 600 s | `root` | `False` |

The number is the same. What changes is what a scheduler may conclude from it:

- **`partial=True`** means "this covers only the scopes seen so far". Work that has
  not started (an evaluation stage, an export) is not in the number, and the
  scheduler should treat it as a lower bound.
- **Completion.** With a root, the job is complete when the root ends and failed
  when the root fails. Without one, a job is never called complete unless a plan lists every
  stage and they have all ended, because between two unplanned stages every scope seen
  so far has ended and more work may follow. Process exit is then the only real end.
- With a root, the other scopes still appear in the output but do not decide the
  estimate.

Pick the scope that spans the job and counts steadily: total optimizer steps rather
than epochs when the number of epochs is small, because a three-step root gives the
estimate only three data points. If the job is a sequence of stages of different size,
a [plan](#phases-and-a-plan) is usually better than a root.

Use at most one root per process. Pace does not stop you from marking two, but a reader
takes the first it sees.

## How often lines are written

On a terminal, Pace redraws one bar about ten times a second and writes no trailer
until the scope ends. When output is not a terminal, it writes a new progress line for
a scope only when **both** hold:

1. at least `PACE_MIN_INTERVAL` seconds have passed since that scope's last line
   (default 15), **and**
2. the scope has advanced by at least 1% of its total since the last line, **or**
   `PACE_HEARTBEAT` seconds have passed since it (default 120).

`start`, `end`, `fail` and `plan` are always written at once. The limit is on time,
not on step count: a loop of a million steps and a loop of a hundred write the same number of
lines per minute. A ten-hour run reaches each 1% about every six minutes, so it writes
about a hundred progress lines plus a heartbeat every two minutes in between.

The heartbeat exists so that a reader can tell a slow job from a stalled one. It is
written when the code reports progress (`advance`, `update`, `set_total`, an item from
`track`), not by a timer. A process that hangs writes nothing, and the silence is how a
reader marks the scope as stale.

Change the limits with `PACE_MIN_INTERVAL` and `PACE_HEARTBEAT` (seconds) or
`configure(min_interval=..., heartbeat=...)`:

```bash
PACE_MIN_INTERVAL=5 PACE_HEARTBEAT=30 python train.py
```

Lower values give a scheduler fresher data and write more lines. With both at `0`, every
reported update is written. Raising `PACE_HEARTBEAT` past the time a scheduler waits
before it calls a job stale makes a healthy job look stuck; keep them consistent
with the reader's own settings (`ForecastConfig.heartbeat_seconds`, see
[Reading Pace output](reading-output.md)).

## The terminal bar

When stdout is a terminal and `PACE_BAR` is `auto` (the default), the bar is drawn in
place and a log line you write in between erases the bar, is printed, and the bar is redrawn
below it, so the two never overlap. Set the mode with `PACE_BAR` or `configure(bar=...)`:

| `PACE_BAR` | Result |
|---|---|
| `auto` | Bar on a terminal, normal rate-limited lines otherwise. |
| `never` | Normal lines everywhere, even on a terminal. |
| `always` | Bar everywhere, even when the output is a file. |

Use `never` when a job runs under a pseudo-terminal that something else reads, such
as a scheduler that allocates a TTY for `docker run -t`, `ssh -t` or a notebook kernel
and also parses the logs. The bar carries no trailer, so a reader sees nothing but
the start and the end. With `never` you get normal trailer lines even on a TTY:

```bash
PACE_BAR=never python train.py
```

`always` on a file writes `\r` redraws into it. That is rarely what you want.

The bar uses `━` and `─` on a UTF-8 terminal and `#` and `-` otherwise
(see [Encodings](logging.md#encodings-and-the-ascii-fallback)).

## `track()` ends the scope on `break` and on exceptions in the body

A generator cannot tell `break` from an exception raised in the loop body. Python
delivers both as the same `GeneratorExit`. So `track` writes `end` with the count reached
in either case, and writes `fail` only when the iterable itself raises:

```python
for item in track(range(5), description="Body crash"):
    if item == 2:
        raise ValueError("bad sample")   # the scope is written as `end`, c=2 of 5
```

```text
2026-11-03T17:08:17Z PROGRESS Body crash  40.0%  2/5 it · done in 0s  [pace:1 s=body k=end c=2 t=5 e=0 q=2]
```

The exception still propagates and your program still crashes as usual. But the
trailer says `end`, and a reader that trusts it may think this scope completed. If a
crash in the body must be marked as a failure, use `progress`, which reports `fail`
for any exception leaving the block:

```python
with progress(total=len(loader), description="Train") as run:
    for batch in loader:
        train_step(batch)   # an exception here is written as `fail`
        run.advance()
```

Abandoning a `track` loop with `break` or `return` also writes `end` with the count
reached, because the loop finished on your terms. Either way, a reader should treat
process exit, not a scope's `end`, as the end of the job unless there is a `root` scope.
