# Migrating to Pace

Pace is not a drop-in replacement for any of these libraries. The notes below show
the equivalent call for the common patterns and say plainly what has no equivalent.

Three differences apply to every migration:

- Pace writes to **stdout**, flushed after every line, and to nothing else. It has no
  file sinks, rotation or handlers. To keep a copy in a file, redirect the stream
  (`python train.py | tee run.log`) or pass a file object to
  `logger.configure(stream=...)`.
- Pace writes **one line format**. There is no format string, no JSON mode, and no per-call
  context fields.
- When output is not a terminal, progress is **rate-limited** (a line at most every
  15 seconds per scope), not drawn every step. See
  [How often lines are written](progress.md#how-often-lines-are-written).

## From `print`

| Before | After |
|---|---|
| `print("Loading model")` | `logger.info("Loading model")` |
| `print(f"epoch {e} loss {loss:.3f}")` | `logger.info(f"epoch {e} loss {loss:.3f}")` |
| `print("Warning: no GPU", file=sys.stderr)` | `logger.warning("No GPU")` |
| `print(msg, flush=True)` | `logger.info(msg)` (every line is flushed already) |
| `traceback.print_exc()` | `logger.exception("Step failed")` inside the `except` |
| `for x in data: ...` with a hand-written `if i % 100 == 0: print(i, len(data))` | `for x in track(data, description="Load", unit="item"): ...` |

```python
# before
for i, batch in enumerate(loader):
    if i % 100 == 0:
        print(f"batch {i}/{len(loader)}")
    infer(batch)

# after
from precisionai.pace import track

for batch in track(loader, description="Inference", unit="batch"):
    infer(batch)
```

Keep per-step values that are not progress (loss, learning rate) as ordinary lines at
a sensible cadence:

```python
if step % 100 == 0:
    logger.info(f"step {step} loss {loss:.3f}")
```

Pace is not a metrics tracker. Loss curves belong in MLflow, Weights & Biases or
similar.

## From `tqdm`

`track` is the equivalent of `tqdm(iterable)`; `progress` is the equivalent of
`tqdm(total=...)` with manual `update` calls.

| `tqdm` | Pace |
|---|---|
| `from tqdm import tqdm` | `from precisionai.pace import track, progress` |
| `for x in tqdm(it)` | `for x in track(it)` |
| `tqdm(it, desc="Train")` | `track(it, description="Train")` |
| `tqdm(it, total=100)` | `track(it, total=100)` |
| `tqdm(it, unit="batch")` | `track(it, unit="batch")` |
| `trange(100)` | `track(range(100))` |
| `tqdm(it, total=5000, initial=3100)` | `track(it, total=5000, completed=3100, resumed=True)` (iterate only the remaining items) |
| `with tqdm(total=100) as bar:` | `with progress(total=100) as run:` |
| `bar.update(1)` / `bar.update(n)` | `run.advance()` / `run.advance(n)` |
| `bar.n = 40; bar.refresh()` | `run.update(40)` |
| `bar.total = 200` | `run.set_total(200)` |
| `bar.close()` | Leave the `with` block (or `run.end()`) |
| `tqdm.write(msg)` | `logger.info(msg)` |
| `tqdm(it, file=sys.stderr)` | `PACE_STREAM=stderr` |
| `tqdm(it, mininterval=1)` | `PACE_MIN_INTERVAL=1` (a different default and meaning, see below) |
| `tqdm(it, disable=True)` | No argument. Skip the wrapper: `it if quiet else track(it)`. Off-rank-0 processes are silent already. |
| `tqdm(it, leave=False)` | No equivalent. The final line stays. |
| `bar.set_description(...)` | No equivalent. The description is fixed when the scope opens. Open a new scope. |
| `bar.set_postfix(loss=...)` | No equivalent. Log the value with `logger.info`. |
| `unit_scale`, `unit_divisor`, `bar_format`, `ncols`, `position`, `colour`, `ascii` | No equivalent. The line layout is fixed. The bar is 20 characters and the line is cut to the terminal width; ASCII is chosen from the stream's encoding. |
| `tqdm.auto`, `tqdm.pandas()`, `process_map` | No equivalent. |
| Nested bars with `position=` | Nest scopes: open one inside the other. |

Differences that bite:

- **Positional arguments differ.** `tqdm(iterable, desc, total, ...)` takes `desc`
  second and `total` third. `track(iterable, description, unit, total)` takes
  `unit` third. Use keywords. `track(it, "Train", 3000)` passes `3000` as the unit and
  raises `TypeError: unit must be a string such as 'step', got 3000; pass the total as
  total=...`.
- **The keyword is `description`, not `desc`.** `track(it, desc="Train")` raises
  `TypeError`, as does any other `tqdm` argument not in the table.
- **A bar is for people; a progress line is for a program.** Outside a terminal there
  is no bar; there is a text line every 15 seconds. Do not expect `tqdm`'s output to
  look like the old one.
- **The estimate is not tqdm's.** `tqdm` shows a time left from the first iteration;
  Pace says `estimating…` for the first minute, then shows an estimate that discounts
  one-off pauses. See the [FAQ](faq.md).
- **A `with` block reports failure.** If an exception leaves a `progress` block the scope
  is written as `fail`. `tqdm` has no such record.

**When a library takes the `tqdm` class.** Code you do not own may accept a `tqdm` class
(vLLM's `use_tqdm=`). There is no loop of yours to replace, so pass
`precisionai.pace.tqdm.PaceTqdm`, which reports what the library counts to Pace and draws
nothing; see [Libraries that take a `tqdm` class](frameworks.md#libraries-that-take-a-tqdm-class-vllm).
It also works as a drop-in for your own `tqdm(...)` calls
(`from precisionai.pace.tqdm import PaceTqdm as tqdm`) when you want to migrate without
touching them, with the positional order `tqdm` uses; `track` and `progress` remain the
recommended form.

If your code leaves the `tqdm` bar on while running in a container, remove it. Two
progress displays on stdout are noise, and `tqdm`'s carriage-return redraws fill a log file.

## From Loguru

| Loguru | Pace |
|---|---|
| `from loguru import logger` | `from precisionai.pace import logger` |
| `logger.info("x")`, `.debug`, `.success`, `.warning`, `.error`, `.critical` | The same methods |
| `logger.info("loaded {} items", n)` | `logger.info(f"loaded {n} items")` |
| `logger.exception("failed")` | `logger.exception("failed")` |
| `logger.opt(exception=True).error("failed")` | `logger.exception("failed")` inside the `except` |
| `logger.log("WARNING", msg)` | `logger.log("WARNING", msg)` |
| `logger.remove(); logger.add(sys.stderr, level="INFO")` | `logger.configure(stream="stderr", level="info")` |
| `logger.add("run.log", level="DEBUG")` | `logger.configure(stream=open("run.log", "a"), level="debug")` (one stream only) |
| `logger.level("INFO")` filtering | `PACE_LEVEL=info` |
| `@logger.catch` | No equivalent. Use `try`/`except` with `logger.exception`. |
| `logger.bind(...)`, `logger.contextualize(...)`, `extra` | No equivalent. Put the context in the message. |
| `logger.trace` | No equivalent. Use `debug`. |
| Custom `format=`, `serialize=True`, `rotation`, `retention`, `enqueue` | No equivalent. See the three differences at the top. |

Loguru's `{}` formatting is the one that breaks code: `logger.info("a {}", x)` raises
`TypeError` in Pace, because each method takes one argument.

## From the standard `logging` module

Library code can keep using `logging.getLogger(__name__)`. Do not rewrite it. Call
`capture_logging()` once in your entry point, and everything that propagates to the root
logger comes out in Pace's format.

| `logging` | Pace |
|---|---|
| `logging.basicConfig(level=logging.INFO)` | `capture_logging(level="INFO")` |
| `log = logging.getLogger(__name__)` in library code | Unchanged |
| `log.info("loaded %d items", n)` in library code | Unchanged (formatting is done by `logging`) |
| `log.info("loaded %d items", n)` in your own script | `logger.info(f"loaded {n} items")` |
| `log.exception("failed")` | Unchanged with `capture_logging()`, or `logger.exception("failed")` |
| `logging.getLogger("x").setLevel(...)` | Unchanged |
| `handler = logging.StreamHandler(...)` | Remove it. `capture_logging()` adds the one handler. |
| `logging.FileHandler("run.log")` | Redirect stdout, or `logger.configure(stream=...)`. |
| `Formatter("%(asctime)s ...")` | No equivalent. The format is fixed. |

```python
# before
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
log.info("Loading model")

# after
from precisionai.pace import capture_logging, logger

capture_logging(level="INFO")   # library logging goes through Pace
logger.info("Loading model")    # your own lines
```

Remove the `basicConfig` call. If it stays, every record is written twice. Details, the
case of libraries that print their own lines (Lightning, Transformers: `take_over=`), and warnings are in
[Standard logging](logging.md#standard-logging-pytorch-and-transformers).

## Checklist

1. Replace `print`, `tqdm` and Loguru calls with the equivalents above.
2. Call `capture_logging()` once if libraries log through `logging`, and delete
   `basicConfig`.
3. Count the work: a `track` or `progress` around the main loop. Mark the main loop
   `root=True`, or declare a [plan](progress.md#phases-and-a-plan) for a multi-stage job.
4. If the job resumes from a checkpoint, pass `completed=` (see
   [Resuming](progress.md#resuming-from-a-checkpoint)).
5. Run it once in a terminal and once with `python train.py | cat` to see both outputs.
