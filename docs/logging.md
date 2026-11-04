# Logging

`precisionai.pace.logger` is a ready-made logger object. Import it and call it; there
is nothing to set up.

```python
from precisionai.pace import logger

logger.info("Loading model")
```

Logging never raises into your code. A message that cannot be formatted or written
(a closed stream, a broken pipe) is dropped.

## Levels

| Method | Number | Colour on a terminal | Use it for |
|---|---|---|---|
| `logger.debug(msg)` | 10 | dim | Details you only want while debugging. |
| `logger.info(msg)` | 20 | none | What the job is doing. |
| `logger.success(msg)` | 25 | green | A step finished as intended. |
| `logger.warning(msg)` | 30 | yellow | Something unexpected that the job worked around. |
| `logger.error(msg)` | 40 | red | Something failed; the job may continue. |
| `logger.critical(msg)` | 50 | bold red | The job cannot continue. |
| `logger.exception(msg)` | 40 | red | An `error` with the traceback attached. |

`PROGRESS` is a seventh label, written only by the progress functions
(see [Progress](progress.md)). It is not filtered by the level setting, because a
scheduler depends on those lines.

Each method takes exactly one argument and calls `str()` on it:

```python
logger.info(f"Batch size is {batch_size}")  # use an f-string
logger.info({"epoch": 3})                  # INFO     {'epoch': 3}
```

There is no `%`-style or `{}`-style argument formatting. `logger.info("a %s", 1)`
raises `TypeError`. Format the string yourself.

To pick the level at run time, use `logger.log(level, message)`. `level` is a name
in any case or a number such as the one the standard `logging` module uses;
`logger.log(35, ...)` is written as `WARNING` and an unknown name as `INFO`.

### Exceptions

Call `logger.exception` inside an `except` block. It writes an `ERROR` line followed
by the traceback of the exception being handled:

```python
config = {"epochs": 3}

try:
    lr = config["lr"]
except KeyError:
    logger.exception("Config is missing a key")
```

```text
2026-11-03T17:15:10Z ERROR    Config is missing a key
Traceback (most recent call last):
  File "/tmp/ex/exc.py", line 6, in <module>
    lr = config["lr"]
         ~~~~~~^^^^^^
KeyError: 'lr'
```

Outside an `except` block it writes the message alone.

### Text that looks like a trailer

If a message contains `[pace`, Pace writes `[ pace` instead, so text from a log
message or a description cannot forge a progress record:

```python
logger.info("[pace x c=1 t=2 e=1 q=2 r=abc]")
```

```text
2026-11-03T17:04:14Z INFO     [ pace x c=1 t=2 e=1 q=2 r=abc]
```

## Configuring

You can run without configuring anything. When you do want to change something,
there are two ways, and they combine:

```python
from precisionai.pace import logger

logger.configure(level="debug", color="never", min_interval=5)
```

```bash
PACE_LEVEL=debug PACE_COLOR=never PACE_MIN_INTERVAL=5 python train.py
```

The environment variables let a scheduler or a job template change the behaviour
without touching your code. Every variable is listed in
[Configuration](configuration.md).

### Precedence

For every setting the order is:

1. a value passed to `logger.configure(...)`,
2. the `PACE_*` environment variable,
3. the built-in default.

```python
import os

from precisionai.pace import logger

logger.configure(level="debug")
os.environ["PACE_LEVEL"] = "error"
logger.debug("still shown")   # configure() wins over the environment

logger.reset()                # forget every explicit setting
logger.debug("hidden")        # now PACE_LEVEL=error applies
```

Details that matter:

- `configure` only changes the options you pass. An option you leave out keeps its
  current value, and there is no way to set one back to "unset" on its own. Use
  `logger.reset()` to drop all explicit settings at once.
- Environment variables are read every time they are needed, not once at import. A
  variable changed while the process runs takes effect on the next line, unless an
  explicit `configure` value overrides it.
- A bad value passed to `configure` raises `ValueError` (unknown level, a colour or
  bar mode other than `auto`, `always`, `never`, a negative interval). A bad value in
  an environment variable is ignored and the default applies.

### `configure` options

| Option | Values | Default |
|---|---|---|
| `level` | `"debug"`, `"info"`, `"success"`, `"warning"`, `"error"`, `"critical"`, or a number | `"info"` |
| `stream` | `"stdout"`, `"stderr"`, or any object with `write` and `flush` | `"stdout"` |
| `color` | `"auto"`, `"always"`, `"never"` | `"auto"` |
| `bar` | `"auto"`, `"always"`, `"never"` | `"auto"` |
| `min_interval` | seconds, 0 or more | `15` |
| `heartbeat` | seconds, 0 or more | `120` |
| `all_ranks` | `True` or `False` | `False` |
| `clock` | `precisionai.pace._env.Clock`, for tests | the real clocks |

`stream` as a name is looked up each time a line is written, so it follows
redirection such as pytest's capture or `contextlib.redirect_stdout`. An object is
used as is:

```python
with open("run.log", "a", encoding="utf-8") as handle:
    logger.configure(stream=handle)
    logger.info("to a file")
```

`min_interval`, `heartbeat`, `bar` and `all_ranks` affect progress lines; see
[Progress](progress.md#how-often-lines-are-written).

If you need an independent logger (for a test, or a second output), create
`Logger()` and pass it to a progress function as `logger=`. The module-level
`logger` is one such object.

## Standard `logging`, PyTorch and Transformers

Your own code uses `logger`. Libraries use the standard `logging` module, and their
lines would otherwise appear in a different format, on stderr, interleaved with
yours. Route them through Pace once at the start of the program:

```python
import logging

from precisionai.pace import capture_logging

capture_logging(level="INFO")

log = logging.getLogger("trainer.data")
log.info("Found %d shards", 12)
log.debug("not routed")
log.warning("Slow disk: %.1f MB/s", 41.7)
logging.getLogger("torch.distributed").error("NCCL timeout")
```

```text
2026-11-03T17:04:40Z INFO     Found 12 shards
2026-11-03T17:04:40Z WARNING  Slow disk: 41.7 MB/s
2026-11-03T17:04:40Z ERROR    NCCL timeout
```

What it does:

- It adds one handler to the **root** logger, so every logger that propagates to
  the root (the default) is routed. Calling it again does not add a second handler;
  it updates the level of the first.
- It lowers the root logger's level to `level` if it was higher, because records
  below the root level never reach a handler.
- Records are written with the level mapped to the nearest Pace level (a level 25
  record is `SUCCESS`). A traceback attached to the record is included. The logger
  name is not shown.
- It returns the handler. To undo it, `logging.getLogger().removeHandler(handler)`.
- It is opt-in. Importing Pace does not change how any other library logs.

Three things to know:

**Other root handlers still run.** `capture_logging` adds a handler; it does not
remove existing ones. If your program already called `logging.basicConfig(...)`, each
record is written twice, once by that handler (on stderr) and once by Pace (on
stdout):

```text
INFO:x:hello
2026-11-03T17:04:55Z INFO     hello
```

Remove the `basicConfig` call, or remove its handler, when you adopt Pace.

**Libraries that print their own log lines.** Some libraries attach a console handler to
their own logger, and some also set `propagate = False`. Their records then either appear
twice (once in their format, once in Pace's) or never reach the root handler at all. Hugging
Face Transformers does the second (checked with 5.22.1). Lightning does the first for its
`lightning` logger, and both for `lightning.pytorch` and `lightning.fabric` when no root handler
existed when it was imported (checked with 2.6.6). Without any change:

```text
INFO: GPU available: False, used: False
2026-11-03T23:34:29Z INFO     GPU available: False, used: False
```

Name such a library in `take_over` and Pace switches its own output off and lets its records
through to the root:

```python
import lightning.pytorch as pl
import transformers

from precisionai.pace import capture_logging

capture_logging("INFO", take_over=["lightning", "transformers"])
```

```text
2026-11-03T23:35:00Z INFO     GPU available: False, used: False
```

For each name it clears the console handlers (a `StreamHandler` that is not a
`FileHandler`) from that logger, from its parents (`lightning` for `lightning.pytorch`, which
would otherwise swallow the record on its way up) and from every logger beneath it that exists
at the time, and sets `propagate = True` on all of them. Handlers that write to a file or a
service are left alone. A bare string is one name; an empty name raises `ValueError` and
changes nothing.

- **Call it after importing the library.** Most libraries create their loggers and handlers
  at import, so a `capture_logging(take_over=...)` placed before the `import` finds nothing to
  take over. Calling it again later is safe and picks up loggers created since.
- The names are the logger names, which are the package names: `lightning` (or
  `pytorch_lightning` for the older package), `transformers`, `accelerate`.
- It is permanent for the process. Removed handlers are not put back.
- The library's own level still applies: a library that logs at `INFO` shows its `INFO`
  lines only if `capture_logging(level=...)` is `INFO` or lower.
- **vLLM is different.** Its engine runs in a separate `EngineCore` process that configures
  its own logging, so clearing the main process's `vllm` logger does not cover it. Use vLLM's
  own switch instead: set `VLLM_CONFIGURE_LOGGING=0` in the environment before vLLM is
  imported. vLLM then installs no handler, and its main-process records propagate to the root,
  where `capture_logging()` sees them.

Check any other library whose lines still appear in the old format; it is the same
cause, and `logging.getLogger("name").handlers` and `.propagate` show it.

**Python warnings.** `warnings.warn` goes to stderr, not to `logging`. Add
`logging.captureWarnings(True)` and they reach Pace as `WARNING` lines:

```text
2026-11-03T17:15:28Z WARNING  /tmp/ex/warn.py:8: UserWarning: tf32 is disabled
  warnings.warn("tf32 is disabled", UserWarning)
```

## Colour

Colour is used only on the level label, only when the stream is a terminal, and never
inside the `[pace ...]` trailer. `PACE_COLOR` (or `configure(color=...)`) decides:

| `PACE_COLOR` | Result |
|---|---|
| `auto` (default) | Colour on a terminal, unless `NO_COLOR` is set to a non-empty value. |
| `always` | Colour whatever the stream is, and whatever `NO_COLOR` says. |
| `never` | No colour. |

`NO_COLOR` follows the [no-color.org](https://no-color.org) convention and only
affects `auto`; an explicit `always` wins. Use `PACE_COLOR=never` when you pipe
output through something that does not strip escape sequences. A reader that parses
trailers removes ANSI sequences before matching, so colour never breaks it.

```text
$ PACE_COLOR=always python warn_demo.py | cat -v
2026-11-03T17:05:11Z ^[[33mWARNING^[[0m  careful
```

On Windows, escape sequences need a console that renders them (Windows Terminal
does). The legacy console host has not been tested; set `NO_COLOR=1` there if you see
raw escape codes.

## Encodings and the ASCII fallback

Pace writes the readable part with Unicode glyphs (`·`, `…`, `━`, `─`) only when the
stream's encoding is UTF-8. On any other encoding it falls back to ASCII glyphs:

| | UTF-8 stream | other encodings |
|---|---|---|
| separator | `·` | `|` |
| ellipsis | `…` | `...` |
| bar | `━` and `─` | `#` and `-` |

A character the stream cannot encode is written as `?`; it never raises
`UnicodeEncodeError`. This is what a job writing to a legacy Windows code page
(cp1252) or an `ascii` locale sees:

```text
$ PYTHONIOENCODING=ascii python demo.py
2026-11-03T17:05:14Z INFO     caf? ? na?ve ???
2026-11-03T17:05:14Z PROGRESS Donn?es  0/3 ?chantillon  [pace donn-es start t=3 u=%C3%A9chantillon n=Donn%C3%A9es r=0fe38035]
```

The trailer is always plain ASCII. Non-ASCII text in it (the label and the unit) is
percent-encoded as UTF-8, so a reader gets the original text back whatever the
stream's encoding is.

Notice that the scope id (`donn-es`) is derived from the description by replacing
every character outside `A-Z a-z 0-9 . _ -` with `-`. A description with few ASCII
characters gives a poor id (a description of only non-ASCII characters becomes `scope`); pass `scope="..."` to
`track`/`progress` when you need a stable, readable one.

To force UTF-8 on a stream that supports it, set `PYTHONIOENCODING=utf-8` or
`PYTHONUTF8=1` for the job. Pace does not change the stream's encoding.

## Distributed runs and ranks

Pace reads the process rank from the first of `RANK` (torchrun, Accelerate),
`OMPI_COMM_WORLD_RANK` (Open MPI), `PMI_RANK` (MPICH, Intel MPI), `SLURM_PROCID` and
`LOCAL_RANK` that holds an integer; with none it is rank 0. The global ranks come first
because `LOCAL_RANK` counts from 0 on every node.

- **Progress** is written only by rank 0. On other ranks `track`, `progress`,
  `timed`, `phase` and `plan` still run your code and yield your items, and write
  nothing. This keeps one set of records per job instead of one per GPU.
- **Log lines** (`logger.info` and the rest) are written by every rank. Pace does not
  filter them. If you want one copy, test the rank yourself or raise the level on the
  other ranks (`PACE_LEVEL=warning`).
- Set `PACE_ALL_RANKS=1` (or `configure(all_ranks=True)`) to write progress from
  every rank. Scope ids are not qualified by rank, so the records from different
  ranks are indistinguishable if you merge them into one file. A reader should read
  one process's output at a time.

```text
$ RANK=1 python rank.py
2026-11-03T17:05:34Z INFO     hello from a worker

$ RANK=1 PACE_ALL_RANKS=1 python rank.py
2026-11-03T17:05:32Z INFO     hello from a worker
2026-11-03T17:05:32Z PROGRESS Train  0/3 step  [pace train start t=3 u=step n=Train r=71cffa57]
2026-11-03T17:05:32Z PROGRESS Train  3/3  done in 0s  [pace train end c=3 r=71cffa57]
```

The rank is read when a progress scope is created, so set the variables before the
first `track` or `progress` call (launchers such as `torchrun` do this for you).
