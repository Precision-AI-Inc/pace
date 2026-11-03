# Design

Status: **reviewed**, see [Decisions](#decisions). Thresholds marked *initial* are
starting points to be tuned against real runs, not commitments.

## Goals

- One logger that Precision AI repositories and outside users can adopt, so that
  logging and progress reporting look and behave the same from project to project.
- Progress reporting for the shapes ML work actually takes: bounded loops, manual
  counters, timed steps, multi-phase pipelines, PyTorch Lightning and Hugging Face
  training.
- Output that a scheduler can read to estimate when a job will finish. OrchAI is the
  first such consumer: it reads the job's logs, estimates completion, and uses the
  estimate to show users when their task will be done and to place queued work.

## Non-goals

- Pace is not a metrics or experiment tracker. Loss curves and evaluation scores
  belong in MLflow or similar. Pace reports *how far along* the work is.
- A consumer never infers progress from arbitrary text, `tqdm` bars or framework
  output. Only the line format below is read.
- An estimate is never a deadline. Nothing in Pace stops or slows a job.

## Package

| | |
|---|---|
| Distribution | `precisionai-pace` (PyPI) |
| Import | `precisionai.pace` |
| Python | 3.8 and later |
| Runtime dependencies | none in the core |
| Extras (planned) | `[lightning]`, `[transformers]`, `[all]` |

Everything lives in this one repository and package: the logger, the progress APIs,
the line parser, the estimator and the framework adapters. The core imports only the standard library. It is installed into users' training
images and into other packages (such as data-transfer tooling), so it must never
add to them. Framework adapters live in their own modules and import their framework
only there; their extras inherit that framework's own Python minimum.

## Public API (proposed)

```python
from precisionai.pace import logger, track, progress, timed, phase

logger.info("Loading model")
logger.success("Model ready")
logger.warning("Falling back to cached metadata")

for batch in track(loader, description="Inference", unit="batch"):
    infer(batch)

with progress(total=len(requests), description="Evaluation", unit="request") as run:
    for future in as_completed(futures):
        consume(future.result())
        run.advance()

@timed(description="Load model")
def load_model(): ...

with phase("Preprocess", weight=0.05):
    ...
```

```python
from precisionai.pace.lightning import PaceCallback      # [lightning]
from precisionai.pace.transformers import PaceCallback   # [transformers]; TRL uses the same callbacks
```

- **Levels**: `debug`, `info`, `success`, `warning`, `error`, `critical`, plus
  `progress` records, which only the progress APIs emit.
- **Standard `logging`**: `precisionai.pace.capture_logging()` routes the standard
  library's records (and so PyTorch's and Transformers') through the same output.
  Opt-in, so importing Pace never changes another library's logging on its own.
- **Configuration**: keyword arguments, with environment-variable overrides so a
  scheduler can adjust behaviour without code changes: `PACE_LEVEL`, `PACE_STREAM`
  (`stdout` default, or `stderr`), `PACE_COLOR` (`auto`/`always`/`never`),
  `PACE_MIN_INTERVAL`, `PACE_HEARTBEAT`.
- **Distributed training**: rank is read from `RANK`, then `LOCAL_RANK`, then
  `SLURM_PROCID`. Only rank 0 emits progress by default, and the adapters report
  the global step, so there is one record per scope, not one per GPU.

### Output stream

Records go to **stdout** by default. Progress and status are the job's normal
output, not diagnostics, and stdout is where people look for them. A program whose
stdout is data piped to another program can set `PACE_STREAM=stderr`. A consumer
reads both streams and removes duplicates using the per-scope sequence number
(`q=`), so a user who mirrors output to both streams is not double-counted.

Every line is flushed as it is written. Python block-buffers stdout when it is not a
terminal, which would otherwise hold progress back for minutes in a container.

### Characters

- The trailer is plain ASCII, so it reads the same on every platform and terminal.
- The readable part uses Unicode (`·`, bar glyphs) only when the stream's encoding
  is UTF-8, and plain ASCII otherwise (for example a Windows console on a legacy
  code page).
- Writing never raises because of encoding: an unencodable character is replaced,
  never allowed to crash the program.
- Colour is used only on a terminal, never inside the trailer, and is turned off by
  `NO_COLOR` or `PACE_COLOR=never`.

### Emission rate

In a terminal, a progress bar redraws in place. When output is not a terminal (a
container, a log file), Pace writes a new line only when all of these hold:

- at least `PACE_MIN_INTERVAL` seconds since the scope's last line (*initial*: 15 s), and
- at least 1% more of the total done, **or** `PACE_HEARTBEAT` seconds have passed
  since the last line (*initial*: 120 s).

`start`, `end` and `fail` records are always written immediately. The limit is set by
time, never by step count, so a run with a million steps writes no more than one
with a thousand. For example, a 10-hour run reaches each 1% about every 6 minutes,
so it writes about 100 progress lines plus a heartbeat every 2 minutes in between:
roughly 300 lines, or 30 an hour. The heartbeat is what lets a consumer tell a slow
job from a stalled one. The values will be tuned against real runs.

## Line format, version 1

One line serves both readers. The readable part comes first; a short trailer at the
end carries the fields a consumer parses.

```text
2026-11-03T17:36:42Z PROGRESS Train  36.8%  1840/5000 step · 2.4 step/s · ~21m57s  [pace:1 s=train root=1 k=upd c=1840 t=5000 u=step e=763.1 q=57]
2026-11-03T17:23:59Z INFO     Loading model Qwen2.5-7B
```

Only `PROGRESS` lines carry a trailer. Everything before the trailer is for people
and may change between releases; consumers must not parse it.

### Grammar

```text
trailer  = "[pace:" major " " field *( " " field ) "]"   ; at the very end of the line
major    = 1*DIGIT                                       ; "1" for this version
field    = key "=" value
key      = 1*( %x61-7A / DIGIT )                          ; lower-case letters and digits
value    = "?" / 1*( ALPHA / DIGIT / "." / "_" / "-" / ":" / "," / "%" HEXDIG HEXDIG )
                                                         ; "?" only as the value of t
```

A consumer matches the trailer only at the end of the line, after removing ANSI
escape sequences and trailing whitespace. Any character outside the value alphabet
is percent-encoded (UTF-8, `%XX`). A line longer than 4096 bytes, an unknown major
version, a missing required field, or a value that fails to parse is ignored.
A raw (still encoded) value may be at most 768 characters, enough for a 64-character
label of 4-byte characters; `ph` may be up to 2560, enough for 32 phases. The writer
shortens labels and units to 64 characters and replaces control characters with spaces,
so it never produces a line the parser rejects.

### Fields

| Key | Required | Meaning |
|---|---|---|
| `s` | always | Scope id, `[A-Za-z0-9._-]{1,64}`. Unique within one process run. |
| `k` | always | Event: `start`, `upd`, `end`, `fail`, `plan`. |
| `q` | always | Sequence number within the scope, from 1. Used for de-duplication and ordering. |
| `e` | always | Seconds since the scope started, from a monotonic clock. Rates use this, never wall-clock time. |
| `c` | start, upd, end | Completed amount, a non-negative number. |
| `t` | start, upd, end | Total, a positive number, or `?` when unknown. |
| `u` | start | Unit (`step`, `batch`, `sample`, `byte`, `file`, ...), at most 64 characters after decoding, no control characters. |
| `p` | optional | Parent scope id. A nested scope is never added to its parent's remaining time. |
| `root` | optional | `1` marks the scope that stands for the whole job. At most one per process. |
| `res` | optional | `1` on a `start` that resumes earlier work, so `c` begins above zero. |
| `ph` | `plan` only | Remaining phases, as scope ids, with weights, e.g. `ph=prep:0.05,train:0.9,eval:0.05`. |
| `n` | optional, `start` only | Short label, at most 64 characters after decoding, no control characters. |
| `run` | optional, `start` only | Id of the process run, `[a-z0-9]{1,16}`, the same for every scope in one process. A `start` with the run id already seen for that scope is a duplicate (the line was read from both streams); a different run id is a restart. |

Additions within version 1 are new optional keys; consumers ignore keys they do not
know. Removing or redefining a key requires `pace:2`.

### Why the timestamp is UTC

The leading timestamp is ISO-8601 UTC so that a line means the same thing in any
container, whatever its timezone. Rates come from `e=`, never from the timestamp.
Converting times for display (for example to Alberta time or Cairo time) is the
consumer's job.

## Forecasting rules

A pure, dependency-free reference implementation ships in the package
(`precisionai.pace.forecast`) next to the parser (`precisionai.pace.protocol`). The
terminal's "~21m left" and a scheduler's estimate then come from the same code.

### States

| State | Meaning |
|---|---|
| Unknown | No total, and no estimate supplied by the user. |
| Bootstrap | Only a user-supplied expected runtime is known. Shown as a guess. |
| Learning | Progress is arriving but there is not yet enough evidence. |
| Live | Enough evidence for an estimate. |
| Stale | A Live scope has stopped reporting for longer than expected. |
| Complete / Failed | The job has ended. Actual runtime replaces every estimate. |

Once a scope is Live, a user-supplied estimate is no longer used. A Stale scope
keeps its last estimate, marked stale, and never falls back to the user's guess.

### Estimate

```text
remaining_time = max(total - completed, 0) / robust_rate
```

- `robust_rate`: an exponentially weighted average of interval rates, where each
  interval is the work done between two consecutive records divided by the change
  in `e`. Intervals far outside the recent median are discounted (stalls, retries,
  evaluation pauses), but a sustained slowdown is not hidden.
- **Learning → Live** (*initial*): at least 60 s observed **and** at least 3 update
  intervals **and** at least 1% of the total done.
- **Stale** (*initial*): no record for `max(3 × median interval, 2 × heartbeat)`.
- A `start` for a scope already seen begins a new segment; rates are never averaged
  across a restart. With `res=1`, the remaining work starts from the resumed `c`.
- A changed `t` replaces the total. If the rate history is still valid it is kept.

### Several scopes

- A `root=1` scope is the job's estimate when present.
- Otherwise: phases in a `plan` that run one after another add up; scopes running
  at the same time take the longest remaining time. Nested scopes (`p=`) are never
  added on top of their parent.
- Without a root or a complete plan, the estimate is labelled as covering the
  current phase only.

## Framework adapters

- **Lightning**: a callback reporting `trainer.global_step` against
  `trainer.estimated_stepping_batches`, as one `root` scope, with validation as
  a child scope. Restores resume state from the checkpoint's global step.
- **Transformers / TRL**: a `TrainerCallback` reporting `state.global_step` against
  `state.max_steps`, rank 0 only.
- **tqdm**: a migration guide rather than a shim. `track()` takes the same core
  arguments (`total`, `desc` as `description`, `unit`).

## Security

- Log lines are untrusted input to any consumer. The parser bounds line length,
  field count, value length and numeric range, and never evaluates anything.
- Pace never writes environment variables, configuration or credentials on its own.
  Labels are short and are not repeated on every update.
- A malformed or unsupported line is ignored and never raises in the consumer.

## Testing

- Conformance fixtures live in `tests/fixtures/protocol/`: input lines with their
  expected parse results, including malformed, oversized, ANSI-coloured, split and
  duplicated cases. Consumers run the same fixtures against their integration.
- Every release is tested on Python 3.8 through 3.13.

## Decisions

Reviewed 2026-11-03.

1. **Trailer syntax**: `[pace:1 ...]` at the end of the line, plain ASCII. The
   readable part falls back to ASCII on non-UTF-8 streams and never crashes on
   encoding (see [Characters](#characters)).
2. **Default stream**: stdout. Progress is normal output, not a diagnostic.
3. **Thresholds**: the *initial* values stand, to be tuned against real runs. The
   limit is time-based, so step count does not increase volume.
4. **One package**: the parser, estimator, logger and adapters all ship in
   `precisionai-pace`; there is no separate repository for any of them.
