# Pace specification

This document specifies Pace (`precisionai-pace`): the logger and progress API, the
line format that carries progress records, and the rules that turn those records into a
completion estimate. The `[pace:1 ...]` trailer format is a public contract versioned by
its major number: within a major version the format only gains optional keys, and any
other change needs a new major version.

The thresholds in this document are defaults that may be tuned in later releases;
override the output ones with `PACE_MIN_INTERVAL` and `PACE_HEARTBEAT` (or
`logger.configure(...)`) and the estimator ones with `ForecastConfig`.

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
| Python | 3.11 and later |
| Runtime dependencies | none in the core |
| Extras | `[lightning]`, `[transformers]` (includes `accelerate`), `[all]` |

Everything lives in this one repository and package: the logger, the progress APIs,
the line parser, the estimator and the framework adapters. The core imports only the standard library. It is installed into users' training
images and into other packages (such as data-transfer tooling), so it must never
add to them. Framework adapters live in their own modules and import their framework
only there; their extras inherit that framework's own Python minimum.

## Public API

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
  `progress` records, which only the progress APIs emit. `PACE_LEVEL` filters log
  records; progress records are never filtered by it, because a scheduler depends on
  them.
- **Full list**: besides the names above, the public API includes `plan`,
  `capture_logging`, `parse_line`, `LineSplitter`, `ScopeForecast` and `JobForecast`;
  `precisionai.pace.__all__` is the complete list.
- **Standard `logging`**: `precisionai.pace.capture_logging()` routes the standard
  library's records (and so PyTorch's and Transformers') through the same output.
  Opt-in, so importing Pace never changes another library's logging on its own.
  `take_over=` names libraries (`"lightning"`, `"transformers"`) whose own console handlers
  are removed and whose loggers are made to propagate, so each line is written once, in Pace's
  format. It acts on the named logger, its parents and the children that exist at the call,
  keeps file and service handlers, and is permanent for the process.
- **Configuration**: keyword arguments, with environment-variable overrides so a
  scheduler can adjust behaviour without code changes: `PACE_LEVEL`, `PACE_STREAM`
  (`stdout` default, or `stderr`), `PACE_COLOR` (`auto`/`always`/`never`), `PACE_BAR` (`auto`/`always`/`never`),
  `PACE_MIN_INTERVAL`, `PACE_HEARTBEAT`, `PACE_ALL_RANKS`. An explicit
  `logger.configure(...)` value beats the environment variable, which beats the default.
- **Distributed training**: rank is read from `RANK`, then `OMPI_COMM_WORLD_RANK`
  (Open MPI), `PMI_RANK` (MPICH, Intel MPI), `SLURM_PROCID` and `LOCAL_RANK`, the global
  ranks first. Only rank 0 emits progress by default, and the adapters report
  the global step, so there is one record per scope, not one per GPU.

### Progress scopes

- `track()` cannot tell a `break` from an exception raised in the loop body: Python
  delivers both to the generator as the same `GeneratorExit`. It therefore writes `end`
  with the count reached for either; `fail` is written only when the iterable itself
  raises. Use `progress()` when a crash in the loop body must be reported as `fail`.
- A heartbeat line is written when progress is reported (`advance()`, `update()`), not
  on a timer. A process that hangs writes nothing, and that silence is what marks it
  Stale.

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

- at least `PACE_MIN_INTERVAL` seconds since the scope's last line (default 15 s), and
- at least 1% more of the total done, **or** `PACE_HEARTBEAT` seconds have passed
  since the last line (default 120 s). With an unknown total, only the heartbeat applies.

On a terminal (`PACE_BAR=auto` and a TTY), progress redraws one bar line in place,
without a trailer, and only `end`, `fail` and `plan` print full lines with trailers.
A job run under a pseudo-terminal whose output a scheduler still reads should set
`PACE_BAR=never`, which writes normal trailer lines even on a TTY.

`start`, `end` and `fail` records are always written immediately. The limit is set by
time, never by step count, so a run with a million steps writes no more than one
with a thousand. For example, a 10-hour run reaches each 1% about every 6 minutes,
so it writes about 100 progress lines plus a heartbeat every 2 minutes in between:
roughly 300 lines, or 30 an hour. The heartbeat is what lets a consumer tell a slow
job from a stalled one.

## Line format, version 1

One line serves both readers. The readable part comes first; a short trailer at the
end carries the fields a consumer parses.

```text
2026-11-03T17:36:42Z PROGRESS Train  36.8%  1840/5000 step · 2.4 step/s · ~21m57s  [pace:1 s=train root=1 k=upd c=1840 t=5000 e=763.1 q=57]
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
label of 4-byte characters; `ph` may be up to 2815, enough for 32 phases with the longest ids and weights. The writer
shortens labels and units to 64 characters and replaces control characters with spaces,
so it never produces a line the parser rejects.

### Fields

| Key | Required | Meaning |
|---|---|---|
| `s` | always | Scope id, `[A-Za-z0-9._-]{1,64}`. Unique within one process run. |
| `k` | always | Event: `start`, `upd`, `end`, `fail`, `plan`. |
| `q` | always | Sequence number within the scope, from 1. Used for de-duplication and ordering. |
| `e` | always | Seconds since the scope started, from a monotonic clock. Rates use this, never wall-clock time. |
| `c` | start, upd, end; fail when known | Completed amount, a non-negative number. |
| `t` | start, upd, end; fail when known | Total, a positive number, or `?` when unknown. |
| `u` | start | Unit (`step`, `batch`, `sample`, `byte`, `file`, ...), at most 64 characters after decoding, no control characters. |
| `p` | optional | Parent scope id. A nested scope is never added to its parent's remaining time. Written on every record of the scope, so a reader that starts mid-log still learns the hierarchy. |
| `root` | optional | `1` marks the scope that stands for the whole job. At most one per process. Written on every record of the scope. |
| `res` | optional | `1` on a `start` that resumes earlier work, so `c` begins above zero. |
| `ph` | `plan` only | Remaining phases, as scope ids, with weights, e.g. `ph=prep:0.05,train:0.9,eval:0.05`. |
| `n` | optional, `start` only | Short label, at most 64 characters after decoding, no control characters. |
| `run` | optional, `start` only | Id of the process run, `[a-z0-9]{1,16}`, the same for every scope in one process. A `start` with the run id already seen for that scope is a duplicate (the line was read from both streams); a different run id is a restart. |

Keys belong to particular events, and a reader rejects a record that carries one
on the wrong event: `run`, `n`, `res` and `u` only on `start`, `ph` only on `plan`,
and no `c` or `t` on `plan`. A writer keeps every line within 4096 bytes by
shortening the readable text before the trailer, never the trailer. It never writes a
number above 1e15, the parser's limit: a larger total is reported as unknown and a
larger count is capped.

Text that looks like a trailer (`[pace:`) inside a log message or description is
rewritten to `[ pace:`, so user text cannot forge a record.

Additions within version 1 are new optional keys; consumers ignore keys they do not
know. Removing or redefining a key requires `pace:2`.

### Timestamp

The leading timestamp is ISO-8601 UTC, so a line means the same thing in any container
whatever its timezone. Rates come from `e=`, never from the timestamp. Converting times
for display is the consumer's job.

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
- **Learning → Live** (default): at least 60 s observed **and** at least 3 update
  intervals **and** at least 1% of the total done.
- **Stale** (default): no record for `max(3 × median interval, 2 × heartbeat)`.
- A `start` for a scope from a new run (a different `run` id, or after `q=1` when there
  is no run id) begins a new segment; rates are never averaged across a restart. The
  segment's rate is measured from the `c` on its `start`, so a resumed scope that
  starts at its restored count measures only the work left. `res=1` itself is
  informational: it tells a reader the scope resumed, and changes no arithmetic.
- A changed `t` replaces the total. If the rate history is still valid it is kept.
- While Live, the remaining time counts down between records by the time since the
  latest one. A Stale scope keeps its last estimate unchanged.
- Known limitation: a pause that recurs (evaluation every N steps) is discounted from
  the rate like a one-off pause, so the estimate is optimistic by the total of the
  pauses still to come.

### Rate smoothing

`ForecastConfig` holds every threshold of the estimator; the values here are the
defaults.

- An interval with no progress (`c` unchanged, a heartbeat) is not measured; the next
  interval with progress spans back to the last one that had it. Intervals with
  `Δe <= 0` are skipped.
- The first intervals are averaged plainly (`1/n`); after that the weight of a new
  interval is 0.3.
- An interval rate more than 3x above or below the median of the last 9 accepted rates
  is an outlier and is left out (a pause for evaluation or a checkpoint). Outlier
  tests start once 3 rates have been accepted. Three outliers in a row on the same
  side are a real change in speed: the average restarts from them.

### Several scopes

- A `root=1` scope is the job's estimate when present.
- Otherwise: phases in a `plan` that run one after another add up; scopes running
  at the same time take the longest remaining time. Nested scopes (`p=`) are never
  added on top of their parent.
- Without a root or a complete plan, the estimate is labelled as covering the
  current phase only.
- **Completion and failure.** With a root, the root alone decides: the job is complete
  or failed when the root is. Without one, a failed top-level scope fails the job, but
  the job is never called complete unless a plan lists every phase and all of them
  (and every top-level scope) have ended: between two unplanned phases every scope
  seen so far has ended, yet more work may follow. A scheduler should treat process
  exit as the real end.
- Concurrent scopes: if any contributing scope is Stale, the combined estimate is Stale.
- A running scope with no estimate of its own takes one from its children. A counted
  parent (such as `Epochs` of 3) estimates the time one unit takes times the units
  still to come, less the time already spent in the running unit, and never less than
  the running child's remaining time. Once a unit has finished, the parent's own
  records give the time per unit, however many children a unit holds (a training and a
  validation loop per epoch); before that, the running unit counts as the time it has
  taken so far plus the running child's remaining time. A parent of more than 100 units
  gets no estimate from its children until it has a pace of its own: it need not write
  every advance, and its units are often steps rather than children, so a child's
  duration says nothing about a unit's. A `phase()` or a parent with no
  total cannot know how many children are still to come, so it takes only the running
  child's estimate and the job estimate is marked partial (a lower bound). Such a
  borrowed estimate is Stale when a running child it borrows from is, or, between two
  children, when neither the parent nor any scope under it has reported within the
  parent's own Stale limit. It then stops counting down at the moment its source last
  reported.
- A running scope's elapsed time counts on from its latest record, by the time
  received since it. A `phase()` writes only when it starts and ends, so this is what
  gives a running phase its elapsed time, which a plan uses to time the phases not yet
  started.

### Plans

- `plan()` writes its record under a scope of its own (`plan`, or the `scope=` given).
  Repeated calls under the same name share that scope, so their `q` orders them. The
  record lists the scope ids the phases will get, including the `-2` suffix for a
  repeated name.
- A plan replaces an earlier plan of the same scope only if its `q` is higher, so a
  plan read late from the other stream never overrides a newer one.
- A plan's finished phases count 0, its active phase its own estimate, and each phase
  not yet started `weight x s`, where `s` is the (elapsed + remaining) seconds of the
  finished and active phases divided by their weights. If a phase in the plan has no
  usable estimate, or nothing has been measured to derive `s`, the plan is not used
  and the top-level scopes decide.
- Without a root, only a finished plan proves the job is done (see Completion and
  failure above).

## Framework adapters

- **Lightning**: a callback reporting `trainer.global_step` against
  `trainer.estimated_stepping_batches`, as one `root` scope, with validation as
  a child scope. Restores resume state from the checkpoint's global step.
- **Transformers / TRL**: a `TrainerCallback` reporting `state.global_step` against
  `state.max_steps`, rank 0 only.
- **tqdm**: a migration guide for code you own (`track()` takes the same core
  arguments: `total`, `desc` as `description`, `unit`), and `PaceTqdm`, a class that stands in
  for `tqdm` where a library takes the bar class from its caller (vLLM's `use_tqdm=`). It
  subclasses nothing and imports nothing: it implements the calls such libraries make
  (`update`, `n`, `total`, `refresh`, `close`, `format_dict`, ...) on one Pace scope that starts
  when the object is created and ends when it is closed, and ignores everything about how a bar
  looks. Subclassing the real `tqdm` was rejected: its constructor draws, starts a monitor
  thread and writes a newline on `close`, and its disabled mode skips the attributes callers
  read, so silencing it takes more code than the stand-in and ties Pace to `tqdm` internals.

## Security

- Log lines are untrusted input to any consumer. The parser bounds line length,
  field count, value length and numeric range, and never evaluates anything.
- Pace never writes environment variables, configuration or credentials on its own.
  Labels are short and are not repeated on every update.
- A malformed or unsupported line is ignored and never raises in the consumer.

## Testing

- Conformance fixtures live in `tests/fixtures/protocol/`: input lines with their
  expected parse results, including malformed, oversized and ANSI-coloured lines.
  They cover parsing one line at a time; splitting byte chunks into lines and removing
  duplicates are tested in Python only. The fixtures are not shipped in the package;
  readers in other languages take them from the repository. Consumers run the same fixtures against their integration.
- Every release is tested on Python 3.11 through 3.13.

## Not in this version

- No record is written when the process is killed. A `SIGKILL` cannot be caught, and
  the scheduler already knows when the process exits, so the consumer treats process
  exit as the end. This is deliberate, not an omission.
- Estimates are point estimates. `JobEstimate` carries a state, a source and `partial`,
  but no range or confidence. Adding one would not change the line format.
