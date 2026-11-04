# Pace specification

This document specifies Pace (`precisionai-pace`): the logger and progress API, the
line format that carries progress records, and the rules that turn those records into a
completion estimate. The `[pace ...]` trailer format is a public contract. It carries no
version number, so it may only grow: new optional keys and words, which every reader
ignores until it knows them. Removing or redefining anything is not allowed.

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
reads both streams and removes duplicates using the run id on every record and the
order of a scope's records (see [Duplicates and restarts](#duplicates-and-restarts)), so a
user who mirrors output to both streams is not double-counted.

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

`start`, `end` and `fail` records are always written immediately, except that a one-off
step or a child scope holds back its `start` (see [One-off steps](#one-off-steps) and
[Short child scopes](#short-child-scopes)). The limit is set by
time, never by step count, so a run with a million steps writes no more than one
with a thousand. For example, a 10-hour run reaches each 1% about every 6 minutes,
so it writes about 100 progress lines plus a heartbeat every 2 minutes in between:
roughly 300 lines, or 30 an hour. The heartbeat is what lets a consumer tell a slow
job from a stalled one.

## Line format

One line serves both readers. The readable part comes first; a short trailer at the
end carries the fields a consumer parses.

```text
2026-11-04T02:22:10Z PROGRESS Generate    0/128 request  [pace generate start t=128 u=request n=Generate r=1b8b7650 root]
2026-11-04T02:22:22Z STEP     ├ Start Ollama  done in 12s  [pace start-ollama<generate step e=12 n=Start%20Ollama r=1b8b7650]
2026-11-04T02:25:16Z PROGRESS Generate   66/128  51.6%  estimating…  [pace generate c=66 t=128 e=186.7 q=2 r=1b8b7650 root]
2026-11-04T02:25:27Z PROGRESS Generate  128/128  done in 3m17s · 0.354 request/s  [pace generate end c=128 e=197.4 r=1b8b7650 root]
2026-11-03T23:59:12Z INFO     Loading model Qwen2.5-7B
```

Only `PROGRESS` and `STEP` lines carry a trailer. Everything before the trailer is for
people and may change between releases; consumers must not parse it. Pace 0.0.x wrote an
earlier, versioned trailer (`[pace:1 ...]`); this reader takes it for ordinary text, and
an 0.0.x reader does not read this format.

### Grammar

```text
trailer  = "[pace " head *( " " item ) "]"              ; at the very end of the line
head     = scope [ "<" scope ]                           ; the scope id, then its parent's
item     = word / field
word     = 1*( %x61-7A / DIGIT )                         ; an event or a flag, see below
field    = key "=" value
key      = 1*( %x61-7A / DIGIT )                          ; lower-case letters and digits
value    = "?" / 1*( ALPHA / DIGIT / "." / "_" / "-" / ":" / "," / "%" HEXDIG HEXDIG )
                                                         ; "?" only as the value of t
```

Items may come in any order; the writer uses the order of the fields table below. A
consumer matches the trailer only at the end of the line, after removing ANSI escape
sequences and trailing whitespace. Any character outside the value alphabet is
percent-encoded (UTF-8, `%XX`). A line longer than 4096 bytes, more than 32 items (the head included), a repeated key or word, a missing required field, a key or
word on the wrong event, or a value that fails to parse is ignored. A raw (still
encoded) value may be at most 768 characters, enough for a 64-character label of 4-byte
characters; `ph` may be up to 2815, enough for 32 phases with the longest ids and
weights. The writer shortens labels and units to 64 characters and replaces control
characters with spaces, so it never produces a line the parser rejects.

### Events and flags

The event is a bare word; an update has none. At most one event per trailer.

| Word | Meaning |
|---|---|
| (none) | An update: progress since the start. |
| `start` | The scope started. Comes first among its run's records for the scope. |
| `end` | The scope finished. Comes last. |
| `fail` | The scope failed. Comes last. |
| `step` | A scope that started and finished before its first line was due: a `start` and an `end` at once. Without `c` and `t` it is a one-off step of one unit (`c=1 t=1`); a child scope's step carries its counts and unit. |
| `plan` | The phases still to come, with their weights. |
| `root` | Flag: the scope stands for the whole job. At most one per process. Written on every record of the scope, so a reader that starts mid-log still learns it. |
| `res` | Flag, `start` only: the scope resumes earlier work, so `c` begins above zero. |

A consumer ignores a word it does not know.

### Fields

| Key | On | Meaning |
|---|---|---|
| `c` | start, update, end, fail, step | Completed amount, a non-negative number. Left out when 0. On a `fail`, present only with `t`; a `fail` with neither does not know how far it got. A `step` with neither `c` nor `t` means `c=1 t=1`. |
| `t` | start, update (required); end, fail, step | Total, a positive number, or `?` when unknown. An `end` or `step` leaves it out when it equals `c`. |
| `u` | start, step | Unit (`step`, `batch`, `sample`, `byte`, `file`, ...), at most 64 characters after decoding, no control characters. |
| `e` | any | Seconds since the scope started, from a monotonic clock. Left out when 0. Rates use this, never wall-clock time. A `start` written late (see [One-off steps](#one-off-steps)) carries the time it was held back. |
| `q` | update, plan (required) | Sequence number within the scope and run, from 2 for an update (the start counts as 1) and from 1 for a plan. Orders the updates; a start comes first and an end, fail or step last, so they need none. |
| `ph` | plan | Remaining phases, as scope ids, with weights, e.g. `ph=prep:0.05,train:0.9,eval:0.05`. |
| `n` | start, step | Short label, at most 64 characters after decoding, no control characters. |
| `r` | any (required) | Id of the process run, `[a-z0-9]{1,16}`, the same for every scope in one process. |

The scope id in the head is `[A-Za-z0-9._-]{1,64}`, unique within one process run. A
nested scope's head is `scope<parent`, written on every record so a reader that starts
mid-log still learns the hierarchy; a nested scope is never added to its parent's
remaining time.

A writer keeps every line within 4096 bytes by shortening the readable text before the
trailer, never the trailer. It never writes a number above 1e15, the parser's limit: a
larger total is reported as unknown and a larger count is capped.

Text that could start a trailer (`[pace`) inside a log message or description is
rewritten to `[ pace`, so user text cannot forge a record.

With no version number, every change must be additive: new optional keys and words,
which consumers ignore until they know them. A key or word is never removed or given a
new meaning.

### Duplicates and restarts

A reader that captures both streams, or a job that mirrors its output to both, sees
every line twice. The run id on every record makes that safe, provided each stream is
read in the order it was written (one stream may be read well behind the other):

- Within one run, a scope's records are ordered: the `start`, then the updates by `q`,
  then the `end`, `fail` or `step`. A record that does not come after the latest one
  taken for that scope and run is a copy or was read late, and is ignored. A `start`
  read after the scope's updates, from the other stream, is therefore a late copy, not
  a restart.
- A record with a run id not seen before for the scope begins a new segment: the
  process restarted. It need not be a `start`, so a reader that missed the restarted
  run's `start` still notices.
- A record from a run that a restart replaced is a late copy, and is ignored.
- Known limitation: a reader that first sees a later run and only then lines of an
  earlier run it never saw takes the earlier run for a restart and follows it.
- A `plan` replaces an earlier plan of the same scope only if its `q` is higher, or it
  comes from a newer run.

### One-off steps

A `phase()` or a `@timed` call is one unit of work that often takes less time than
`PACE_MIN_INTERVAL`. When output is not a terminal and the interval is above 0, such a
scope holds back its `start`:

- If it ends before the interval has passed, it writes one `step` line, labelled `STEP`,
  with its parent, elapsed time, label and run id.
- If it is still running when the interval has passed, its `start` is written with the
  next line the logger writes (any scope's line, a log line) or the next progress check
  of any scope (with `e` the time held back), and its `end` follows as usual. Pace starts
  no threads, so this needs other activity: a long step with nothing else reporting, such
  as a lone download, shows nothing until it ends. Report its bytes with `progress()` to
  make it visible while it runs. A forked child never writes its parent's held-back line.
- Before any line of a scope nested in it, its `start` is written first, so a reader
  always meets a parent before its children and the text indents under it.
- If it fails, it writes its `start` and then `fail`, so a failure is never folded away.

On a terminal (`PACE_BAR`), and with `PACE_MIN_INTERVAL=0`, nothing is held back: a
`start` and an `end` are written as for any scope. A reader treats a `step` as a `start`
and an `end` at once: a finished phase with its elapsed time, which a plan uses to time
the phases not yet started.

### Short child scopes

A counted scope opened inside another (a validation loop inside training, one batch of
work inside a request) is often over in less than `PACE_MIN_INTERVAL`, and a start line
plus an end line says nothing the end line alone does not. Such a child holds back its
`start` under the same rules as a one-off step, with these differences:

- If it ends before the interval has passed, it writes one `step` line that carries its
  counts and unit as well (`c`, `t` as on an `end`, `u`), labelled `PROGRESS`, since it
  reports a counted scope like the end line it replaces:

  ```text
  2026-11-04T15:34:05Z PROGRESS ├ Validation  1/1 batch  done in 0s  [pace validation-263<train step c=1 u=batch e=0.1 n=Validation r=0a0dfc37]
  ```

- If it is still running when its line comes due, its `start` is written as it was when
  the scope started (no `e`, its starting `c`), so a reader times its first units from
  the real start. When the start was released by another line (a log line, another
  scope's line, a line of its own child), an update with its count so far follows at
  once, so a reader knows how long it has been running, and its next update is still
  due as if the start had been written on time. A long child therefore writes every
  record it would have written without holding back, plus at most one update.
- Its updates are rate-limited from when it started, as if its start had been written.
- A `root` scope, a scope with no parent and a resumed scope (whose `start` carries
  `res`) never hold back their start. Neither does any scope on a terminal or with
  `PACE_MIN_INTERVAL=0`, nor on a rank that does not write progress.
- A short child inside a short child folds only the inner one: the inner `step` needs
  its parent's `start` first.

A `step` with `c`, `t` or `u` is the only form a 0.1.0 reader rejects (those keys
belonged only to other events then); it ignores the line and misses that child, which
never stands for the job. A new event word would be ignored the same way, and a `start`
with a new "done" flag would be read by a 0.1.0 reader as a scope that never ends, so
extending `step` is the change an older reader handles best.

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
- A record for a scope from a new run (a run id not seen before) begins a new segment; rates
  are never averaged across a restart. A `step` is a `start` and an `end` at once. The
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
  started. A phase that holds back its start (a one-off step) is unknown to a reader
  until its `start` or `step` line arrives, at most `PACE_MIN_INTERVAL` later, so a plan
  counts it as not started until then; the `step` line carries its elapsed time. A child
  scope that holds back its start is likewise unknown until its first line, and a
  child's `step` is a finished child like any other for timing its parent.

### Plans

- `plan()` writes its record under a scope of its own (`plan`, or the `scope=` given).
  Repeated calls under the same name share that scope, so their `q` orders them. The
  record lists the scope ids the phases will get, including the `-2` suffix for a
  repeated name.
- A plan replaces an earlier plan of the same scope only if its `q` is higher or it
  comes from a newer run, so a plan read late from the other stream never overrides a
  newer one.
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

- Conformance fixtures live in `tests/fixtures/protocol/`: input lines with their expected parse results, including malformed, oversized and
  ANSI-coloured lines.
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
