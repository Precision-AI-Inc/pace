# Reading Pace output

This page is for the program that reads a job's logs: a scheduler, a dashboard, a
wrapper script. If you are adding Pace to a job, you do not need it.

The package ships the reading side, so a Python consumer and the terminal's own
"time left" run the same code:

```text
bytes ──> LineSplitter ──> parse_line ──> Record ──> JobForecast ──> JobEstimate
```

Only the `[pace ...]` trailer at the end of a `PROGRESS` or `STEP` line is a contract.
The text before it is for people and changes between releases. Never parse it, and never
infer progress from other output such as a `tqdm` bar. The grammar, the fields and the
forecasting rules are specified in the [specification](design.md#line-format); this page
shows how to use them. The `[pace:1 ...]` lines that Pace 0.0.x wrote are not read: they
are ordinary text to this reader.

## A complete reader

This reads a child process's output as it arrives and prints the estimate after every
record. `train.py` is any program that uses Pace (here a 400-step `track` loop with
`root=True`, 0.2 s per step).

```python
import os
import subprocess
import sys
import time

from precisionai.pace import JobForecast, LineSplitter, parse_line

proc = subprocess.Popen(
    [sys.executable, "train.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,  # one pipe for both streams
    env={**os.environ, "PACE_MIN_INTERVAL": "5"},
)
splitter = LineSplitter()
job = JobForecast()

while True:
    chunk = os.read(proc.stdout.fileno(), 65536)  # whatever has arrived, split anywhere
    if not chunk:
        break
    for line in splitter.feed(chunk):
        record = parse_line(line)
        if record is None:
            continue  # an ordinary log line
        now = time.monotonic()
        job.observe(record, received_at=now)
        estimate = job.snapshot(now)
        left = "-" if estimate.remaining_seconds is None else f"{estimate.remaining_seconds:.0f}s"
        print(f"{record.event.value:<5} c={record.completed:<5g} {estimate.state.value:<8} remaining={left:<4} source={estimate.source}")

for line in splitter.flush():
    pass  # a last line with no newline, if any
print("process exited with", proc.wait())
```

```text
start c=0     learning remaining=-    source=None
upd   c=25    learning remaining=-    source=None
...
upd   c=275   learning remaining=-    source=None
upd   c=300   live     remaining=20s  source=root
upd   c=325   live     remaining=15s  source=root
upd   c=350   live     remaining=10s  source=root
upd   c=375   live     remaining=5s   source=root
upd   c=400   live     remaining=0s   source=root
end   c=400   complete remaining=0s   source=None
process exited with 0
```

The estimate is `learning` for the first minute, then `live`. The sections below
explain each step.

## Splitting bytes into lines: `LineSplitter`

A pipe or a file delivers bytes in arbitrary chunks. `LineSplitter` turns them into
text lines:

```python
splitter = LineSplitter()
splitter.feed(b"first\nsecond ")   # ['first']
splitter.feed(b"line\r\n")         # ['second line']
splitter.flush()                   # [] : nothing left over
```

- The chunk may end anywhere: inside a multi-byte character, or between `\r` and `\n`.
  The result does not depend on where the chunks fall.
- `\n` and `\r\n` end a line. A bare `\r` does not, so the frames of an in-place terminal
  bar (separated by `\r`) arrive as one long piece of text. If you read a pseudo-terminal,
  have the job set `PACE_BAR=never`.
- Lines are decoded as UTF-8, with undecodable bytes replaced.
- Memory is bounded. A line longer than 4096 bytes cannot hold a valid trailer, so it is
  dropped instead of buffered, however hostile the stream is.
- **`reset()`** drops the partial line. Call it when you detect a gap: you reconnected to a
  stream, a log file was truncated or rotated, or you skipped bytes. Without it the end of
  the old partial line would be glued to the start of the next chunk.
- **`flush()`** returns the last line when the stream ends without a trailing newline.

```python
splitter.feed(b"half a li")
splitter.reset()                # the gap: forget "half a li"
splitter.feed(b"ne\n")          # ['ne']
```

## One line to one record: `parse_line`

```python
from precisionai.pace import parse_line

record = parse_line("2026-11-04T02:25:16Z PROGRESS Generate   66/128  51.6%  estimating…  "
                    "[pace generate c=66 t=128 e=186.7 q=2 r=1b8b7650 root]")
```

```text
Record(scope='generate', event=<Event.UPDATE: 'upd'>, seq=2, elapsed=186.7, completed=66.0,
       total=128.0, unit=None, parent=None, root=True, resumed=False, phases=(), label=None, run='1b8b7650')
```

`parse_line` never raises. A line with no trailer, or a trailer that breaks any rule (a
malformed value, a missing required field, a line over 4096
bytes, a value out of range) returns `None`. Treat every line as untrusted input: a job's
logs can contain anything.

The `Record` fields:

| Field | In the trailer | Meaning |
|---|---|---|
| `scope` | the head, `scope<parent` | Scope id, unique within one process run. |
| `event` | the event word | `Event.START`, `UPDATE` (no word), `END`, `FAIL`, `STEP` or `PLAN`. A `STEP` is a start and an end at once: a short `@timed` call or `phase()`, or a child scope that ended before its first line was due. |
| `seq` | `q` | Sequence number within the scope, on an update or plan; `None` on the others. |
| `elapsed` | `e` | Seconds since the scope started, on the job's monotonic clock; 0 when left out. |
| `completed` | `c` | Amount done (0 when left out, 1 on a `step` with neither `c` nor `t`), or `None` (a `fail` may not know it, and a `plan` has none). |
| `total` | `t` | Total, or `None` when written as `?`. An `end` or `step` with `c` but no `t` has its count as total; a `step` with neither has 1. |
| `unit` | `u` | Unit, on `start`, and on a `step` that carries its counts (a child scope's). |
| `parent` | after `<` in the head | Parent scope id for a nested scope. |
| `root` | `root` | `True` when the scope stands for the whole job. |
| `resumed` | `res` | `True` on a `start` that resumes earlier work. |
| `phases` | `ph` | On `plan`, a tuple of `(scope id, weight)`. |
| `label` | `n` | Short human label, on `start` and `step`. |
| `run` | `r` | Id of the process run, on every record. |

Compute rates from `elapsed`, never from the timestamp at the front of the line.
`elapsed` is on a monotonic clock, so it is unaffected by clock changes. The timestamp
is UTC and meant for display.

Keys and words a reader does not know are ignored, so a later minor version stays
readable. The trailer has no version number, so the format only ever grows. One case
where an older reader loses a line: from 0.1.1 a short child scope is one `step` that
carries `c`, `t` and `u`, which a 0.1.0 reader rejects as keys on the wrong event. It
then misses that one child; its parent and the job estimate are unaffected.

## Two streams, duplicate lines

Pace writes to stdout by default, or to stderr with `PACE_STREAM=stderr`. A reader that
captures both (or a user who mirrors the output to both) sees every line twice. Each
stream must be read in the order it was written, but one may be read well behind the
other, even to its end first. Every record carries what a reader needs to
ignore the copies:

- **`r`**, the run id, is on every record and the same for every scope in one process run.
- Within one run, a scope's records come in a fixed order: the `start`, then the updates
  by **`q`**, then the `end`, `fail` or `step`. A record that does not come after the
  latest one taken for its scope and run is a copy, or was read late, and is ignored. So
  a `start` read after its scope's updates, from the other stream, is a late copy and not
  a restart, even for a reader that joined mid-run and never saw the first copy.
- A record from a run id not seen before begins a new segment: the process restarted.
  Rates are never averaged across a restart. A record from a run that a restart left
  behind is a late copy.
- The job follows its latest run: once a new run id appears, the scopes and plan of the
  run before are dropped from the job estimate and from `scopes`, even those the new run
  never writes again (a root under another name, or not reached yet). Until the new
  run's root appears, its own plan and top-level scopes give the estimate.
- One limitation: the reader does not know the order of runs it has not seen. If it first
  sees a later run and only then lines of an earlier run it never saw (it attached to the
  second stream of a restarted job late), it takes the earlier run for a restart and
  follows that one. Start reading both streams from their beginning.
- A `plan` replaces an earlier one only if its `q` is higher or it comes from a newer run,
  so a plan read late from the other stream never overrides a newer one.

`JobForecast` does all of this: feed it both streams as they arrive, without
de-duplicating yourself. `observe` returns `False` for a record it ignored.

```python
log = [
    "[pace train start t=100 u=step r=a1b2c3d4 root]",
    "[pace train c=10 t=100 e=10 q=2 r=a1b2c3d4 root]",
    "[pace train c=20 t=100 e=20 q=3 r=a1b2c3d4 root]",
]
job = JobForecast()
for second, line in enumerate(log):
    job.observe(parse_line(line), received_at=float(second))   # stdout
for line in log:
    job.observe(parse_line(line), received_at=3.0)             # stderr, read later
print(job.snapshot(now=3.0).scopes["train"].completed)   # 20.0: the late start is a copy

# The same job restarted: a new run id begins a new segment.
job.observe(parse_line("[pace train start t=100 u=step r=ffff0000 root]"), 4.0)
print(job.snapshot(4.0).scopes["train"].completed)       # 0.0
```

If you write your own reader, in any language, keep for each scope its current run id,
the few run ids a restart left behind, and the position of the latest record taken (0
for a `start`, `q` for an update, last for an `end`, `fail` or `step`), and apply the same
rules. For a job estimate, also keep the job's current run id and the ones a restart left
behind, and drop every scope and plan when a new run id appears.

## From records to an estimate: `JobForecast`

```python
from precisionai.pace import ForecastConfig, JobForecast

job = JobForecast(expected_seconds=None, config=ForecastConfig())
job.observe(record, received_at=time.monotonic())
estimate = job.snapshot(time.monotonic())
```

`received_at` and the `now` of `snapshot` are seconds on **your** clock. Any monotonic
clock works if both use the same one. They are not read from the log, because the
reader needs to know when it received a line to tell a stalled job from a slow one.

`JobForecast` is one job, that is, one process run. Keep one per job.

### The estimate

`snapshot` returns a `JobEstimate`:

| Field | Meaning |
|---|---|
| `state` | A `ForecastState`, below. |
| `remaining_seconds` | Seconds until the job finishes, or `None` when there is no estimate. |
| `source` | Where the number comes from: `"root"`, `"plan"`, `"concurrent"`, `"user"`, or `None`. |
| `partial` | `True` when the number covers only the scopes seen so far (no root and no complete plan), or rests on a running `phase()` timed only by the loop inside it. |
| `scopes` | A dict of each scope's own `Forecast` (`state`, `remaining_seconds`, `rate`, `completed`, `total`, `unit`). |

The states:

| `state` | `remaining_seconds` | Meaning |
|---|---|---|
| `unknown` | `None` | No total, and no `expected_seconds` was supplied. |
| `bootstrap` | the guess, counting down | Only the user's `expected_seconds` is known (`source="user"`). Present it as a guess. |
| `learning` | `None` | Progress is arriving but there is not enough evidence yet. |
| `live` | a number | Enough evidence. The number counts down between records. |
| `stale` | the last number, frozen | A live scope stopped reporting for longer than expected. |
| `complete` | `0.0` | The job ended. Replace every estimate with the actual runtime. |
| `failed` | `None` | The job failed. |

A scope goes from `learning` to `live` after at least 60 seconds of observations, three
measured intervals and 1% of its total done. Until then the estimate is `learning`, which
is why the first minute of any job says `estimating…`. These thresholds are in
`ForecastConfig` (`min_observed_seconds`, `min_intervals`, `min_fraction`); they are
defaults that may be tuned in later releases. See the [specification](design.md#forecasting-rules).

`expected_seconds` is a runtime the user supplied, for example from the job spec. It is
used only until some scope is live, and never again afterwards, so the guess cannot
override a real measurement.

How the number is chosen, in order:

1. If a scope is marked `root`, the root's estimate (`source="root"`). The root alone
   decides completion and failure.
2. Else, the latest `plan`: finished phases count zero, the running phase uses its own
   estimate, and phases not yet started use `weight x seconds-per-weight`
   (`source="plan"`).
3. Else, the longest remaining time among the top-level scopes running at once
   (`source="concurrent"`, `partial=True`). Nested scopes, those with a `parent`, are never
   added on top of their parent.

A plan is only usable when every running phase has an estimate. A running scope with
no estimate of its own takes one from its children: a counted parent times its units
(from its own records once one has finished, so a unit may hold several children, such
as a training and a validation loop) and adds the units still to come, while a `phase()`
takes only the running child's estimate and the job estimate is then marked `partial=True`
(see [Phases and a plan](progress.md#phases-and-a-plan)). A counted parent of more than
100 units (a root over training steps, say) never takes an estimate from its children: it
has one only once its own records make it Live, the same as its own line shows.

Always show `partial=True` estimates as a lower bound, for example "at least 10 min
left".

### Staleness

A live scope becomes **stale** when no record has been received for
`max(3 x median gap between records, 2 x heartbeat_seconds)`. With the defaults,
at least 240 seconds. A stale scope keeps its last estimate, marked stale; it never falls back
to the user's guess. If any scope that contributes to the estimate is stale, the combined
estimate is stale.

Staleness is computed against `now`, so call `snapshot` on a timer as well as after each
record. Without new lines nothing else will change the state:

```python
job = JobForecast(config=ForecastConfig(heartbeat_seconds=120))
job.observe(parse_line("[pace t start t=100 r=a1 root]"), 0.0)
for i in range(1, 5):
    job.observe(parse_line(f"[pace t c={i * 20} t=100 e={i * 20} q={i + 1} r=a1 root]"), i * 20.0)
for now in (80.0, 200.0, 500.0):
    estimate = job.snapshot(now)
    print(now, estimate.state.value, estimate.remaining_seconds, estimate.source)
```

```text
80.0 live 20.0 root
200.0 live 0.0 root
500.0 stale 20.0 root
```

At 200 s the estimate has counted down to zero and is still `live`: the job is overdue, not
finished. Set `ForecastConfig.heartbeat_seconds` to the `PACE_HEARTBEAT` the jobs run with
(default 120); a reader that assumes a shorter heartbeat than the writer uses will call
healthy jobs stale.

### Process exit is the real end

Only a `root` scope's `end` or `fail` is a definite end of the job. Without a root, the job is
reported complete only when a plan lists every phase and all of them ended, because between
two unplanned phases every scope seen so far has ended, yet more work may follow. A
`fail` of any top-level scope fails the job. So a scheduler should treat the **process
exit** as the real end of a job, and use the estimate to say how long it has left, not to
decide that it is done.

Two consequences:

- A process that exits without a final `end` or `fail` (killed, out of memory) leaves the
  estimate `live` or `stale`. Combine the estimate with the exit code.
- A `track` loop that crashes in its body writes `end`, not `fail`, so a scope can
  look finished when it crashed (see
  [the `track()` caveat](progress.md#track-ends-the-scope-on-break-and-on-exceptions-in-the-body)).
  Another reason to trust the exit code first.

## Conformance fixtures for other languages

`tests/fixtures/protocol/cases.jsonl` in the repository is the shared specification of
the line format as test data. It holds 184 cases, one JSON object per line: 58 lines
that parse and 126 that must be rejected (oversize, bad escapes, an old `[pace:1 ...]` marker, wrong
number syntax, a key on the wrong event, trailer not at the end, ANSI colour, CRLF and so
on). The Python parser
passes every case in `tests/test_protocol_reading.py`.

Each case has three keys:

| Key | Meaning |
|---|---|
| `name` | A unique name for the case. |
| `line` | The text of one log line, without its line ending. |
| `record` | `null` when a reader must ignore the line; otherwise the expected record. |

A record uses the keys `scope`, `event` (`upd` for an update), `seq`, `elapsed`,
`completed`, `total`, `unit`, `parent`, `root`, `resumed`, `phases` (a list of
`[scope id, weight]` pairs), `label` and `run`. A key that is absent has its default:
`seq`, `completed`, `total`, `unit`, `parent`, `label` and `run` are none, `elapsed` is 0,
`root` and `resumed` are false, `phases` is empty. Compare
numbers as numbers (`1` equals `1.0`).

To check a parser in another language, load the file and, for each case, run your parser on
`line`. It must return nothing when `record` is `null`, and otherwise a record equal to
`record`. The whole logic is a few lines:

```python
import json

for case in map(json.loads, open("tests/fixtures/protocol/cases.jsonl", encoding="utf-8")):
    parsed = my_parse_line(case["line"])
    if case["record"] is None:
        assert parsed is None, case["name"]
    else:
        assert parsed == expected_record(case["record"]), case["name"]
```

The fixtures live in the repository, not in the wheel or the sdist (the build excludes
`tests/`), so a consumer in another language gets them from the repository. They cover
`parse_line` only. Splitting a byte stream into lines, de-duplication and the
estimator are tested in Python (`tests/test_protocol_reading.py`, `tests/test_forecast.py`)
and have no shared fixtures: a reader in another language needs its own tests for those.
