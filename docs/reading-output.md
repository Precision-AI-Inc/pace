# Reading Pace output

This page is for the program that reads a job's logs: a scheduler, a dashboard, a
wrapper script. If you are adding Pace to a job, you do not need it.

The package ships the reading side, so a Python consumer and the terminal's own
"time left" run the same code:

```text
bytes ──> LineSplitter ──> parse_line ──> Record ──> JobForecast ──> JobEstimate
```

Only the `[pace:1 ...]` trailer at the end of a `PROGRESS` line is a contract. The
text before it is for people and changes between releases. Never parse it, and never infer
progress from other output such as a `tqdm` bar. The grammar, the fields and the
forecasting rules are specified in [Design](design.md#line-format-version-1); this page
shows how to use them.

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
        print(f"q={record.seq:<2} {record.event.value:<5} {estimate.state.value:<8} remaining={left:<4} source={estimate.source}")

for line in splitter.flush():
    pass  # a last line with no newline, if any
print("process exited with", proc.wait())
```

```text
q=1  start learning remaining=-    source=None
q=2  upd   learning remaining=-    source=None
...
q=12 upd   learning remaining=-    source=None
q=13 upd   live     remaining=21s  source=root
q=14 upd   live     remaining=16s  source=root
q=15 upd   live     remaining=11s  source=root
q=16 upd   live     remaining=6s   source=root
q=17 upd   live     remaining=1s   source=root
q=18 end   complete remaining=0s   source=None
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

record = parse_line("2026-11-03T17:36:42Z PROGRESS Train  36.8%  1840/5000 step · 2.4 step/s · ~21m57s  "
                    "[pace:1 s=train root=1 k=upd c=1840 t=5000 e=763.1 q=57]")
```

```text
Record(scope='train', event=<Event.UPDATE: 'upd'>, seq=57, elapsed=763.1, completed=1840.0,
       total=5000.0, unit='step', parent=None, root=True, resumed=False, phases=(), label=None, run=None)
```

`parse_line` never raises. A line with no trailer, or a trailer that breaks any rule (a
malformed value, a missing required field, an unknown major version, a line over 4096
bytes, a value out of range) returns `None`. Treat every line as untrusted input: a job's
logs can contain anything.

The `Record` fields:

| Field | Trailer key | Meaning |
|---|---|---|
| `scope` | `s` | Scope id, unique within one process run. |
| `event` | `k` | `Event.START`, `UPDATE`, `END`, `FAIL` or `PLAN`. |
| `seq` | `q` | Sequence number within the scope, from 1. |
| `elapsed` | `e` | Seconds since the scope started, on the job's monotonic clock. |
| `completed` | `c` | Amount done, or `None` (a `fail` may omit it, and a `plan` has none). |
| `total` | `t` | Total, or `None` when written as `?`. |
| `unit` | `u` | Unit, on `start` only. |
| `parent` | `p` | Parent scope id for a nested scope. |
| `root` | `root` | `True` when the scope stands for the whole job. |
| `resumed` | `res` | `True` on a `start` that resumes earlier work. |
| `phases` | `ph` | On `plan`, a tuple of `(scope id, weight)`. |
| `label` | `n` | Short human label, on `start` only. |
| `run` | `run` | Id of the process run, on `start` only. |

Compute rates from `elapsed`, never from the timestamp at the front of the line.
`elapsed` is on a monotonic clock, so it is unaffected by clock changes. The timestamp
is UTC and meant for display.

Keys a reader does not know are ignored, so a later minor version stays readable.
Removing or redefining a key would be `pace:2`, which `parse_line` rejects.

## Two streams, duplicate lines

Pace writes to stdout by default, or to stderr with `PACE_STREAM=stderr`. A reader that
captures both (or a user who mirrors the output to both) sees every line twice. Records
carry what a reader needs to ignore the copies:

- **`q`** is a sequence number within each scope. A record whose `q` is not above the
  highest already seen for that scope is a duplicate.
- **`run`** appears on every `start` and is the same for every scope in one process run.
  A `start` with a run id already seen for the scope is the same line read twice. A
  different run id means the scope restarted: begin a new segment and do not average
  rates across the restart. Without run ids, a `start` after the scope had reported
  beyond `q=1` is taken as a restart.
- A `plan` replaces an earlier one only if its `q` is higher, so a plan read late from the
  other stream never overrides a newer one.

`JobForecast` does all three. Feed it both streams without de-duplicating yourself:

```python
log = [
    "[pace:1 s=train root=1 k=start c=0 t=100 u=step e=0 q=1 run=a1b2c3d4]",
    "[pace:1 s=train root=1 k=upd c=10 t=100 e=10 q=2]",
    "[pace:1 s=train root=1 k=upd c=20 t=100 e=20 q=3]",
]
job = JobForecast()
for second, line in enumerate(log):
    for _ in range(2):  # once from stdout, once from stderr
        job.observe(parse_line(line), received_at=float(second))
print(job.snapshot(now=2.0).scopes["train"].completed)   # 20.0, not doubled

# The same job restarted: a new run id on start begins a new segment.
job.observe(parse_line("[pace:1 s=train root=1 k=start c=0 t=100 u=step e=0 q=1 run=ffff0000]"), 3.0)
print(job.snapshot(3.0).scopes["train"].completed)       # 0.0
```

If you write your own reader, in any language, keep for each scope the highest `q` and the
`run` of its latest `start`, and apply the same rules.

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
| `partial` | `True` when the number covers only the scopes seen so far (no root and no complete plan). |
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
`ForecastConfig` (`min_observed_seconds`, `min_intervals`, `min_fraction`) and were
chosen as starting points; see [Design](design.md#forecasting-rules).

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
no estimate of its own, such as a `phase()` that only counts 0 of 1, takes the longest
estimate among its children, so a phase wrapping a counted loop still has one
(see [Phases and a plan](progress.md#phases-and-a-plan)).

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
for i in range(5):
    job.observe(parse_line(f"[pace:1 s=t root=1 k={'start' if i == 0 else 'upd'} c={i * 20} t=100 e={i * 20} q={i + 1}]"), i * 20.0)
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
the line format as test data. It holds 132 cases, one JSON object per line: 38 lines
that parse and 94 that must be rejected (oversize, bad escapes, wrong version, wrong
number syntax, trailer not at the end, ANSI colour, CRLF and so on). The Python parser
passes every case in `tests/test_protocol_reading.py`.

Each case has three keys:

| Key | Meaning |
|---|---|
| `name` | A unique name for the case. |
| `line` | The text of one log line, without its line ending. |
| `record` | `null` when a reader must ignore the line; otherwise the expected record. |

A record uses the keys `scope`, `event`, `seq`, `elapsed`, `completed`, `total`, `unit`,
`parent`, `root`, `resumed`, `phases` (a list of `[scope id, weight]` pairs), `label` and
`run`. A key that is absent has its default: `completed`, `total`, `unit`, `parent`,
`label` and `run` are none, `root` and `resumed` are false, `phases` is empty. Compare
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
