# FAQ

## Why does Pace write to stdout?

Progress and status are a job's normal output, not diagnostics, and stdout is where
people look for them in a container log. Every line is flushed as it is written,
because Python block-buffers stdout when it is not a terminal and would otherwise hold
progress back for minutes.

If your program's stdout is data that another program reads, set
`PACE_STREAM=stderr` (or `logger.configure(stream="stderr")`). A reader that reads both
streams removes the duplicates by scope and sequence number; see
[Reading Pace output](reading-output.md#two-streams-duplicate-lines).

## Why does the estimate say "estimating…" for the first minute?

An estimate from the first few steps is typically wrong: warm-up, data-loader start-up and
compilation make early steps unrepresentative. Pace shows a time only once it has at
least 60 seconds of observations, three measured intervals and 1% of the work done.
While each interval is faster than the one before (by more than 1.25 times), the rate
restarts from the latest interval and no time is shown yet. The first interval that is
not that much faster ends this for good, so a slow load followed by a compile that is
only a little faster still counts the compile.

In a container, intervals are measured between the lines that are written, one every
15 seconds at most, so the first time usually appears on about the fifth record, a little after the
first minute (`e=60.6` in the [example](getting-started.md#what-you-see-in-a-container-or-a-log-file)).
On a terminal the bar updates ten times a second, so the same wait is about 60 seconds.

A long first step (loading a model, say) delays it further. A scheduler can show the
job's expected runtime as a guess until then; see `expected_seconds` in
[Reading Pace output](reading-output.md#from-records-to-an-estimate-jobforecast).

## Does it slow my training loop?

Not measurably for a loop of GPU steps. On a development machine (WSL2, Python 3.12) a
`track` loop costs about 2 microseconds per iteration over a bare `for`, and
`run.advance()` about the same, when nothing is written. A line is written only
when the rate limits allow it: at most one per scope per 15 seconds outside a terminal,
or about ten bar redraws a second on one.

Wrap the loop over **batches**, not a tight loop over millions of cheap items; at
that scale 2 microseconds per item is measurable. For a loop like that, advance in blocks
(`run.advance(1000)` after every 1000 items).

Pace never raises into your code, and nothing it does can stop or slow your job on
purpose: an estimate is not a deadline.

## Does it work on Windows?

The package is plain standard-library Python and has no platform-specific code. It was run
on Python 3.13 for Windows with output redirected to a pipe: log lines and progress
lines were written, and on a legacy cp1252 stream the readable part fell back to ASCII
(`|`, `...`) as described in [Encodings](logging.md#encodings-and-the-ascii-fallback).
The in-place bar and colour have not been tested in a Windows terminal; colour needs one
that renders ANSI escape sequences. Set `NO_COLOR=1` if you see raw escape codes.

The tests run in CI on Linux, Python 3.11 through 3.13.

## What about multiple processes and ranks?

- **Distributed training.** Rank 0 writes progress; other ranks write none and run your
  code unchanged. Log lines are written by every rank. See
  [Distributed runs](logging.md#distributed-runs-and-ranks). Set `PACE_ALL_RANKS=1` to
  write progress from every rank.
- **Worker processes** (a `DataLoader` with `num_workers`, `multiprocessing`). Do not
  call `track` or `progress` inside them. A worker inherits its parent's rank variables, so
  a worker of rank 0 writes progress too, under its own run id. Report from the main
  process.
- **Several jobs in one log.** Records carry no process id, only a per-run `run` id. A
  reader should read one process's output at a time: two processes writing progress to
  one stream at the same time look like restarts to a reader, and `JobForecast` follows
  only the run whose lines it saw begin last.

## What if I don't set a total?

Then there is nothing to estimate. The line shows a count and a rate and `no estimate`,
the trailer has `t=?`, and a reader has no time to show. The heartbeat still tells it the
job is alive.

Ways to fix it: pass `total=` (`track` takes `len()` automatically when it exists), call
`run.set_total(n)` when you learn the total, or declare a
[plan](progress.md#phases-and-a-plan) so that a scheduler can estimate from the other
phases. See [When you do not know the total](progress.md#when-you-do-not-know-the-total).

## How do I turn the bar off?

`PACE_BAR=never` in the environment, or `logger.configure(bar="never")`. Progress is then
written as normal lines with the trailer, even on a terminal. That is also what you want
when a pseudo-terminal is attached to a job whose logs something else reads. See
[The terminal bar](progress.md#the-terminal-bar).

To remove progress from a run entirely, do not wrap the loop; there is no "disable"
switch. Processes that are not rank 0 write no progress automatically.

## How do I keep secrets out of logs?

Pace never writes environment variables, configuration or credentials on its own, and a
label is written once, on `start`, not repeated on every update. Everything else it writes
is what you pass it. So:

- Do not put a token, a password or a signed URL in a log message, a `description`, a
  `unit`, a `phase` name or a `scope` id. They all end up in the log, and descriptions
  and units also in the machine-readable trailer.
- `logger.exception` appends the full traceback, and the text of an exception can contain
  a URL or a header. Check what your libraries put in their exception messages.
- `capture_logging()` forwards other libraries' log lines. Some libraries print request
  details at `DEBUG`; keep the level at `INFO` (the default) in production and be
  careful with `PACE_LEVEL=debug`.
- Assume that anyone who can read the job's logs can read every line. A scheduler that
  parses them stores at least the trailers.

Text that looks like a trailer in a message or a description is rewritten (`[pace`
becomes `[ pace`), so user text cannot forge a progress record. That protects the
reader; it does not hide anything.

## Why is a repeated description numbered (`train-2`)?

Scope ids are unique within a process run, so a second scope with the same description gets a
numeric suffix. Pass `scope="..."` to choose the id yourself, for example
`scope=f"train-{epoch}"`.

## Does the estimate allow for evaluation pauses?

Partly. An interval much slower or faster than the recent median (a checkpoint, an
evaluation run) is discounted so one pause does not move the estimate. A sustained
change of speed is followed. A pause that recurs, such as evaluation every N steps, is
discounted every time, so the estimate is optimistic by the total of the pauses still to
come. This is a known limitation, noted in the [specification](design.md#estimate).
