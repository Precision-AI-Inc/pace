# Configuration

Every setting can be given in code with `logger.configure(...)` or in the environment
with a `PACE_*` variable. The order is always **explicit `configure` value, then
environment variable, then default**. See [Logging](logging.md#configuring) for the
details and examples.

## Every variable

| Environment variable | `configure(...)` | Values | Default | Effect |
|---|---|---|---|---|
| `PACE_LEVEL` | `level=` | `debug`, `info`, `success`, `warning`, `error`, `critical` (any case); `configure` also takes a number | `info` | Lowest level written. Progress lines are never filtered by it. |
| `PACE_STREAM` | `stream=` | `stdout`, `stderr`; `configure` also takes any object with `write` and `flush` | `stdout` | Where lines go. Any other value of the variable means `stdout`. |
| `PACE_COLOR` | `color=` | `auto`, `always`, `never` | `auto` | Colour of the level label. `auto` is colour on a terminal unless `NO_COLOR` is set. |
| `NO_COLOR` | none | any non-empty value | unset | Turns colour off in `auto` mode. An explicit `always` still wins. |
| `PACE_BAR` | `bar=` | `auto`, `always`, `never` | `auto` | The in-place progress bar. `auto` is a bar on a terminal and normal lines otherwise. |
| `PACE_MIN_INTERVAL` | `min_interval=` | seconds, 0 or more | `15` | Least time between two progress lines of one scope (outside the terminal bar). |
| `PACE_HEARTBEAT` | `heartbeat=` | seconds, 0 or more | `120` | A progress line is written after this long even if the work moved less than 1%. |
| `PACE_ALL_RANKS` | `all_ranks=` | `1`, `true`, `yes`, `on` (any case) mean on; anything else is off | off | Write progress from every rank instead of rank 0 only. |

Rules that apply to all of them:

- Values are read case-insensitively with surrounding whitespace removed. An empty
  variable is the same as an unset one.
- A value that is not valid is ignored and the default applies; the environment never
  raises. `configure` is stricter: an invalid level, mode or interval raises `ValueError`.
- Environment variables are read each time a line is written, so a change made while the
  process runs applies from the next line.

## Variables Pace reads but does not define

| Variable | Used for |
|---|---|
| `RANK`, `OMPI_COMM_WORLD_RANK`, `PMI_RANK`, `SLURM_PROCID`, `LOCAL_RANK` | The process rank, in that order of precedence: the global ranks (torchrun, Open MPI, MPICH and Intel MPI, SLURM) before the per-node `LOCAL_RANK`. The first one that holds an integer is used, and the rank is 0 when none does. Only rank 0 writes progress unless `PACE_ALL_RANKS` is on. See [Distributed runs](logging.md#distributed-runs-and-ranks). |
| `NO_COLOR` | See the table above. |

Pace does not read `PYTHONIOENCODING`, but it follows the encoding of the stream it writes to
(see [Encodings](logging.md#encodings-and-the-ascii-fallback)).

## `configure` only

| Option | Meaning |
|---|---|
| `clock=` | For tests only: a clock of two functions, `monotonic` and `wall`, so a test runs without sleeping (see `FakeClock` in the repository's `tests/conftest.py`). |

`logger.reset()` drops every explicit setting so the environment and the defaults apply
again.

## Settings for common situations

| Situation | Setting |
|---|---|
| A job under a scheduler that reads progress | The defaults. Do not change `PACE_HEARTBEAT` to more than the scheduler expects. |
| The job's stdout is data for another program | `PACE_STREAM=stderr` |
| The job runs with a pseudo-terminal (`docker run -t`, `ssh -t`) and something parses its logs | `PACE_BAR=never` |
| Plain logs without escape sequences | `NO_COLOR=1` or `PACE_COLOR=never` |
| A short job where you want to see progress lines | `PACE_MIN_INTERVAL=1` |
| Debug output | `PACE_LEVEL=debug` |
| Every rank's progress in one run | `PACE_ALL_RANKS=1` |
