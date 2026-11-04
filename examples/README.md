# Examples

One small, runnable script per style of logging and progress reporting. Each one finishes
in about a second on fake work, needs nothing beyond `precisionai-pace`, and models the
practice the docs and the `adopt-pace` skill describe: one root scope, real totals, short
fixed labels, keyword arguments. Copy the one that matches your job.

## Running one

```bash
pip install precisionai-pace
git clone https://github.com/Precision-AI-Inc/pace && cd pace
python examples/track_loop.py
```

On a terminal progress is one bar that redraws in place. When the output is a pipe or a
file, Pace writes a progress line at most every 15 seconds, so a one-second example would
write only its first and last line. To see every progress line:

```bash
PACE_MIN_INTERVAL=0 python examples/track_loop.py
PACE_MIN_INTERVAL=0 PACE_BAR=never python examples/track_loop.py | tail
```

`PACE_BAR=never` writes the lines (with their `[pace:1 ...]` trailer) even on a terminal.
Every setting is described in `docs/configuration.md`.

The scripts import only the standard library and `precisionai.pace`, except the two
framework examples, which need an extra:

```bash
pip install "precisionai-pace[lightning]"                  # lightning_example.py
pip install "precisionai-pace[transformers]"               # transformers_example.py
```

## The examples

| Example | Situation it models | Shows |
|---|---|---|
| [`basic_logging.py`](basic_logging.py) | A script that only needs good log lines | `logger` levels, `exception`, `log`, `configure`, `reset` |
| [`track_loop.py`](track_loop.py) | A plain loop that is nearly the whole job | `track`, `description=`, `unit=`, `root=True` |
| [`epochs_and_batches.py`](epochs_and_batches.py) | Training for several epochs of batches | One root `progress` over `epochs * batches`, `advance` |
| [`manual_progress.py`](manual_progress.py) | Concurrent requests that finish out of order | `progress`, `advance`, `concurrent.futures` |
| [`phases_and_plan.py`](phases_and_plan.py) | Preprocess, then train, then evaluate | `plan`, `phase` (also as a plan entry), a counted loop inside a phase |
| [`resume.py`](resume.py) | Continuing from a checkpoint | `completed=`, `resumed=True` |
| [`unknown_total.py`](unknown_total.py) | A paged stream whose length is learned later | `progress` with no total, `set_total` |
| [`timed_steps.py`](timed_steps.py) | One-off steps (load a model, save results) around a loop | `@timed`, steps nested in a root |
| [`nested_scopes.py`](nested_scopes.py) | Train and validate loops inside an epoch scope | `p=` parents, `scope=`, what nesting means for the estimate |
| [`stdlib_logging.py`](stdlib_logging.py) | A library that logs with the standard `logging` module | `capture_logging`, `logging.captureWarnings` |
| [`library_hooks.py`](library_hooks.py) | A library draws its own tqdm bar and prints its own log lines | `PaceTqdm` as `use_tqdm=`, `capture_logging(take_over=[...])` |
| [`distributed.py`](distributed.py) | A multi-GPU job; two ranks simulated with `RANK` | Progress from rank 0 only, rank-guarded log lines |
| [`reading_output.py`](reading_output.py) | The consumer side: a scheduler or dashboard reading a job's output | `LineSplitter`, `parse_line`, `JobForecast` |
| [`lightning_example.py`](lightning_example.py) | A PyTorch Lightning `Trainer` | `precisionai.pace.lightning.PaceCallback` (needs the `lightning` extra) |
| [`transformers_example.py`](transformers_example.py) | A Hugging Face `Trainer` (TRL takes the same callback) | `precisionai.pace.transformers.PaceCallback` (needs the `transformers` extra) |

## Checking an example

`tests/test_examples.py` runs every example except the two framework ones, and checks the
output with the `adopt-pace` skill's checker, `skills/adopt-pace/verify.py`, which you can
point at any job of your own:

```bash
PACE_MIN_INTERVAL=0 PACE_BAR=never python examples/phases_and_plan.py > run.log
python skills/adopt-pace/verify.py run.log
```
