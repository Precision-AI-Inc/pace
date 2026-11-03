# Pace patterns: before and after

Snippets for the situations `SKILL.md` sends you here for. Imports go at the top of
the file, with the other imports. Always pass `description=`, `unit=`, `total=` and
`scope=` by keyword: `progress()` takes `total` first, `track()` takes the iterable
first and `total` last.

What every pattern relies on:

- A **scope** is one counted piece of work. `progress(...)` is the handle form
  (`run.advance()`, `run.update(n)`, `run.set_total(n)`), `track(iterable, ...)` is the
  loop form. Either writes `start`, throttled `upd`, then `end` (or `fail` if an
  exception leaves a `progress` block).
- `root=True` marks the one scope that stands for the whole job. When the root ends,
  the scheduler considers the job complete.
- Pace itself keeps rank 0 only (it reads `RANK`, `LOCAL_RANK`, `SLURM_PROCID`), so
  progress calls need no `if rank == 0:` guard.
- Output goes to stdout, flushed per line, rate-limited by time (a line every 15 s at
  most, a heartbeat every 120 s at most). Step count does not change the volume.

## Contents

1. [Plain loop, with tqdm](#1-plain-loop-with-tqdm)
2. [Epochs and batches](#2-epochs-and-batches)
3. [Resuming from a checkpoint](#3-resuming-from-a-checkpoint)
4. [Multi-phase pipeline](#4-multi-phase-pipeline)
5. [Concurrent evaluation harness](#5-concurrent-evaluation-harness)
6. [Unknown totals, files and bytes](#6-unknown-totals-files-and-bytes)
7. [Distributed training (DDP, Accelerate, torchrun)](#7-distributed-training)
8. [PyTorch Lightning](#8-pytorch-lightning)
9. [Hugging Face Transformers and TRL](#9-hugging-face-transformers-and-trl)
10. [Standard `logging`](#10-standard-logging)
11. [Dependency declarations](#11-dependency-declarations)

## 1. Plain loop, with tqdm

Before:

```python
from tqdm import tqdm

pbar = tqdm(loader, desc="Train", total=len(loader))
for step, batch in enumerate(pbar):
    loss = train_step(batch)
    pbar.set_postfix(loss=loss)
```

After. The loop is the job, so it is the root:

```python
from precisionai.pace import logger, track

for step, batch in enumerate(track(loader, description="Train", unit="batch", root=True)):
    loss = train_step(batch)
    if step % 100 == 0:
        logger.info(f"step {step} loss {loss:.4f}")
```

`tqdm` to Pace:

| tqdm | Pace |
|---|---|
| `tqdm(it, desc="x", total=n)`, `trange(n)` | `track(it, description="x", total=n)`, `track(range(n))` |
| `pbar.update(k)` | `run.advance(k)` on a `progress(...)` handle |
| `pbar.set_postfix(...)`, `pbar.set_description(...)` | no equivalent: log a line every N steps with `logger.info`, or drop it. Metrics belong in MLflow or similar. |
| `tqdm.write(msg)` | `logger.info(msg)` |
| `leave=`, `ncols=`, `position=`, `disable=` | not needed |

`track()` takes `total` from `len(iterable)`. A generator or `itertools.islice` has no length: pass `total=`.

`track()` writes `end` (not `fail`) when the loop body raises, because Python hands it the
same `GeneratorExit` for `break` and for an exception. If a crash in the body must show as
failed, use the handle form:

```python
with progress(total=len(loader), description="Train", unit="batch", root=True) as run:
    for batch in loader:
        train_step(batch)
        run.advance()
```

`tqdm` that is not the job's main indicator (a download bar, a one-off preprocessing
bar) stays as it is.

## 2. Epochs and batches

Do not make one `track()` per epoch and call each a root. One scope with the whole
denominator gives the scheduler one smooth rate:

```python
total_steps = args.epochs * len(loader)
with progress(total=total_steps, description="Train", unit="step", root=True) as run:
    for epoch in range(args.epochs):
        for batch in loader:
            train_step(batch)
            run.advance()
        logger.info(f"epoch {epoch + 1}/{args.epochs} done")
```

- With a `max_steps` cap, the total is `min(args.max_steps, args.epochs * len(loader))`.
- With early stopping, the estimate is an upper bound: say so in the report.
- With gradient accumulation, advance per micro-batch (what the loop iterates) and keep
  the total in the same unit.
- Validation inside the loop: leave it inside the root. A pause that recurs is smoothed
  out of the rate, so the estimate is a little optimistic by the evaluation time still to
  come; report it as a known gap. Do not give it a root of its own.

## 3. Resuming from a checkpoint

Pass the amount already done, and `resumed=True` only when the run really resumes. The
total stays the full total, not the remaining work:

```python
start_step = 0
if args.resume:
    state = load_checkpoint(args.resume)
    start_step = state["step"]

total_steps = args.epochs * len(loader)
with progress(
    total=total_steps,
    description="Train",
    unit="step",
    root=True,
    completed=start_step,
    resumed=start_step > 0,
) as run:
    step = start_step
    while step < total_steps:
        train_step(next_batch())
        step += 1
        run.advance()
```

Use the unit the checkpoint stores (global step, or epoch index times batches per
epoch). If the loop skips already-seen batches by slicing the loader, keep
`total=` explicit so the total is not the length of the slice. The scheduler starts the
remaining-work calculation from the resumed `c=` and does not average the rate across
the restart.

## 4. Multi-phase pipeline

Preprocess, then train, then evaluate. Two shapes; pick one.

### a. A plan, when each phase has a count of its own

Declare the phases first, then run each as a counted scope **whose `scope=` id is the
name in the plan**:

```python
from precisionai.pace import plan, progress, track

plan([("prepare", 0.10), ("train", 0.80), ("evaluate", 0.10)])

with progress(total=len(raw_files), description="Prepare", unit="file", scope="prepare") as run:
    for path in raw_files:
        preprocess(path)
        run.advance()

for batch in track(loader, description="Train", unit="step", scope="train", total=total_steps):
    train_step(batch)

with progress(total=len(val_set), description="Evaluate", unit="sample", scope="evaluate") as run:
    for item in val_set:
        score(item)
        run.advance()
```

- Weights are each phase's share of the **runtime**, not of the number of items. They
  only matter for phases that have not started: the scheduler learns the seconds per
  weight from the phases already finished. Ask the user for rough shares; if nobody
  knows, leave the weights out (all phases count equally) and say they are guesses.
- The name in `plan(...)` becomes the scope id (lower case, anything else becomes `-`).
  Setting `scope=` explicitly on each phase keeps the two in step. A mismatch is
  silent at runtime, and `verify.py` reports it.
- There is no `root=True` here: the plan covers the whole job, and the job counts as
  complete when every planned phase has ended.
- **Do not wrap a long phase in `with phase("Train"):` and put the counted loop inside
  it.** A `phase` is a scope of total 1 that stays at 0 until it exits, so it has no
  estimate while it runs, and a plan cannot give an estimate while any of its phases
  has none. The scheduler sees nothing during the longest phase. Make the counted
  scope itself the plan entry, as above. Use `phase()` only for short, uncounted
  steps:

```python
from precisionai.pace import phase, timed

with phase("Load model"):
    model = load_model()

@timed("Export")
def export(model): ...
```

  A `phase` or `@timed` function that is not listed in the plan is just a timed step;
  one that is listed in the plan blocks the plan's estimate while it runs, which is
  harmless when it lasts seconds.

### b. One root over a common unit

When the whole job moves in one unit (N documents go through every stage, or the work
can be put in step-equivalents), a single root with a single total beats a plan:

```python
with progress(total=len(documents), description="Index", unit="doc", root=True) as run:
    for doc in documents:
        embed(chunk(parse(doc)))
        run.advance()
```

### c. A root that is only the main loop

When the main loop is nearly the whole runtime and the rest is seconds (load a
model, write a small result), make the loop the root and leave the rest as `phase()` or
plain `logger.info` lines. Remember what the root means: **the job is complete when the
root ends**. Work that runs after the root's `end` (final evaluation, export,
upload) is invisible to the scheduler. If that work is more than a few percent of the
runtime, use (a) or (b) instead.

## 5. Concurrent evaluation harness

Count finished requests, from the thread that collects the results. Scopes are thread
safe, and a scope opened inside a worker thread would have no parent anyway:

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

from precisionai.pace import progress

with ThreadPoolExecutor(max_workers=args.workers) as pool, progress(
    total=len(cases), description="Evaluate", unit="request", root=True
) as run:
    futures = [pool.submit(score, case) for case in cases]
    for future in as_completed(futures):
        results.append(future.result())
        run.advance()
```

- `progress` is entered after the pool and so exits first: if `future.result()` raises,
  the scope is written as `fail`.
- Count a request once, when it is finally done. Retries are not extra requests.
- With `ProcessPoolExecutor`, advance in the parent; do not create scopes in the workers.
- With asyncio, `run.advance()` after each `await` of `asyncio.as_completed(...)` works the same.
- Several harnesses running side by side in one process (rare): one root over their sum, not
  one root each.

## 6. Unknown totals, files and bytes

Find a real denominator before settling for "unknown". `len(dataset)`, a `wc -l`, the
number of files from a glob, `os.path.getsize`, the count a manifest holds:

```python
files = sorted(Path(args.input).glob("**/*.jsonl"))
with progress(total=len(files), description="Convert", unit="file", root=True) as run:
    for path in files:
        convert(path)
        run.advance()
```

```python
size = os.path.getsize(path)
with progress(total=size, description="Download", unit="byte") as run:
    for chunk in stream:
        out.write(chunk)
        run.advance(len(chunk))
```

If the total only becomes known later (the first pass counts), start with `total=None`
and call `run.set_total(n)` once it is known. A scope that never has a total is reported as
unknown; the scheduler gets no estimate from it.

## 7. Distributed training

Instrument the script that runs on every rank, and write the loop exactly as for one
process. Pace reads the rank from the environment, so only rank 0 writes progress
lines, and the other ranks run the same calls silently:

```python
with progress(total=total_steps, description="Train", unit="step", root=True) as run:
    for batch in loader:        # per-rank loader: len(loader) is already per rank
        train_step(batch)
        run.advance()
```

- Count optimizer steps (or batches per rank), which advance in lockstep. If the unit is
  samples, advance by `batch_size * world_size` and use the global sample count as the total.
- `logger.info(...)` is not filtered by rank, and neither is anything else the job
  prints. Keep the existing `if rank == 0:` / `accelerator.is_main_process` guards
  around the messages that had them.
- The entry point is the script, not the launcher: instrument `train.py`, not
  `torchrun` / `accelerate launch`. Set nothing for ranks; `PACE_ALL_RANKS=1` exists
  only for debugging.

## 8. PyTorch Lightning

<!-- verify against adapters: the exact constructor arguments of PaceCallback, and whether it accepts the legacy pytorch_lightning import as well as lightning.pytorch -->

Add the callback. Do not touch the training loop. It reports `trainer.global_step`
against `trainer.estimated_stepping_batches` as the root scope, with validation as a
child scope, and restores the resume position from the checkpoint:

```python
import lightning.pytorch as pl
from precisionai.pace.lightning import PaceCallback

trainer = pl.Trainer(
    max_epochs=args.epochs,
    callbacks=[PaceCallback(), *existing_callbacks],
)
trainer.fit(model, datamodule=dm, ckpt_path=args.resume)
```

- Dependency: `precisionai-pace[lightning]`.
- If the trainer is built from a config file or `LightningCLI`, add the callback where
  callbacks are declared (for `LightningCLI`, `trainer.callbacks` in the YAML with
  `class_path: precisionai.pace.lightning.PaceCallback`) <!-- verify against adapters: class_path/init_args form -->.
- Lightning's own progress bar writes to stderr and does not conflict. Leave it unless the
  user wants the log quieter (`enable_progress_bar=False`).
- `trainer.estimated_stepping_batches` is infinite when neither `max_steps` nor
  `max_epochs` bounds the run <!-- verify against adapters: how the callback reports this -->; say so in the report if the trainer is unbounded.

## 9. Hugging Face Transformers and TRL

<!-- verify against adapters: the exact constructor arguments of PaceCallback -->

```python
from transformers import Trainer
from precisionai.pace.transformers import PaceCallback

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    callbacks=[PaceCallback()],
)
trainer.train(resume_from_checkpoint=args.resume)
```

`SFTTrainer`, `DPOTrainer` and the other TRL trainers take the same `callbacks=[...]`,
or `trainer.add_callback(PaceCallback())` after construction. The callback reports
`state.global_step` against `state.max_steps`, rank 0 only.

- Dependency: `precisionai-pace[transformers]`.
- When a wrapper builds the trainer (Axolotl, LLaMA-Factory, torchtune, a company
  library), look for its hook for extra callbacks. If it has none, say so in the report
  rather than patching an installed package.
- `disable_tqdm=True` in `TrainingArguments` removes the Trainer's tqdm bar; offer it,
  do not apply it unasked.
- `max_steps = -1` with no epoch bound gives no total <!-- verify against adapters: what the callback does then -->; the Trainer still computes `max_steps` from epochs and dataset length, so this is rare.

## 10. Standard `logging`

Optional, and only when the user wants one log format. Call it once at startup:

```python
from precisionai.pace import capture_logging

capture_logging("INFO")
```

It routes the root logger's records (so PyTorch's, Transformers' and the repo's own)
through Pace's formatter. If the repo already called `logging.basicConfig(...)` or
added a `StreamHandler` to the root logger, every record would be written twice:
remove that handler in the same change, or skip `capture_logging()`.

Loguru and `print` stay as they are. Pace does not bridge Loguru.

## 11. Dependency declarations

Add `precisionai-pace` everywhere the repo installs its own dependencies, with the same
pinning style as its neighbours. The extras go in the same brackets.

| Where | Entry |
|---|---|
| `requirements*.txt` | `precisionai-pace` (or `precisionai-pace[lightning]`) |
| `pyproject.toml`, PEP 621 | `dependencies = [..., "precisionai-pace"]` (or the optional group the job installs) |
| `pyproject.toml`, Poetry | `precisionai-pace = "*"`, or `precisionai-pace = { version = "*", extras = ["lightning"] }` |
| `setup.py`, `setup.cfg` | `install_requires` list |
| conda `environment.yml` | under the `pip:` list, not as a conda package |
| `Dockerfile`: `pip install a b c` | add it to that line. If the Dockerfile only runs `pip install -r requirements.txt`, the requirements file is enough. |
| `uv.lock`, `poetry.lock`, `Pipfile.lock` | do not edit by hand; tell the user to re-lock, or run the repo's own lock command if it works offline-safe |

A repo that pins exact versions gets an exact pin: look up the version with
`pip index versions precisionai-pace`. Note that a Dockerfile installing from a private or
`--no-index` mirror needs Pace available there; flag it when the install fails.
