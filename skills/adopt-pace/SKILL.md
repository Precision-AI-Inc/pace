---
name: adopt-pace
description: Add Pace (precisionai-pace) logging and progress reporting to this repository so its jobs report completion estimates to a scheduler such as OrchAI. Use when asked to "add Pace", "add progress reporting", "make this job report its ETA", or to adopt precisionai-pace in a training, evaluation, inference or data-pipeline repo (plain PyTorch, PyTorch Lightning, Hugging Face Transformers/TRL, custom loops, concurrent evaluation harnesses, model servers).
---

# Adopt Pace in this repository

Pace (`precisionai-pace`, import `precisionai.pace`) is a zero-dependency Python logger with
progress reporting. Each progress line ends with a trailer such as
`[pace train c=1840 t=5000 e=763.1 q=57 r=6bfc3dd5 root]`; a scheduler reads them from the
job's output to estimate when it finishes. Make this repo's jobs write them, with the
smallest correct change, and prove it.

Five steps: **survey, plan, edit, verify, report**. Do not skip the plan. Each situation has
its own file under `patterns/` (beside this one): read **only** the ones the survey points to.
`<skill-dir>` below is this folder (for example `.claude/skills/adopt-pace`).

## What the scheduler needs

- A **total**, so progress is a fraction. No known total gives no estimate.
- Exactly **one `root` scope** standing for the whole job (or a `plan(...)` listing every
  phase). **The job is complete when the root ends**; work after it is invisible.
- An **`end`** (or `fail`) for every scope. Never hand-write a `[pace ...]` trailer or put one in
  a log message.

Pace flushes every line, rate-limits by time, writes progress from rank 0 only (no
`if rank == 0:` guard) and never raises into the job. A **scope** is one counted piece of
work: `with progress(total=n, ...) as run:` then `run.advance()`, or `track(iterable, ...)`.
Pass `description=`, `unit=`, `total=`, `scope=` by keyword.

## 1. Survey (read-only)

With `git ls-files`, `ls` and grep, note:

1. **What the job runs** (Dockerfile `CMD`, `orch_starter/config.yaml`, `[project.scripts]`,
   Makefile, README). Several scripts: which are jobs? Ask if unclear. Instrument the script,
   not the launcher (`torchrun`, `accelerate launch`).
2. **Framework** (Lightning, `Trainer`/`SFTTrainer`, `accelerate`, DDP, a plain loop). One that
   builds the trainer from config (Axolotl, LLaMA-Factory, LightningCLI) needs another hook.
3. **Existing progress and logging** (`print`, `logging`, `tqdm`, `rich`, home-made ETA code):
   which is the **main progress indicator**, which incidental?
4. **Loops and totals**: what each main loop iterates, its natural total and where it comes
   from; phases and their share of runtime; resume, distributed, concurrency, a server the
   job starts.
5. **Every place dependencies are declared**, including each `pip install` in Dockerfiles
   and scripts; Python version; a private or `--no-index` index; lock files.
6. **A short smoke test** (`--max-steps`, tiny config, CPU path); note it if the job needs a
   GPU or data you do not have.

## 2. Choose the scopes

Each file ends with runnable examples to copy: Pace's
[`examples/`](https://github.com/Precision-AI-Inc/pace/blob/main/examples/README.md) and
tested ones in OrchAI's [example repo](https://github.com/Precision-AI-Inc/orchai-examples)
(detection, LLM, VLM).

| Situation in the repo | Read |
|---|---|
| One main loop that is the job; a `tqdm` to replace | `patterns/loop.md` |
| Epochs times batches; validation inside the loop | `patterns/epochs.md` |
| Resumes from a checkpoint | `patterns/resume.md` |
| Phases in sequence; work after the main loop; one unit through every stage | `patterns/pipeline.md` |
| Concurrent requests or workers | `patterns/concurrent.md` |
| The job starts a model server (Ollama, vLLM server) and calls it | `patterns/model-server.md` |
| No known total; files or bytes | `patterns/unknown-total.md` |
| DDP, Accelerate, torchrun | `patterns/distributed.md` |
| Lightning `Trainer` or `LightningCLI` | `patterns/lightning.md` |
| HF `Trainer`, TRL, or a wrapper that builds one | `patterns/transformers.md` |
| A library call that takes a `tqdm` class (vLLM `generate(use_tqdm=)`), or vLLM in chunks | `patterns/tqdm-class.md` |
| Standard `logging`, a library's own log lines | `patterns/logging.md` |
| Any dependency edit (always) | `patterns/dependencies.md` |
| The job runs on OrchAI (`orch_starter/` in the repo, or the user says so) | `patterns/orchai.md` |

**Choosing the root.** Ask: "when this scope ends, is the job really done?" If more than a few
percent of the runtime comes after the loop (evaluation, export, upload), the loop alone is
not the root: use a plan or a root over a common unit (`pipeline.md`). Never mark two roots.
A framework callback is the root; do not add another.

## 3. Plan (before any edit)

Show the user a short plan and ask for confirmation (if they said to go ahead, show it anyway
and continue):

```text
Pace plan
  Job: train.py (Dockerfile CMD), PyTorch loop, single GPU, resumes with --resume
  Root scope: "train" = epochs x len(loader) steps
  Changes: requirements.txt and Dockerfile:12 add precisionai-pace;
           train.py:41-58 replace the tqdm loop with progress(...), resume-aware
  Optional (say yes): capture_logging() so library logging uses the same format
  Left alone: tqdm in download_data.py, the print() calls
  Will report: step progress for the whole job. Gap: the final evaluation has no count
  Smoke test: python train.py --epochs 1 --max-steps 30 on CPU
```

**Offer**, do not do, `capture_logging()`, turning off a framework's own bar,
`PACE_STREAM=stderr` and `PACE_BAR=never`.

## 4. Edit

Smallest correct change: no reformatting, renaming or new abstractions. Dependencies in every
place found (`patterns/dependencies.md`). A framework gets `PaceCallback()`, not loop edits;
a custom loop gets a real total and exactly one `root=True`. Replace `tqdm` only where it is
the main indicator. Match the repo's imports and style; short fixed labels (`"Train"`) and
lower-case scope ids (`scope="train"`).

**Never put secrets, tokens, prompts, credential paths or a whole config in a label, scope id
or log message**; labels are written once per scope, so keep them static, not `f"{args}"`.
Pace writes to stdout: if stdout carries data another program reads, set `PACE_STREAM=stderr`;
under a pseudo-terminal the scheduler reads, set `PACE_BAR=never`.

## 5. Verify

You must run something: the real job with a tiny config, a few steps of the real script, or
the instrumented section alone, after `pip install precisionai-pace` (with the extra). If that
fails, say so and do not claim verification. Outside a terminal Pace writes at most a line per
15 s, so make a short run print its updates (keep logs out of the repo):

```bash
PACE_MIN_INTERVAL=0 PACE_HEARTBEAT=5 PACE_BAR=never python train.py <tiny args> > /tmp/pace-smoke.log 2>&1
python <skill-dir>/verify.py /tmp/pace-smoke.log
```

`verify.py` checks that every trailer parses, the root count, a total and an `end` or `fail`
per scope (a `step`, the one line of a short `@timed` call, `phase()` or child scope, counts as both),
plan names against scope ids, and the final estimate (a finished job reads
`complete`; a run under a minute never reaches a live one). Also, when they apply: **resume**
(run twice; look for `res` on the start), **DDP** (`RANK=1` writes no progress lines), **callbacks** (a
few steps on a tiny model), **exceptions** (raise once; expect a `fail`). Fix and rerun.

## 6. Report

In this order: **Changed** (a line per file); **Root scope** (id, unit, total, why); **What
the scheduler can estimate**; **Verified** (the exact command and verdict) and **Not run**
(GPU, DDP, callback, Docker build, and why); **Remaining gaps** (unknown totals, work after
the root, evaluation pauses, early stopping, lock files, wrappers with no hook, an unused
`tqdm` dependency left in place); **Optional follow-ups** not applied. If the repo has
`orch_starter/config*.yaml` files, offer `project.expected_runtime_minutes` (`patterns/orchai.md`);
invent no other config keys.

An honest "unknown" beats an invented total. If Pace is not on the index the repo installs
from, the dependency edit is still right; say the install fails until it is.
