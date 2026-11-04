---
name: adopt-pace
description: Add Pace (precisionai-pace) logging and progress reporting to this repository so its jobs report completion estimates to a scheduler such as OrchAI. Use when asked to "add Pace", "add progress reporting", "make this job report its ETA", or to adopt precisionai-pace in a training, evaluation, inference or data-pipeline repo (plain PyTorch, PyTorch Lightning, Hugging Face Transformers/TRL, custom loops, concurrent evaluation harnesses).
---

# Adopt Pace in this repository

Pace (`precisionai-pace`, import `precisionai.pace`) is a zero-dependency Python
logger with progress reporting. Each progress line ends with a trailer such as
`[pace:1 s=train root=1 k=upd c=1840 t=5000 e=763.1 q=57]`. A scheduler reads
the trailers from the job's output to estimate when the job will finish. Your task is to
make this repository's jobs write them, with the smallest correct change, and to prove it.

Work in five steps: **survey, plan, edit, verify, report**. Do not skip the plan.
Snippets for each situation are in [patterns.md](patterns.md); the checker is
[verify.py](verify.py) (both beside this file; `<skill-dir>` below means this folder, for
example `.claude/skills/adopt-pace` in the repo or `~/.claude/skills/adopt-pace`).

## What the scheduler needs

- A **total**, so progress can be a fraction. A loop with no known total gives no estimate.
- Exactly **one `root` scope**, the one that stands for the whole job (or a `plan(...)`
  that lists every phase). **The job counts as complete when the root ends**, so work that
  runs after the root is invisible.
- An **`end`** (or `fail`) record for every scope.
- Progress lines that **parse**. Pace writes them; never hand-write a `[pace:` trailer,
  and never put one in a log message.

What Pace does for you: it flushes every line, rate-limits by time (not by step count),
writes only from rank 0 under DDP, and never raises into the job.

## 1. Survey (read-only)

Find out the following and keep notes for the plan. Use `git ls-files`, `ls`, and grep.

1. **What the job actually runs.** Dockerfile `CMD`/`ENTRYPOINT`, `orch_starter/config.yaml`
   or any job config's command, `pyproject.toml` `[project.scripts]`, Makefile targets,
   shell scripts, README "how to run" sections. A repo often has several scripts (train,
   evaluate, infer, preprocess): note which ones are the jobs. If it is unclear which
   to instrument, ask. Instrument the script, not the launcher (`torchrun`,
   `accelerate launch`, `python -m`).
2. **Framework.** grep for `pl.Trainer|lightning|pytorch_lightning`,
   `transformers|Trainer(|SFTTrainer|DPOTrainer|TrainingArguments`, `accelerate`,
   `torchrun|init_process_group|DistributedDataParallel`, or a plain
   `for epoch in range` loop. A framework that builds the Trainer from config (Axolotl,
   LLaMA-Factory, LightningCLI) needs a different hook: see patterns 8 and 9.
3. **Existing logging and progress:** `print(`, `logging.`, `loguru`, `tqdm|trange`,
   `rich.progress`, `set_postfix`, home-made ETA code (`time.time()` arithmetic, "remaining").
   Note which of these is the **main progress indicator** of the job and which are
   incidental (a download bar).
4. **Loops and denominators.** For each main loop: what it iterates and its natural
   total (steps, epochs times batches, samples, requests, files, bytes), and where that
   total comes from (`len(loader)`, `max_steps`, a manifest). Note loops with no known
   total.
5. **Phases.** Setup (data load, model load), main work, evaluation, export or upload.
   Roughly what share of runtime each takes, if the README, logs or the user say.
6. **Distributed setup.** DDP, Accelerate, torchrun, SLURM, multi-node.
7. **Checkpoint resume.** `--resume`, `resume_from_checkpoint`, `ckpt_path`,
   `load_state_dict`, saved `step`/`epoch`, and where the loop restarts.
8. **Concurrency.** `ThreadPoolExecutor`, `ProcessPoolExecutor`, `as_completed`,
   asyncio, joblib, multiprocessing.
9. **How dependencies are declared**, every place: `requirements*.txt`, `constraints*.txt`,
   `pyproject.toml` (PEP 621, Poetry, uv, PDM), `setup.py`/`setup.cfg`, `Pipfile`,
   conda `environment*.yml`, and **every `pip install` in Dockerfiles**, shell scripts
   and Makefiles. Also: the Python version, a private or `--no-index` package index, and
   lock files.
10. **How to run a short smoke test**: a `--max-steps`/`--epochs 1`/tiny-config flag, a CPU
    path, a test fixture. If the job needs a GPU or data you do not have, note it now.

Do not edit anything yet.

## 2. Choose the scopes

| What you found | Use |
|---|---|
| Lightning `Trainer` | `PaceCallback()` from `precisionai.pace.lightning` (pattern 8). No loop edits. |
| HF `Trainer` / TRL trainers | `PaceCallback()` from `precisionai.pace.transformers` (pattern 9). No loop edits. |
| One main loop that is nearly the whole runtime | `track(...)` or `progress(...)` with `root=True` (patterns 1, 2) |
| Phases in sequence, each with a count (preprocess, train, evaluate) | `plan([...])` plus one counted scope per phase whose `scope=` matches the plan name; no root (pattern 4a) |
| One unit moves through every stage | one `progress(..., root=True)` over that unit (4b) |
| Concurrent requests or workers | one `progress(..., root=True)` advanced where results are collected (pattern 5) |
| One library call that runs the whole job and takes a `tqdm` class (vLLM `LLM.generate(..., use_tqdm=...)`) | pass `PaceTqdm` from `precisionai.pace.tqdm` as that argument (pattern 12) |
| No known total | find one (count first, file size, manifest); only then leave it `None` and report it as a gap (pattern 6) |

**Read the matching example before you edit.** Each situation above has a small runnable
script in the Pace repository, written the way this skill asks for it. Copy its shape
rather than inventing one ([all of them](https://github.com/Precision-AI-Inc/pace/blob/main/examples/README.md)):

| Situation | Example in `examples/` |
|---|---|
| One main loop | `track_loop.py` (`track`), `manual_progress.py` (`progress`, concurrent results) |
| Epochs times batches | `epochs_and_batches.py` (one root over every step), `nested_scopes.py` (child scopes) |
| Preprocess, train, evaluate | `phases_and_plan.py` (`plan` and `phase`) |
| Resuming from a checkpoint | `resume.py` (`completed=`, `resumed=True`) |
| No known total | `unknown_total.py` (`set_total` when it appears) |
| One-off steps | `timed_steps.py` (`@timed`) |
| Standard `logging`, ranks | `stdlib_logging.py`, `distributed.py` |
| Lightning, Hugging Face | `lightning_example.py`, `transformers_example.py` |

**Choosing the root.** Ask: "when this scope ends, is the job really done?" If more than a
few percent of the runtime comes after the loop (final evaluation, export, upload), the
loop alone is not the root: use a plan, or a root over a common unit. Never mark two roots.
A framework callback is the root; do not add another.

A long counted loop inside `with phase(...)` is fine: a `phase` has no rate of its own, so
the scheduler uses the estimate of the loop inside it. If the phase holds several loops one
after another (one per epoch), that estimate covers only the running loop and is a lower
bound, so count the whole stage in one scope instead (pattern 4a).

## 3. Plan (before any edit)

Show the user a short plan and ask for confirmation, unless they already told you to go
ahead (then show it anyway, and continue). Format:

```text
Pace plan
  Job: train.py (Dockerfile CMD), PyTorch loop, single GPU, resumes with --resume
  Root scope: "train" = epochs x len(loader) steps
  Changes
    requirements.txt           add precisionai-pace
    Dockerfile:12              add precisionai-pace to the explicit pip install
    train.py:41-58             replace the tqdm loop with progress(...), resume-aware
  Optional (say yes to include)
    capture_logging() in train.py so library logging uses the same format
  Left alone: tqdm in download_data.py, the print() calls
  Will report: step progress and time remaining for the whole job. Gap: the final
               evaluation is a few seconds and has no count of its own
  Smoke test: python train.py --epochs 1 --max-steps 30 on CPU
```

List every file and the kind of change, name the root scope and its unit and total, name
anything the scheduler will not be able to estimate, and offer the optional items
(`capture_logging()`, turning off a framework's own progress bar, `ENV PACE_STREAM=stderr`
or `PACE_BAR=never` in a Dockerfile) rather than doing them.

## 4. Edit

Make the smallest change that is correct.

1. **Dependencies**: add `precisionai-pace` (with `[lightning]` or `[transformers]` when
   an adapter is used) in **every** place the repo declares dependencies, in the pinning
   style of its neighbours. That includes each explicit `pip install` line in a Dockerfile
   or script. Do not hand-edit lock files: say they need re-locking. Table in pattern 11.
   Pace needs Python 3.11 or later.
2. **Frameworks**: add `PaceCallback()` to the trainer's callbacks. Do not also
   instrument the loop.
3. **Custom loops**: wrap the main loop with `track(...)` or `progress(...)` with a real
   total. Mark exactly one `root=True`. Give a substantial phase before or after it its own
   counted scope, or use a `plan(...)`. On resume, pass `completed=` and `resumed=True`.
4. **tqdm**: replace it only where it is the job's main progress indicator
   (pattern 1 has the mapping). Leave unrelated `tqdm` and `print` calls alone. Replacing
   `print` is not part of this task unless the user asks. Remove a `tqdm` import that the
   change left unused, but leave the dependency declaration alone (the repo may use it
   elsewhere) and say so in the report. Optionally route standard
   `logging` through Pace with `capture_logging()` (pattern 10), only if the user agreed.
5. **Style**: imports at the top of the file, in the repo's import order; match the
   repo's formatting and typing conventions. Use keyword arguments, short fixed labels
   (`"Train"`, `"Evaluate"`), lower-case scope ids (`scope="train"`).
6. **Never put secrets, tokens, prompts, paths with credentials or a whole config in a
   label, scope id or log message.** Labels are written once per scope; keep them short and
   static, not `f"{args}"`.
7. **Output stream.** Pace writes to stdout. If the job's stdout carries data that another
   program consumes, set `PACE_STREAM=stderr` (an environment variable in the Dockerfile or
   the run command, or `logger.configure(stream="stderr")`). If a job runs under a
   pseudo-terminal whose output the scheduler reads, set `PACE_BAR=never`.

Keep the diff small: no reformatting of untouched code, no renamed variables, no new
abstractions.

## 5. Verify

You must run something. In order of preference: the real job with a tiny config on CPU;
a few steps of the real script (`--max-steps 20`); the instrumented section alone.

1. Install Pace into the environment you run in: `pip install precisionai-pace` (with the
   extra). If it cannot be installed (for example no network, or no access to the package index), say so and stop at
   the part you cannot run: do not claim verification.
2. Run the job with output captured. Outside a terminal Pace writes a line at most every
   15 s, so make a short run print its updates:

   ```bash
   PACE_MIN_INTERVAL=0 PACE_HEARTBEAT=5 PACE_BAR=never python train.py <tiny args> > /tmp/pace-smoke.log 2>&1
   python <skill-dir>/verify.py /tmp/pace-smoke.log
   ```

   Keep logs out of the repo. `verify.py` is stdlib plus `precisionai.pace`; it needs the
   same environment as the run.
3. Read its result. It checks that every trailer line parses, how many roots there are
   (it warns on none and fails on more than one), that every scope has a total and an
   `end` or `fail`, that planned phase names match real scope ids, and what the
   estimator concludes at the end (a finished job should read `complete`). A run under a
   minute never reaches a live estimate; that is expected.
4. Special cases to check, when they apply:
   - **Resume**: run once to write a checkpoint, then again with the resume flag, and look for
     `res=1` and a start `c=` above zero (`grep 'res=1' log`; `verify.py` shows the `resumed` flag).
   - **DDP / rank**: `RANK=1 python script.py` writes no progress lines, `RANK=0` does
     (plain `logger.info` and `print` lines still appear on every rank, as before).
     Run under `torchrun --nproc_per_node=2` if the job can run on CPU (`gloo`).
   - **Framework callbacks**: run a few steps with the real trainer on a tiny model or
     dataset. If the adapter cannot be run, say that it was not exercised.
   - **Exceptions**: if cheap, make the loop raise once and confirm a `fail` is written
     (`verify.py` then warns about the `fail` and reads the job as `failed`: expected).
5. A one-off check without the script:

   ```bash
   python -c "
   import sys
   from precisionai.pace import parse_line
   recs = [r for r in map(parse_line, open('/tmp/pace-smoke.log')) if r]
   print(len(recs), 'records;', 'roots:', {r.scope for r in recs if r.root},
         '; ended:', {r.scope for r in recs if r.event.value == 'end'})
   "
   ```

Fix what the check finds and run again. Report honestly what could not be run.

## 6. Report

Short, in this order:

1. **Changed**: files and what changed in each, one line each.
2. **Root scope**: its id, unit and total, and why it is the root.
3. **What the scheduler can now estimate**: whole job or only some phases; which unit.
4. **Verified**: the exact command run and the verdict of `verify.py`. **Not run**: anything you
   could not run (GPU path, DDP, the framework callback, Docker build), and why.
5. **Remaining gaps**: phases with no known total (reported as unknown), work after the
   root, evaluation pauses inside a loop (the estimate is optimistic by their remaining
   total), early stopping (upper bound), lock files to refresh, framework wrappers with no
   callback hook.
6. **Optional follow-ups** you did not apply: `capture_logging()`, turning off the
   framework's own progress bar, `PACE_STREAM`/`PACE_BAR`, replacing more `print`s.

## OrchAI (optional)

If the repo has an `orch_starter/config.yaml` (OrchAI's job config), mention in the report
that an expected runtime can also be set there once OrchAI supports it, as a first guess
used only until Pace's own estimate is live. Do not add or invent config keys: this is a
pointer only.

## Gotchas

- `track()` writes `end`, not `fail`, when the body of the loop raises (a `break` and an
  exception look the same to a generator). Use `with progress(...) as run:` when failure
  must be recorded.
- Scope ids are unique per process: a second scope named `train` becomes `train-2`. A
  plan lists the id the phase *will* get; set `scope=` explicitly so they match.
- Scopes opened in new threads have no parent. Advance from the thread that collects results.
- A scope that never gets a `total` is reported as unknown; set it with `run.set_total(n)`
  when it becomes known.
- Text containing `[pace:` in a log message is rewritten; do not try to build trailers
  yourself.
- Pace's estimate is only as good as the total. Do not invent a total to make a loop
  estimable: an honest "unknown" beats a wrong number.
- If Pace is not available from the package index the repo installs from (a private
  mirror, for example), the dependency edit is still right; tell the user the install
  will fail until it is.
