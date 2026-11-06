# Running Pace on a cluster or scheduler

A job scheduler or log collector reads the `[pace ...]` lines your job writes to its log,
and from them can show how far each job has got, when it should finish, and when the jobs
queued behind it will start. This page is the short path for a job that runs in a
container or under a scheduler, with what running real jobs that way has taught. The rest
of these docs explain each piece in depth.

## Five steps

**1. Add the dependency.** Wherever your image installs its packages (usually
`requirements.txt`, which the Dockerfile installs from):

```text
precisionai-pace>=1.0.0                 # a plain loop
precisionai-pace[lightning]>=1.0.0      # PyTorch Lightning
precisionai-pace[transformers]>=1.0.0   # Hugging Face Trainer, TRL
```

Use a minimum version, never an exact pin. A build cache, or a node that keeps the images
it has built, reuses an image until a file the Dockerfile copies in changes, so an
unchanged requirements file keeps whichever Pace was installed first. To pick up a new
release, raise the minimum. If the Dockerfile installs packages on a `RUN pip install`
line itself, quote the requirement there:

```dockerfile
RUN pip install --no-cache-dir "precisionai-pace>=1.0.0"
```

Unquoted, the shell reads `>` as a redirect: pip installs any version and the minimum is
lost. Pace needs Python 3.11 or later and has no dependencies of its own.
`pyproject.toml`, Poetry, conda and lock files: see the skill's
[dependency table](https://github.com/Precision-AI-Inc/pace/blob/main/skills/adopt-pace/patterns/dependencies.md).

**2. Report the main loop as one root scope.** The root stands for the whole job: when it
ends, a reader treats the job as finished. Mark exactly one per job. Pick the shape that
matches your code.

Your own loop:

```python
from precisionai.pace import logger, progress

total_steps = args.epochs * len(loader)
with progress(total=total_steps, description="Train", unit="step", root=True) as run:
    for epoch in range(args.epochs):
        for batch in loader:
            train_step(batch)
            run.advance()
logger.success("Training finished")
```

PyTorch Lightning or a Hugging Face `Trainer` (TRL's `SFTTrainer` and the other TRL
trainers too): add the callback and leave the loop alone. It is the root.

```python
import lightning.pytorch as pl
from precisionai.pace.lightning import PaceCallback

trainer = pl.Trainer(max_epochs=args.epochs, callbacks=[PaceCallback("Train")])
```

```python
from transformers import Trainer
from precisionai.pace.transformers import PaceCallback

trainer = Trainer(model=model, args=training_args, train_dataset=train_ds,
                  callbacks=[PaceCallback("Train")])
```

Inference, several phases, vLLM, a model server, or a total you only learn later: see
[Progress](progress.md), [Framework integrations](frameworks.md), the lessons below and
the [examples](https://github.com/Precision-AI-Inc/pace/blob/main/examples/README.md).

**3. Make the output reach the log.** Pace flushes every line it writes. Your own `print`
calls are block-buffered when stdout is not a terminal, so set `PYTHONUNBUFFERED=1` in the
image (`ENV PYTHONUNBUFFERED=1`) if the job prints.

**4. Check a short run locally.** Outside a terminal Pace writes at most one line every 15
seconds. `PACE_MIN_INTERVAL=0` lifts that, so a short run writes a line at each 1% of its
total (or each heartbeat). Then check the log with `verify.py`, the checker that ships
with the [`adopt-pace` skill](#let-a-coding-agent-do-it):

```bash
PACE_MIN_INTERVAL=0 python train.py --epochs 1 > /tmp/pace.log 2>&1
python ~/.claude/skills/adopt-pace/verify.py /tmp/pace.log   # or .claude/skills/adopt-pace/verify.py in the repo
```

It reports one root, a total and an `end` for every scope, or says what is missing.

**5. Submit and watch.** For about the first minute a reader has no estimate yet; then
it has the time left and a finish time.

## What running real jobs taught

- **One root, and nothing big after it.** Work after the root ends (a final evaluation,
  an export) is not counted. If it is more than a few percent of the run, use a
  [plan](progress.md#phases-and-a-plan) instead.
- **Count steps, not epochs.** One scope over `epochs * len(loader)` steps gives one
  steady rate. Ten epochs give the estimate ten data points; only a scope of more than 100
  units gets the start-up correction (a slow first few steps while the model loads or
  compiles do not drag the estimate down for the rest of the run).
- **Several GPUs: no `if rank == 0:`.** Pace writes progress from rank 0 by itself.
  Instrument the script, not `torchrun`. Keep your own log lines and checkpoint saves on
  rank 0. See [`distributed.py`](https://github.com/Precision-AI-Inc/pace/blob/main/examples/distributed.py).
- **vLLM batch generation: loop over chunks.** One `llm.generate(...)` call over every
  prompt reports nothing until it returns. Open one root over all the prompts, loop over
  chunks with `use_tqdm=False`, and advance by the chunk size; a smaller chunk gives a
  smoother estimate. A single call can take `use_tqdm=PaceTqdm` instead; see
  [Libraries that take a `tqdm` class](frameworks.md#libraries-that-take-a-tqdm-class-vllm).
- **A model server the job starts (Ollama, a vLLM server).** Open the root before starting
  it, make start-up and model loading `@timed` steps inside it, and count requests as they
  finish. A server with no end of its own gives no estimate.
- **TRL fine-tunes need nothing extra.** `SFTTrainer`, `DPOTrainer` and the other TRL
  trainers take the Transformers `PaceCallback` like any `Trainer`.
- **Log collectors read stdout and stderr.** Moving Pace to stderr does not hide it, and a
  library's bar on stderr still lands in the job's log. Replace the bar that tracks the
  job; a download bar can stay. Turn off a framework's own bar if you want a quieter log
  (`enable_progress_bar=False` in Lightning, `disable_tqdm=True` in `TrainingArguments`),
  and set `TQDM_DISABLE=1` (in the image's `ENV`) when Pace has replaced your tqdm and
  libraries still draw bars while loading.
- **No known total, no estimate.** A download of unknown size shows no total rather than
  a made-up percentage. Find the real count if there is one; otherwise leave it unknown.
- **Short demo jobs do not reach an estimate.** Pace writes a line at most every 15 seconds,
  and the estimate needs a minute of observations and three measured intervals (longer when
  the first steps are slow). A job that finishes within a couple of minutes may go straight
  from estimating to finished. Give a demo job at least five minutes of work to see a live
  estimate, or set `PACE_MIN_INTERVAL` lower (for example `1`) so it writes more often.

## Let a coding agent do it

Pace ships a skill, `adopt-pace`, that surveys your repository, shows you a plan, makes
the edits above, runs a short smoke test and checks the output. Install it once for every
repository on your machine (Claude Code):

```bash
src=$(mktemp -d) && git clone --depth 1 https://github.com/Precision-AI-Inc/pace.git "$src" &&
  mkdir -p ~/.claude/skills && rm -rf ~/.claude/skills/adopt-pace &&
  cp -r "$src/skills/adopt-pace" ~/.claude/skills/ && rm -rf "$src"
```

Run the same command again to update it: it removes the old copy first, so no stale
files are left behind.

Then, in your repository, ask Claude Code to "add Pace to this repo". It asks before it
edits anything. Other ways to install it, and how to use it with other coding agents:
[skills/README.md](https://github.com/Precision-AI-Inc/pace/blob/main/skills/README.md).

## When something looks wrong

| You see | Why, and what to do |
|---|---|
| No progress for a long time after the job starts | The root has not started yet, for example while a download runs first. Or the job's own output is buffered: set `PYTHONUNBUFFERED=1`. |
| An estimate never appears | The scope has no total, or the run is too short (see above). |
| The job shows finished while it is still running | The root ended before the last stage. Move the root, or use a plan. |
| A new Pace release does not reach the job | The image was reused from cache. Raise the minimum version in your requirements. |
| Library progress bars clutter the log | See "Log collectors read stdout and stderr" above. |
