# Getting started on OrchAI

OrchAI is Precision AI's job platform, and Pace is part of setting a repo up for it, next
to the Dockerfile and `config.yaml`. OrchAI reads the `[pace ...]` lines your job writes to
its log: from them **My Jobs** and **Tasks** show how far each job has got and when it
should finish, and the queue forecasts when the jobs behind it will start. A job without
Pace still runs, but shows no progress or time left ("Not reported"), makes the queue's
forecasts worse, and gets a "This job won't report progress" warning in Confirm Job.

This page is the short path for an OrchAI user. The rest of these docs explain each
piece in depth.

## Five steps

**1. Add the dependency.** Wherever your image installs its packages (usually
`requirements.txt`, which the Dockerfile installs from):

```text
precisionai-pace>=1.0.0                 # a plain loop
precisionai-pace[lightning]>=1.0.0      # PyTorch Lightning
precisionai-pace[transformers]>=1.0.0   # Hugging Face Trainer, TRL
```

Use a minimum version, never an exact pin. An OrchAI node rebuilds your image only when a
file the Dockerfile copies in changes, so an unchanged requirements file keeps whichever
Pace was installed first. To pick up a new release, raise the minimum. Pace needs Python
3.11 or later and has no dependencies of its own. `pyproject.toml`, Poetry, conda and lock
files: see the skill's
[dependency table](https://github.com/Precision-AI-Inc/pace/blob/main/skills/adopt-pace/patterns/dependencies.md).

**2. Report the main loop as one root scope.** The root stands for the whole job: when it
ends, OrchAI shows the job as finished. Pick the shape that matches your code.

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

PyTorch Lightning or a Hugging Face `Trainer` (TRL's `SFTTrainer` too): add the
callback and leave the loop alone. It is the root.

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
[Progress](progress.md), [Framework integrations](frameworks.md) and the tested
examples below.

**3. Check it locally.** Outside a terminal Pace writes at most one line every 15
seconds. `PACE_MIN_INTERVAL=0` lifts that, so a short run writes a line at each 1% of
its total (or each heartbeat). Then check the log with the checker that ships with the
[`adopt-pace` skill](#let-claude-code-do-it):

```bash
PACE_MIN_INTERVAL=0 python train.py --epochs 1 > /tmp/pace.log 2>&1
python ~/.claude/skills/adopt-pace/verify.py /tmp/pace.log   # or .claude/skills/adopt-pace/verify.py in the repo
```

It reports one root, a total and an `end` for every scope, or says what is missing.

**4. Optionally, give a first guess.** In `orch_starter/config.yaml`, set
`project.expected_runtime_minutes` (a multiple of 15, up to 10080). My Jobs shows it as a
"Guess", in amber, until Pace has measured the job. It is never a limit.

**5. Submit and watch.** Confirm Job no longer shows "This job won't report progress". In
My Jobs the Progress column reads "Estimating" for about the first minute, then the time
left and a finish time. Select the job for counts, rate and each step inside it.

## Let Claude Code do it

Pace ships a Claude Code skill, `adopt-pace`, that surveys your repository, shows you a
plan, makes the edits above, runs a short smoke test and checks the output. Install it
once for every repository on your machine:

```bash
src=$(mktemp -d) && git clone --depth 1 https://github.com/Precision-AI-Inc/pace.git "$src" &&
  mkdir -p ~/.claude/skills && rm -rf ~/.claude/skills/adopt-pace &&
  cp -r "$src/skills/adopt-pace" ~/.claude/skills/ && rm -rf "$src"
```

Run the same command again to update it: it removes the old copy first, so no stale
files are left behind.

Then, in your repository, ask Claude Code to "add Pace to this repo". It asks before it
edits anything. Other ways to install it:
[skills/README.md](https://github.com/Precision-AI-Inc/pace/blob/main/skills/README.md).

## What the example jobs taught

The [OrchAI example jobs](https://github.com/Precision-AI-Inc/orchai-examples) (a private
repository: ask for access if the links do not open) each run one pattern end to end on
OrchAI, with a config to submit it:

| Your job | Pattern | Example |
|---|---|---|
| Your own PyTorch training loop | root `progress`, validation as a child scope, `@timed`, resume | [`detection/train_loop.py`](https://github.com/Precision-AI-Inc/orchai-examples/blob/main/detection/train_loop.py) |
| The same loop on several GPUs (`torchrun`) | progress from rank 0, no guard | `train_loop.py` with `config_detection_train_ddp.yaml` |
| PyTorch Lightning | Lightning `PaceCallback` | [`detection/train_lightning.py`](https://github.com/Precision-AI-Inc/orchai-examples/blob/main/detection/train_lightning.py) |
| Hugging Face / TRL fine-tune | Transformers `PaceCallback` | [`llm/sft_trl_lora.py`](https://github.com/Precision-AI-Inc/orchai-examples/blob/main/llm/sft_trl_lora.py) |
| Inference then metrics | `plan`, a `track` stage, a `phase` | [`detection/infer_ssdlite.py`](https://github.com/Precision-AI-Inc/orchai-examples/blob/main/detection/infer_ssdlite.py) |
| vLLM batch generation | `progress` advanced a chunk at a time | [`llm/generate_vllm.py`](https://github.com/Precision-AI-Inc/orchai-examples/blob/main/llm/generate_vllm.py), [`vlm/caption_vllm.py`](https://github.com/Precision-AI-Inc/orchai-examples/blob/main/vlm/caption_vllm.py) |
| An Ollama server the job starts | `@timed` start-up, requests counted as they finish | [`llm/serve_ollama.py`](https://github.com/Precision-AI-Inc/orchai-examples/blob/main/llm/serve_ollama.py) |

What they taught:

- **One root, and nothing big after it.** Work after the root ends (a final evaluation,
  an export) is not counted. If it is more than a few percent of the run, use a
  [plan](progress.md#phases-and-a-plan) instead.
- **Count steps, not epochs.** One scope over `epochs * len(loader)` steps gives one
  steady rate. Ten epochs give the estimate ten data points; only a scope of more than 100
  units gets the start-up correction (a slow first few steps while the model loads or
  compiles do not drag the estimate down for the rest of the run).
- **Several GPUs: no `if rank == 0:`.** Pace writes progress from rank 0 by itself.
  Instrument the script, not `torchrun`. Keep your own log lines and checkpoint saves on
  rank 0.
- **vLLM: loop over chunks.** One `llm.generate(...)` call over every prompt reports
  nothing until it returns. Loop over chunks with `use_tqdm=False` and advance by the
  chunk size; a smaller chunk gives a smoother estimate.
- **Keep `tqdm` where it is not the main progress.** Replace the bar that tracks the job;
  a download bar can stay. Turn off a framework's own bar if you want a quieter log
  (`enable_progress_bar=False` in Lightning, `disable_tqdm=True` in `TrainingArguments`,
  `TQDM_DISABLE=1` for libraries that draw bars while loading).
- **No known total, no estimate.** A download of unknown size shows "No total" rather than
  a made-up percentage. Find the real count if there is one; otherwise leave it unknown.
- **Short demo jobs do not reach an estimate.** Pace writes a line at most every 15 seconds,
  and the estimate needs a minute of observations and three measured intervals (longer when
  the first steps are slow). A job that finishes within a couple of minutes may go straight
  from "Estimating" to "Finished". Give a demo job at least five minutes of work (a
  `custom_args` value such as `epochs` or `passes`) to see a live estimate.

## When something looks wrong

| You see | Why, and what to do |
|---|---|
| Confirm Job still says "This job won't report progress" | Any one of these clears it: `precisionai-pace` in `requirements*.txt`/`.in`, `requirements/*.txt`, `pyproject.toml`, `setup.cfg`/`setup.py`, `Pipfile`, `environment*.yml`/`.yaml`, the Dockerfile, or a file the Dockerfile installs from with `-r`/`-c` (pip) or `-f` (conda); or `import precisionai.pace` in the script `container.command` runs (up to 3 entry-point files are read, not what they import). Adding it to the file your Dockerfile installs from is enough. Submitting still works either way. |
| "Not yet" for a long time | Pace is installed but the root has not started, for example while a download runs first. |
| "Estimating" never changes to a time | The scope has no total, or the run is too short (see above). |
| The job shows finished while it is still running | The root ended before the last stage. Move the root, or use a plan. |
| A new Pace release does not reach the job | The node reused the cached image. Raise the minimum version in your requirements. |
