# A job that runs on OrchAI

OrchAI reads the job's log, shows progress in My Jobs and Tasks, and decides "finished"
from the root scope. Everything in `SKILL.md` applies; these are the OrchAI specifics,
learned building the example jobs. The user-facing version is Pace's
[`docs/orchai.md`](https://github.com/Precision-AI-Inc/pace/blob/main/docs/orchai.md):
point the user at it in the report.

- **Dependency: a minimum, never a pin.** `precisionai-pace>=1.0.0` (with the extra). A node
  rebuilds the image only when a file the Dockerfile copies in changes, so an unchanged
  requirements file keeps the Pace installed first; raising the minimum is what picks up a
  new release. Put it in the file the Dockerfile installs from (`requirements.txt` when it
  runs `pip install -r`); a base image with only system Python (Ollama) takes
  `python3 -m pip install --break-system-packages "precisionai-pace>=1.0.0"` in the
  Dockerfile.
- **What Confirm Job looks for.** A mention of `precisionai-pace` in `requirements*.txt`/`.in`,
  `requirements/*.txt`, `pyproject.toml`, `setup.cfg`/`setup.py`, `Pipfile`,
  `environment*.yml`/`.yaml`, the Dockerfile, or a file the Dockerfile hands pip `-r`/`-c`
  or conda `-f`; or `import precisionai.pace` in the scripts `container.command` runs (up
  to 3 entry-point files, not the modules they import). Without
  one it warns "This job won't report progress" (a warning: submitting still works). The
  dependency edit in the file the image installs from clears it.
- **Which script.** `container.command` in `orch_starter/config*.yaml` is the job. A repo
  with several configs has several jobs: instrument each one the user names. The command
  runs from `/src` (the repository) through `/bin/sh -c`; do not change it.
- **Expected runtime.** Offer, do not set, `project.expected_runtime_minutes` in the config:
  a first guess at the run's length in minutes, a multiple of 15 from 15 to 10080. My Jobs
  shows it as a "Guess" until Pace has measured the job; it is never a limit. Estimate it
  from what the survey found (epochs, steps, any timings) and let the user confirm, or leave
  it out.
- **Length in `custom_args`.** If the job's length is a flag (`--epochs`, `--steps`,
  `--passes`), say a quick check is a one-line change there. A run of a couple of minutes
  may go from "Estimating" straight to "Finished": a demo needs about five minutes of work
  to show a live estimate.
- **Units.** Count optimizer steps, images, prompts or requests, not epochs: a scope of
  more than 100 units gets a start-up correction (slow first steps while the model
  loads or compiles stop dragging the estimate down), and more units give more updates.
- **Library bars.** OrchAI reads the whole log, so a framework's own bar is noise there.
  Offer `enable_progress_bar=False` (Lightning), `disable_tqdm=True` (`TrainingArguments`),
  or `TQDM_DISABLE=1` in the Dockerfile's `ENV` for libraries that draw bars while loading
  (vLLM, `datasets`). Leave a download bar that is not the main indicator.
- **Output.** Pace flushes every line itself. The repo's own `print` calls are block-buffered
  in a container unless `PYTHONUNBUFFERED=1` is set; offer it in the Dockerfile's `ENV` when
  the job prints.
- **After submitting**, tell the user what to look for: Confirm Job without the warning,
  then My Jobs' Progress column ("Estimating" for about a minute, then time left and a
  finish time). "Not yet" means the root has not started (a download running first);
  "No total" means a scope without a count.

## Examples

Each job in OrchAI's example repo has a config in `orch_starter/` beside a script:
[`README.md`](https://github.com/Precision-AI-Inc/orchai-examples/blob/main/README.md)
lists which config runs which pattern. The repo is private to Precision AI; the user may
need access.

- Images and dependencies:
  [`detection/Dockerfile`](https://github.com/Precision-AI-Inc/orchai-examples/blob/main/detection/Dockerfile),
  [`llm/Dockerfile`](https://github.com/Precision-AI-Inc/orchai-examples/blob/main/llm/Dockerfile)
  (`TQDM_DISABLE=1`, `VLLM_CONFIGURE_LOGGING=0`),
  [`llm/Dockerfile.ollama`](https://github.com/Precision-AI-Inc/orchai-examples/blob/main/llm/Dockerfile.ollama)
  (system Python).
- A multi-GPU config:
  [`orch_starter/config_detection_train_ddp.yaml`](https://github.com/Precision-AI-Inc/orchai-examples/blob/main/orch_starter/config_detection_train_ddp.yaml).
