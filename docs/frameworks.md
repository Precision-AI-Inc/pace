# Framework integrations

For PyTorch Lightning and Hugging Face Transformers (and TRL, which builds on the
Transformers trainer) Pace ships a callback, so you do not write the loop
instrumentation yourself. The callbacks live in their own modules and import their
framework only there; the core package still has no dependencies.

## Install

```bash
pip install "precisionai-pace[lightning]"      # PyTorch Lightning
pip install "precisionai-pace[transformers]"   # Hugging Face Transformers and TRL (adds accelerate)
pip install "precisionai-pace[all]"            # both
```

Each extra inherits its framework's own Python minimum, which may be higher than
the core's 3.11. The callbacks are tested with Lightning 2.6, Transformers 5.22 and
TRL 1.17 on CPU. The lower bounds in the extras are what the callbacks import; behaviour
on older versions is untested.

Both callbacks take the same arguments, all optional:

```python
PaceCallback(description="Train", *, scope=None, root=True, logger=None)
```

`description` names the training scope (its scope id is derived from it, `train` by
default), `scope` sets the id explicitly, `root=False` stops the callback marking its
scope as the job's root (use it when something else in the program is the root), and
`logger` sends output to a `Logger` other than the module-level one.

## PyTorch Lightning

```python
import lightning.pytorch as pl

from precisionai.pace.lightning import PaceCallback

trainer = pl.Trainer(max_epochs=3, callbacks=[PaceCallback()])
trainer.fit(model, datamodule=data)
```

The callback works with both `lightning.pytorch` and the older `pytorch_lightning`
package name. A runnable CPU example is in `examples/lightning_example.py`.

What it reports:

- One `root` scope for training: `trainer.global_step` against
  `trainer.estimated_stepping_batches`. That scope is what a scheduler uses as the job's
  estimate. See [Choosing a root scope](progress.md#choosing-a-root-scope). When the
  number of steps is not known in advance (an infinite dataloader, no `max_steps`), the
  total is reported as unknown.
- A child scope for each validation pass, nested under the training scope (`validation<train`
  in the trailer), counting batches. Each pass is its own scope: `validation`,
  `validation-2`, and so on. The sanity-check pass before training is not reported.
- `trainer.validate()`, `trainer.test()` or `trainer.predict()` called on their own,
  without a fit, report a top-level scope (`validation`, `test`, `predict`) that is
  not the root, so a baseline evaluation before training never marks the job
  complete. Only training is the root.
- On resume, the scope starts from the checkpoint's global step with the `res` flag, so the
  estimate measures the work left rather than the work already done
  (see [Resuming from a checkpoint](progress.md#resuming-from-a-checkpoint)).
- An exception during the run writes `fail`. Early stopping or `max_time` ends the
  scope normally with `end`, even though fewer steps ran than the total.

## Hugging Face Transformers and TRL

```python
from transformers import Trainer, TrainingArguments

from precisionai.pace.transformers import PaceCallback

trainer = Trainer(
    model=model,
    args=TrainingArguments(output_dir="out", max_steps=5000),
    train_dataset=train_ds,
    callbacks=[PaceCallback()],
)
trainer.train()
```

TRL's trainers (`SFTTrainer` and the others) subclass the Transformers `Trainer` and
take the same `callbacks=[...]` list, so the same callback is used; it is tested with
`SFTTrainer`. A runnable CPU example is in `examples/transformers_example.py`.

What it reports:

- One `root` scope for training: `state.global_step` against `state.max_steps`, from
  rank 0 only. With no step budget (`max_steps=-1` and no length), the total is unknown.
- A child scope for each evaluation or prediction pass (`evaluation`, `evaluation-2`,
  ...), counting batches, nested under the training scope. The Trainer tells the callback
  about a batch only after it has run, and has no hook before the first, so the scope opens
  already at `c=1`. Its rate and its "done in" time cover the batches after the first, and
  leave out how long the first one took. (Lightning has a hook before the first batch, so
  its validation scopes start at 0 and time every batch.)
- On `resume_from_checkpoint`, the scope starts from the restored global step with the `res` flag.
- **A crash writes no `fail`.** Transformers gives callbacks no hook when training
  raises, so the scope stays open until the process exits. A scheduler should treat
  process exit as the real end. To record the failure yourself, call the callback's
  `fail()` in your own error handling:

```python
callback = PaceCallback()
trainer = Trainer(model=model, args=args, train_dataset=train_ds, callbacks=[callback])
try:
    trainer.train()
except Exception:
    callback.fail()
    raise
```

## Both frameworks

- **Distributed runs.** Only global rank 0 writes progress (`trainer.is_global_zero`,
  `state.is_world_process_zero`), one record per scope rather than one per GPU, whatever
  `PACE_ALL_RANKS` says; see [Distributed runs](logging.md#distributed-runs-and-ranks).
- **Log lines.** The callbacks report progress. They do not change how the framework
  logs. To bring the framework's own log lines into the same format, call
  `capture_logging()`. Both frameworks print through loggers of their own, with their own
  console handlers: Lightning's lines would appear twice (`INFO: ...` and Pace's format),
  and Transformers' logger does not pass its records on to the root logger at all. Name them
  in `take_over`, after importing them:

  ```python
  import lightning.pytorch as pl
  import transformers

  from precisionai.pace import capture_logging

  capture_logging("INFO", take_over=["lightning", "transformers"])
  ```

  Use `"pytorch_lightning"` for the older package name. See
  [Libraries that print their own log lines](logging.md#standard-logging-pytorch-and-transformers).
  The same section covers vLLM, which needs `VLLM_CONFIGURE_LOGGING=0` instead.
- **The terminal bar.** The framework's own progress bar (Lightning's `RichProgressBar` or
  `TQDMProgressBar`, the Transformers `tqdm` bar) is separate. In a container two
  progress displays are noise; turn the framework's off
  (`enable_progress_bar=False` for the Lightning `Trainer`, `disable_tqdm=True` in
  `TrainingArguments`) and let Pace write the lines a scheduler reads.

## Libraries that take a `tqdm` class (vLLM)

Some libraries run a whole job inside one call and let you choose the progress-bar
class they use. vLLM's `LLM.generate(..., use_tqdm=...)` and `LLM.chat(..., use_tqdm=...)`
are the common case: you pass every prompt at once, and until the call returns there is
nothing to report from your own code. Pass Pace's stand-in for the class instead of
`True`:

```python
from vllm import LLM, SamplingParams

from precisionai.pace.tqdm import PaceTqdm

llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct")
outputs = llm.generate(prompts, SamplingParams(max_tokens=128), use_tqdm=PaceTqdm)
```

`PaceTqdm` draws nothing. It opens a Pace scope with the total and description the library
gives it, counts what the library counts, and ends the scope when the library closes it. It
does not import `tqdm`, so nothing is added to your install. It supports what such a
library calls (`update`, `n`, `total`, `refresh`, `close`, `reset`, `format_dict`, `postfix`,
`set_description`, `set_postfix`, the `with` statement and iterating an iterable); any other
`tqdm` method is accepted and ignored, so Pace never makes the library's call fail.
`tqdm` arguments that only change how a bar looks (`leave`, `file`, `ncols`, ...) are ignored,
and `disable=True` turns reporting off. A bar's postfix text, such as vLLM's tokens per
second, is not written.

What a scheduler sees:

- vLLM opens two bars per call: `Rendering prompts` (as the prompts are tokenised) and then
  `Processed prompts`, which is the generation itself. They become the scopes
  `rendering-prompts` and `processed-prompts`; a second call to `generate` makes
  `processed-prompts-2`, and so on. A scope opened inside `phase()` or `progress()` names it as
  its parent.
- Neither is a root by default. Without a root the estimate covers the current phase only and
  is marked partial; see [Choosing a root scope](progress.md#choosing-a-root-scope). When one
  `generate` call is the whole job, make the second bar the root. Mark one only: the
  first root a reader sees decides, and its end completes the job.

```python
class Generation(PaceTqdm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, root=kwargs.get("desc") == "Processed prompts", **kwargs)


outputs = llm.generate(prompts, params, use_tqdm=Generation)
```

- To set other options, use `functools.partial(PaceTqdm, scope="generate", logger=my_logger)`.
- Closing a bar before its total was reached (an abandoned loop, a library that stops early)
  writes `end` with the count reached, as `track()` does.

The tests replay the calls vLLM 0.32 makes on the class (`format_dict["elapsed"]`, assigning
`postfix`, `update(n)`, `refresh()` and `close()`). A library that needs more of `tqdm`
(class-level locks, `tqdm.pandas()`) is not covered. Hugging Face `datasets.map` uses its own
bar class internally and takes none from you.

## Without a callback

Any other training loop, or a framework without an adapter, uses the same functions
as everything else:

```python
from precisionai.pace import progress

with progress(total=total_steps, description="Train", unit="step", root=True) as run:
    for batch in loader:
        train_step(batch)
        run.advance()
```

See [Progress](progress.md).
