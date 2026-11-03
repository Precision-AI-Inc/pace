# Framework integrations

For PyTorch Lightning and Hugging Face Transformers (and TRL, which builds on the
Transformers trainer) Pace ships a callback, so you do not write the loop
instrumentation yourself. The callbacks live in their own modules and import their
framework only there; the core package still has no dependencies.

<!-- verify against adapters: the adapters were not merged when this page was written. Everything below is taken from the "Framework adapters" section of design.md and has not been run. -->

## Install

```bash
pip install "precisionai-pace[lightning]"      # PyTorch Lightning
pip install "precisionai-pace[transformers]"   # Hugging Face Transformers and TRL
```

<!-- verify against adapters: extra names are the ones in design.md ("Extras (planned)"); pyproject.toml has no [lightning] or [transformers] extra yet, and design.md also lists [all]. -->

Each extra inherits its framework's own Python minimum, which may be higher than
the core's 3.8.

## PyTorch Lightning

```python
import lightning.pytorch as pl

from precisionai.pace.lightning import PaceCallback

trainer = pl.Trainer(max_epochs=3, callbacks=[PaceCallback()])
trainer.fit(model, datamodule=data)
```

<!-- verify against adapters: import path and constructor with no arguments, per design.md. The Lightning package name in the example (`lightning.pytorch` vs `pytorch_lightning`) depends on what the adapter supports. -->

What it reports:

- One `root` scope for the training run: `trainer.global_step` against
  `trainer.estimated_stepping_batches`. That scope is what a scheduler uses as the job's
  estimate. See [Choosing a root scope](progress.md#choosing-a-root-scope).
- A child scope for each validation run, nested under the training scope
  (`p=` in the trailer).
- On resume, the scope starts from the checkpoint's global step, so the estimate
  measures the work left rather than the work already done
  (see [Resuming from a checkpoint](progress.md#resuming-from-a-checkpoint)).

<!-- verify against adapters: that the training scope is written with root=1, the validation scope nests under it, and the scope ids. -->

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
take the same `callbacks=[...]` list, so the same callback is used.

<!-- verify against adapters: that TRL needs nothing beyond callbacks=[PaceCallback()], and that the Trainer example runs (a real model and dataset are needed). -->

What it reports: `state.global_step` against `state.max_steps`, from rank 0 only.

<!-- verify against adapters: whether this scope is marked root; whether evaluation is a child scope; the behaviour when max_steps is -1 (no total), and on resume_from_checkpoint. -->

## Both frameworks

- **Distributed runs.** Only rank 0 writes progress, one record per scope rather than
  one per GPU; see [Distributed runs](logging.md#distributed-runs-and-ranks).
- **Log lines.** The callbacks report progress. They do not change how the framework
  logs. To bring the framework's own log lines into the same format, call
  `capture_logging()` and, for Transformers, switch off its own handler; see
  [Standard logging](logging.md#standard-logging-pytorch-and-transformers).
- **The terminal bar.** The framework's own progress bar (Lightning's `RichProgressBar` or
  `TQDMProgressBar`, the Transformers `tqdm` bar) is separate. In a container two
  progress displays are noise; turn the framework's off
  (`enable_progress_bar=False` for the Lightning `Trainer`, `disable_tqdm=True` in
  `TrainingArguments`) and let Pace write the lines a scheduler reads.

<!-- verify against adapters: the two option names above are the frameworks' own and should be re-checked against the versions the adapters support. -->

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
