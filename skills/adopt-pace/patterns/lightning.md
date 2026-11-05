# PyTorch Lightning

`PaceCallback(description="Train", *, logger=None, scope=None, root=True)` works with `lightning.pytorch`
and the older `pytorch_lightning`. Add the callback. Do not touch the training loop. It reports `trainer.global_step` against
`trainer.estimated_stepping_batches` as the root scope, with validation as a child scope,
and restores the resume position from the checkpoint:

```python
import lightning.pytorch as pl
from precisionai.pace.lightning import PaceCallback

trainer = pl.Trainer(
    max_epochs=args.epochs,
    callbacks=[PaceCallback(), *existing_callbacks],
)
trainer.fit(model, datamodule=dm, ckpt_path=args.resume)
```

- Dependency: `precisionai-pace[lightning]>=0.1.3`.
- If the trainer is built from a config file or `LightningCLI`, add the callback where
  callbacks are declared (for `LightningCLI`, `trainer.callbacks` in the YAML with
  `class_path: precisionai.pace.lightning.PaceCallback`, plus `init_args: {description: Fit}`
  for a label other than `"Train"`).
- Lightning's own progress bar writes to stderr and does not conflict. Leave it unless the
  user wants the log quieter (`enable_progress_bar=False`).
- `trainer.estimated_stepping_batches` is infinite when neither `max_steps` nor `max_epochs`
  bounds the run (`max_epochs=-1`), or the dataloader has no length. The callback then
  reports a scope with no total, which gives no estimate: say so in the report.
- Pass `root=False` only when the script already opens a root scope of its own around
  `trainer.fit(...)`.
- Lightning's log lines: see `logging.md` (`take_over=["lightning"]`).

## Examples

- Pace: [`lightning_example.py`](https://github.com/Precision-AI-Inc/pace/blob/main/examples/lightning_example.py).
- Tested: [`detection/train_lightning.py`](https://github.com/Precision-AI-Inc/orchai-examples/blob/main/detection/train_lightning.py)
  in OrchAI's example repo (`PaceCallback("Train")`, `ckpt_path` resume, own bar off).
