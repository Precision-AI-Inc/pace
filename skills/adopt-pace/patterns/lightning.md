# PyTorch Lightning

<!-- verify against adapters: the exact constructor arguments of PaceCallback, and whether it accepts the legacy pytorch_lightning import as well as lightning.pytorch -->

Add the callback. Do not touch the training loop. It reports `trainer.global_step` against
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

- Dependency: `precisionai-pace[lightning]`.
- If the trainer is built from a config file or `LightningCLI`, add the callback where
  callbacks are declared (for `LightningCLI`, `trainer.callbacks` in the YAML with
  `class_path: precisionai.pace.lightning.PaceCallback`) <!-- verify against adapters: class_path/init_args form -->.
- Lightning's own progress bar writes to stderr and does not conflict. Leave it unless the
  user wants the log quieter (`enable_progress_bar=False`).
- `trainer.estimated_stepping_batches` is infinite when neither `max_steps` nor `max_epochs`
  bounds the run <!-- verify against adapters: how the callback reports this -->; say so in
  the report if the trainer is unbounded.
- Lightning's log lines: see `logging.md` (`take_over=["lightning"]`).

## Examples

- Pace: [`lightning_example.py`](https://github.com/Precision-AI-Inc/pace/blob/main/examples/lightning_example.py).
- Tested: [`detection/train_lightning.py`](https://github.com/Precision-AI-Inc/orch-example-yolov5/blob/feat/pace-examples/detection/train_lightning.py)
  in OrchAI's example repo (`PaceCallback("Train")`, `ckpt_path` resume, own bar off).
