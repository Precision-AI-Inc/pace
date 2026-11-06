# Hugging Face Transformers and TRL

`PaceCallback(description="Train", *, logger=None, scope=None, root=True)`, and `fail()` for your own
error handling.

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

`SFTTrainer`, `DPOTrainer` and the other TRL trainers take the same `callbacks=[...]`, or
`trainer.add_callback(PaceCallback())` after construction. The callback reports
`state.global_step` against `state.max_steps`, rank 0 only.

- Dependency: `precisionai-pace[transformers]>=1.0.0`.
- When a wrapper builds the trainer (Axolotl, LLaMA-Factory, torchtune, a company library),
  look for its hook for extra callbacks. If it has none, say so in the report rather than
  patching an installed package.
- `disable_tqdm=True` in `TrainingArguments` removes the Trainer's tqdm bar; offer it, do
  not apply it unasked.
- `state.max_steps` of 0 or less (no step budget) makes the total unknown, which gives no
  estimate; the Trainer computes `max_steps` from epochs and dataset length, so this is rare.
- The `Trainer` has no hook for an exception, so a crash writes no `fail` record: the log
  just stops. If the script already catches errors around `trainer.train()`, offer
  `callback.fail()` there (keep a reference to the callback); do not add a new `try`.
- Transformers' log lines: see `logging.md` (`take_over=["transformers"]`).

## Examples

- Pace: [`transformers_example.py`](https://github.com/Precision-AI-Inc/pace/blob/main/examples/transformers_example.py).
- Tested, for OrchAI users (private repository): [`llm/sft_trl_lora.py`](https://github.com/Precision-AI-Inc/orchai-examples/blob/main/llm/sft_trl_lora.py)
  in OrchAI's example repo (TRL `SFTTrainer` with LoRA, `PaceCallback("Fine-tune")`, resume).
