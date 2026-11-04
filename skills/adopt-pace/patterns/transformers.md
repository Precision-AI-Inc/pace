# Hugging Face Transformers and TRL

<!-- verify against adapters: the exact constructor arguments of PaceCallback -->

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

- Dependency: `precisionai-pace[transformers]`.
- When a wrapper builds the trainer (Axolotl, LLaMA-Factory, torchtune, a company library),
  look for its hook for extra callbacks. If it has none, say so in the report rather than
  patching an installed package.
- `disable_tqdm=True` in `TrainingArguments` removes the Trainer's tqdm bar; offer it, do
  not apply it unasked.
- `max_steps = -1` with no epoch bound gives no total <!-- verify against adapters: what the callback does then -->;
  the Trainer still computes `max_steps` from epochs and dataset length, so this is rare.
- Transformers' log lines: see `logging.md` (`take_over=["transformers"]`).

## Examples

- Pace: [`transformers_example.py`](https://github.com/Precision-AI-Inc/pace/blob/main/examples/transformers_example.py).
- Tested: [`llm/sft_trl_lora.py`](https://github.com/Precision-AI-Inc/orchai-examples/blob/main/llm/sft_trl_lora.py)
  in OrchAI's example repo (TRL `SFTTrainer` with LoRA, `PaceCallback("Fine-tune")`, resume).
