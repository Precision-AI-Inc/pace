# Resuming from a checkpoint

Pass the amount already done, and `resumed=True` only when the run really resumes. The
total stays the full total, not the remaining work:

```python
start_step = 0
if args.resume:
    state = load_checkpoint(args.resume)
    start_step = state["step"]

total_steps = args.epochs * len(loader)
with progress(
    total=total_steps,
    description="Train",
    unit="step",
    root=True,
    completed=start_step,
    resumed=start_step > 0,
) as run:
    step = start_step
    while step < total_steps:
        train_step(next_batch())
        step += 1
        run.advance()
```

- Use the unit the checkpoint stores (global step, or epoch index times batches per epoch).
- If the loop skips already-seen batches by slicing the loader, keep `total=` explicit so
  the total is not the length of the slice.
- The scheduler starts the remaining-work calculation from the resumed `c=` and does not
  average the rate across the restart.
- A framework callback restores the position from the checkpoint itself; pass nothing
  (see `lightning.md`, `transformers.md`).

## Examples

- Pace: [`resume.py`](https://github.com/Precision-AI-Inc/pace/blob/main/examples/resume.py)
  (`completed=`, `resumed=True`).
- Tested: [`detection/train_loop.py`](https://github.com/Precision-AI-Inc/orch-example-yolov5/blob/feat/pace-examples/detection/train_loop.py)
  in OrchAI's example repo (`--resume-from`).
