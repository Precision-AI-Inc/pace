# Distributed training (DDP, Accelerate, torchrun)

Instrument the script that runs on every rank, and write the loop exactly as for one
process. Pace reads the rank from the environment, so only rank 0 writes progress lines and
the other ranks run the same calls silently:

```python
with progress(total=total_steps, description="Train", unit="step", root=True) as run:
    for batch in loader:        # per-rank loader: len(loader) is already per rank
        train_step(batch)
        run.advance()
```

- Count optimizer steps (or batches per rank), which advance in lockstep. If the unit is
  samples, advance by `batch_size * world_size` and use the global sample count as the total.
- `logger.info(...)` is not filtered by rank, and neither is anything else the job prints.
  Keep the existing `if rank == 0:` / `accelerator.is_main_process` guards around the
  messages that had them.
- The entry point is the script, not the launcher: instrument `train.py`, not `torchrun` /
  `accelerate launch`. Set nothing for ranks; `PACE_ALL_RANKS=1` exists only for debugging.
- Check: `RANK=1 python script.py` writes no progress lines, `RANK=0` does. Under
  `torchrun --nproc_per_node=2` the job runs on CPU with `gloo`.

## Examples

- Pace: [`distributed.py`](https://github.com/Precision-AI-Inc/pace/blob/main/examples/distributed.py).
- Tested, for OrchAI users (private repository): [`detection/train_loop.py`](https://github.com/Precision-AI-Inc/orchai-examples/blob/main/detection/train_loop.py)
  in OrchAI's example repo (DDP under `torchrun`).
