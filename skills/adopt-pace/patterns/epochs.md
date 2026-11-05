# Epochs and batches

Do not make one `track()` per epoch and call each a root. One scope with the whole
denominator gives the scheduler one smooth rate:

```python
total_steps = args.epochs * len(loader)
with progress(total=total_steps, description="Train", unit="step", root=True) as run:
    for epoch in range(args.epochs):
        for batch in loader:
            train_step(batch)
            run.advance()
        logger.info(f"epoch {epoch + 1}/{args.epochs} done")
```

- Count steps, not epochs, even when the loop reads naturally as `for epoch in ...`: an
  epoch-unit root of 10 gives the estimate ten data points, and only a scope of more than
  100 units gets a start-up correction (slow first steps while the model loads or compiles
  stop dragging the rate down).
- With a `max_steps` cap, the total is `min(args.max_steps, args.epochs * len(loader))`.
- With early stopping, the estimate is an upper bound: say so in the report.
- With gradient accumulation, advance per micro-batch (what the loop iterates) and keep the
  total in the same unit.
- Validation inside the loop stays inside the root, either uncounted or as a `progress`
  opened inside it (a child scope). A pause that recurs is smoothed out of the rate, so the
  estimate is a little optimistic by the evaluation time still to come; report it as a known
  gap. Do not give it a root of its own.

## Examples

- Pace: [`epochs_and_batches.py`](https://github.com/Precision-AI-Inc/pace/blob/main/examples/epochs_and_batches.py)
  (one root over every step), [`nested_scopes.py`](https://github.com/Precision-AI-Inc/pace/blob/main/examples/nested_scopes.py)
  (child scopes).
- Tested: [`detection/train_loop.py`](https://github.com/Precision-AI-Inc/orchai-examples/blob/main/detection/train_loop.py)
  in OrchAI's example repo (epochs times batches, each validation pass a child scope).
