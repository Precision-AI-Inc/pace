# A plain loop, replacing tqdm

The loop is the job, so it is the root. Before:

```python
from tqdm import tqdm

pbar = tqdm(loader, desc="Train", total=len(loader))
for step, batch in enumerate(pbar):
    loss = train_step(batch)
    pbar.set_postfix(loss=loss)
```

After:

```python
from precisionai.pace import logger, track

for step, batch in enumerate(track(loader, description="Train", unit="batch", root=True)):
    loss = train_step(batch)
    if step % 100 == 0:
        logger.info(f"step {step} loss {loss:.4f}")
```

| tqdm | Pace |
|---|---|
| `tqdm(it, desc="x", total=n)`, `trange(n)` | `track(it, description="x", total=n)`, `track(range(n))` |
| `pbar.update(k)` | `run.advance(k)` on a `progress(...)` handle |
| `pbar.set_postfix(...)`, `pbar.set_description(...)` | no equivalent: log a line every N steps with `logger.info`, or drop it. Metrics belong in MLflow or similar. |
| `tqdm.write(msg)` | `logger.info(msg)` |
| `leave=`, `ncols=`, `position=`, `disable=` | not needed |

- `track()` takes `total` from `len(iterable)`. A generator or `itertools.islice` has no
  length: pass `total=`.
- `track()` writes `end` (not `fail`) when the loop body raises, because Python hands it the
  same `GeneratorExit` for `break` and for an exception. If a crash must show as failed, use
  the handle form:

  ```python
  with progress(total=len(loader), description="Train", unit="batch", root=True) as run:
      for batch in loader:
          train_step(batch)
          run.advance()
  ```

- A `tqdm` that is not the job's main indicator (a download bar, a one-off preprocessing
  bar) stays as it is. Remove a `tqdm` import the change left unused, but leave the
  dependency declaration (the repo may use it elsewhere) and say so in the report.

## Examples

- Pace: [`track_loop.py`](https://github.com/Precision-AI-Inc/pace/blob/main/examples/track_loop.py)
  (`track`), [`manual_progress.py`](https://github.com/Precision-AI-Inc/pace/blob/main/examples/manual_progress.py)
  (`progress`).
