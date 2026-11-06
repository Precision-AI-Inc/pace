# A multi-phase pipeline

Preprocess, then train, then evaluate. Pick one of three shapes.

## a. A plan, when each phase has a count of its own

Declare the phases first, then run each as a counted scope **whose `scope=` id is the
name in the plan**:

```python
from precisionai.pace import plan, progress, track

plan([("prepare", 0.10), ("train", 0.80), ("evaluate", 0.10)])

with progress(total=len(raw_files), description="Prepare", unit="file", scope="prepare") as run:
    for path in raw_files:
        preprocess(path)
        run.advance()

for batch in track(loader, description="Train", unit="step", scope="train", total=total_steps):
    train_step(batch)

with progress(total=len(val_set), description="Evaluate", unit="sample", scope="evaluate") as run:
    for item in val_set:
        score(item)
        run.advance()
```

- Weights are each phase's share of the **runtime**, not of the number of items. They only
  matter for phases that have not started: the scheduler learns the seconds per weight from
  the phases already finished. Ask the user for rough shares; if nobody knows, leave the
  weights out (all phases count equally) and say they are guesses.
- The name in `plan(...)` becomes the scope id (lower case, anything else becomes `-`).
  Scope ids are unique per process (a second `train` becomes `train-2`), and a plan lists
  the id the phase *will* get, so set `scope=` explicitly. A mismatch is silent at runtime;
  `verify.py` reports it.
- There is no `root=True`: the job counts as complete when every planned phase has ended.
- A counted loop inside `with phase("Train"):` is fine: a phase has no rate of its own, so
  the scheduler uses the loop's estimate. If the phase holds several loops in a row (one per
  epoch), that estimate covers only the running loop and is a lower bound: count the whole
  stage in one scope. `phase()` on its own suits short, uncounted steps:

  ```python
  from precisionai.pace import phase, timed

  with phase("Load model"):
      model = load_model()

  @timed("Export")
  def export(model): ...
  ```

  A `phase` or `@timed` function that is not in the plan is just a timed step; one that is
  in the plan blocks the plan's estimate while it runs, which is harmless when it lasts seconds.

## b. One root over a common unit

When the whole job moves in one unit (N documents go through every stage, or the work can
be put in step-equivalents), a single root with a single total beats a plan:

```python
with progress(total=len(documents), description="Index", unit="doc", root=True) as run:
    for doc in documents:
        embed(chunk(parse(doc)))
        run.advance()
```

## c. A root that is only the main loop

When the main loop is nearly the whole runtime and the rest is seconds (load a model, write
a small result), make the loop the root and leave the rest as `phase()` or `logger.info`
lines. **The job is complete when the root ends**: work after the root's `end` (final
evaluation, export, upload) is invisible. If that work is more than a few percent of the
runtime, use (a) or (b).

## Examples

- Pace: [`phases_and_plan.py`](https://github.com/Precision-AI-Inc/pace/blob/main/examples/phases_and_plan.py)
  (`plan` and `phase`), [`timed_steps.py`](https://github.com/Precision-AI-Inc/pace/blob/main/examples/timed_steps.py)
  (`@timed`).
- Tested, for OrchAI users (private repository): [`detection/infer_ssdlite.py`](https://github.com/Precision-AI-Inc/orchai-examples/blob/main/detection/infer_ssdlite.py)
  in OrchAI's example repo (a plan, a `track` stage, a `phase`).
