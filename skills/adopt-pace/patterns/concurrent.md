# A concurrent evaluation harness

Count finished requests, from the thread that collects the results. Scopes are thread safe,
and a scope opened inside a worker thread would have no parent anyway:

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

from precisionai.pace import progress

with ThreadPoolExecutor(max_workers=args.workers) as pool, progress(
    total=len(cases), description="Evaluate", unit="request", root=True
) as run:
    futures = [pool.submit(score, case) for case in cases]
    for future in as_completed(futures):
        results.append(future.result())
        run.advance()
```

- `progress` is entered after the pool and so exits first: if `future.result()` raises, the
  scope is written as `fail`.
- Count a request once, when it is finally done. Retries are not extra requests.
- With `ProcessPoolExecutor`, advance in the parent; do not create scopes in the workers.
- With asyncio, `run.advance()` after each `await` of `asyncio.as_completed(...)` works the
  same.
- Several harnesses side by side in one process (rare): one root over their sum, not one
  root each.
- A model server the job starts itself is a variant of this: see `model-server.md`.

## Examples

- Pace: [`manual_progress.py`](https://github.com/Precision-AI-Inc/pace/blob/main/examples/manual_progress.py)
  (`progress`, results advanced as they arrive).
- Tested: [`llm/serve_ollama.py`](https://github.com/Precision-AI-Inc/orchai-examples/blob/main/llm/serve_ollama.py)
  in OrchAI's example repo (concurrent requests, out of order).
