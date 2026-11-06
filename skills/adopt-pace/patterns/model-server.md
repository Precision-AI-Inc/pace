# A model server started inside the job

The job starts a server (Ollama, a llama.cpp or vLLM server) as a subprocess, loads a model
into it, and sends it requests. The root counts the requests; starting the server and
loading the model are one-off `@timed` children:

```python
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from precisionai.pace import progress, timed

@timed("Start server")
def start_server() -> subprocess.Popen:
    log = Path(tempfile.mkdtemp()) / "server.log"
    with log.open("w") as handle:
        server = subprocess.Popen(["ollama", "serve"], stdout=handle, stderr=subprocess.STDOUT)
    try:
        wait_until_ready(timeout=120)  # raise, naming `log`, if it never answers
    except BaseException:
        stop(server)  # the caller's `finally` is not reached yet
        raise
    return server

def stop(server: subprocess.Popen) -> None:
    server.terminate()
    try:
        server.wait(timeout=30)
    except subprocess.TimeoutExpired:
        server.kill()
        server.wait()

@timed("Import model")
def import_model(path: str) -> None: ...

with progress(total=len(requests), description="Generate", unit="request", root=True) as run:
    server = start_server()
    try:
        import_model(args.model)
        with ThreadPoolExecutor(args.parallel) as pool:
            for future in as_completed(pool.submit(ask, r) for r in requests):
                answers.append(future.result())
                run.advance()
    finally:
        stop(server)
```

- Requests finish out of order, so advance as each one completes (`as_completed`), from the
  collecting thread. The rest follows `concurrent.md`.
- Open the root **before** starting the server, so startup counts as part of the job and
  the `@timed` steps are its children. The total is the request count; startup has no count.
  A step over within `PACE_MIN_INTERVAL` is one `STEP` line under the root; a slower one
  writes its start with the next line once the interval has passed (Pace starts no threads,
  so a slow import with nothing else reporting shows only when it ends).
- The server's own log is per-request chatter. Send it to a temporary file and name that
  file only if the server fails to start. Do not put it in the job's outputs or stdout.
- Pass `check=True` or check the return code for the model import, and always stop the
  server in a `finally`, so a failure still ends the scope as `fail`. A server that never becomes ready is
  stopped inside `start_server`, since the caller's `finally` does not exist yet.
- A server that runs until stopped and has no end of its own (vLLM's OpenAI-compatible
  server) gives no completion estimate: there is no total to count. Report only its startup
  (`@timed` or `phase`) and say in the report that the job has no estimate.

## Examples

- Tested, for OrchAI users (private repository): [`llm/serve_ollama.py`](https://github.com/Precision-AI-Inc/orchai-examples/blob/main/llm/serve_ollama.py)
  in OrchAI's example repo (Ollama from a GGUF file, `@timed` startup, concurrent requests).
- Pace: [`timed_steps.py`](https://github.com/Precision-AI-Inc/pace/blob/main/examples/timed_steps.py)
  (`@timed`), [`manual_progress.py`](https://github.com/Precision-AI-Inc/pace/blob/main/examples/manual_progress.py).
