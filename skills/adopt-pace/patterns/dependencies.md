# Dependency declarations

Add `precisionai-pace` everywhere the repo installs its own dependencies, as a minimum
version: `precisionai-pace>=1.0.0`. The extras (`[lightning]`, `[transformers]`) go in the
same brackets, `precisionai-pace[lightning]>=1.0.0`. Pace needs Python 3.11 or later.

| Where | Entry |
|---|---|
| `requirements*.txt` | `precisionai-pace>=1.0.0` (or `precisionai-pace[lightning]>=1.0.0`) |
| `pyproject.toml`, PEP 621 | `dependencies = [..., "precisionai-pace>=1.0.0"]` (or the optional group the job installs) |
| `pyproject.toml`, Poetry | `precisionai-pace = ">=1.0.0"`, or `precisionai-pace = { version = ">=1.0.0", extras = ["lightning"] }` |
| `setup.py`, `setup.cfg` | `install_requires` list |
| conda `environment.yml` | under the `pip:` list, not as a conda package |
| `Dockerfile`: `pip install a b c` | add it to that line, quoted: `"precisionai-pace>=1.0.0"` (unquoted, the shell reads `>` as a redirect and pip gets no minimum). If the Dockerfile only runs `pip install -r requirements.txt`, the requirements file is enough. |
| `uv.lock`, `poetry.lock`, `Pipfile.lock` | do not edit by hand; tell the user to re-lock, or run the repo's own lock command if it works offline-safe |

- A minimum, not a pin: an image built once is reused until a file it copies in changes
  (build caches and nodes that keep built images work this way), so the minimum is what a fresh build guarantees, and raising
  it is what picks up a fix. If `pip index versions precisionai-pace` shows a release
  newer than the minimum above, use that.
- A repo that pins every dependency exactly as a policy (not just in a lock file): ask the
  user. If they want a pin, use the latest version and say each new release then needs a
  bump.
- A Dockerfile that installs from a private or `--no-index` mirror needs Pace available
  there. The dependency edit is still right; flag it when the install fails and tell the
  user the install fails until Pace is on that index.

## Examples

`requirements.txt`, which the Dockerfile installs with `pip install -r`:

```text
torch>=2.3
precisionai-pace[lightning]>=1.0.0
```

A Dockerfile that installs packages on its own line, quoted:

```dockerfile
RUN pip install --no-cache-dir "precisionai-pace>=1.0.0" vllm
```

A base image with only the system Python (an Ollama image, for example) may need
`python3 -m pip install --break-system-packages "precisionai-pace>=1.0.0"`.
