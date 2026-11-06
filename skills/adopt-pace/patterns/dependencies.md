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
  (OrchAI nodes work this way), so the minimum is what a fresh build guarantees, and raising
  it is what picks up a fix. If `pip index versions precisionai-pace` shows a release
  newer than the minimum above, use that.
- A repo that pins every dependency exactly as a policy (not just in a lock file): ask the
  user. If they want a pin, use the latest version and say each new release then needs a
  bump.
- A Dockerfile that installs from a private or `--no-index` mirror needs Pace available
  there. The dependency edit is still right; flag it when the install fails and tell the
  user the install fails until Pace is on that index.

## Examples

- Tested: the `requirements.txt` and `Dockerfile` beside each script in OrchAI's example
  repo, for example [`detection/requirements.txt`](https://github.com/Precision-AI-Inc/orchai-examples/blob/main/detection/requirements.txt)
  and [`llm/requirements.txt`](https://github.com/Precision-AI-Inc/orchai-examples/blob/main/llm/requirements.txt).
