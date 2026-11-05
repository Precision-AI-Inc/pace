# Dependency declarations

Add `precisionai-pace` everywhere the repo installs its own dependencies, as a minimum
version: `precisionai-pace>=0.1.3`. The extras (`[lightning]`, `[transformers]`) go in the
same brackets, `precisionai-pace[lightning]>=0.1.3`. Pace needs Python 3.11 or later.

| Where | Entry |
|---|---|
| `requirements*.txt` | `precisionai-pace>=0.1.3` (or `precisionai-pace[lightning]>=0.1.3`) |
| `pyproject.toml`, PEP 621 | `dependencies = [..., "precisionai-pace>=0.1.3"]` (or the optional group the job installs) |
| `pyproject.toml`, Poetry | `precisionai-pace = ">=0.1.3"`, or `precisionai-pace = { version = ">=0.1.3", extras = ["lightning"] }` |
| `setup.py`, `setup.cfg` | `install_requires` list |
| conda `environment.yml` | under the `pip:` list, not as a conda package |
| `Dockerfile`: `pip install a b c` | add it to that line. If the Dockerfile only runs `pip install -r requirements.txt`, the requirements file is enough. |
| `uv.lock`, `poetry.lock`, `Pipfile.lock` | do not edit by hand; tell the user to re-lock, or run the repo's own lock command if it works offline-safe |

- A minimum, not a pin: an image built once is reused until a file it copies in changes
  (OrchAI nodes work this way), so the minimum is what a fresh build guarantees, and raising
  it is what picks up a fix. 0.1.3 is the first release with the start-up correction in the
  estimate; if `pip index versions precisionai-pace` shows a newer one, use that.
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
