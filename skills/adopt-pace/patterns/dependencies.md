# Dependency declarations

Add `precisionai-pace` everywhere the repo installs its own dependencies, with the same
pinning style as its neighbours. The extras (`[lightning]`, `[transformers]`) go in the same
brackets. Pace needs Python 3.11 or later.

| Where | Entry |
|---|---|
| `requirements*.txt` | `precisionai-pace` (or `precisionai-pace[lightning]`) |
| `pyproject.toml`, PEP 621 | `dependencies = [..., "precisionai-pace"]` (or the optional group the job installs) |
| `pyproject.toml`, Poetry | `precisionai-pace = "*"`, or `precisionai-pace = { version = "*", extras = ["lightning"] }` |
| `setup.py`, `setup.cfg` | `install_requires` list |
| conda `environment.yml` | under the `pip:` list, not as a conda package |
| `Dockerfile`: `pip install a b c` | add it to that line. If the Dockerfile only runs `pip install -r requirements.txt`, the requirements file is enough. |
| `uv.lock`, `poetry.lock`, `Pipfile.lock` | do not edit by hand; tell the user to re-lock, or run the repo's own lock command if it works offline-safe |

- A repo that pins exact versions gets an exact pin: look up the version with
  `pip index versions precisionai-pace`.
- A Dockerfile that installs from a private or `--no-index` mirror needs Pace available
  there. The dependency edit is still right; flag it when the install fails and tell the
  user the install fails until Pace is on that index.

## Examples

- Tested: the `requirements.txt` and `Dockerfile` beside each script in OrchAI's example
  repo, for example [`detection/requirements.txt`](https://github.com/Precision-AI-Inc/orchai-examples/blob/main/detection/requirements.txt)
  and [`llm/requirements.txt`](https://github.com/Precision-AI-Inc/orchai-examples/blob/main/llm/requirements.txt).
