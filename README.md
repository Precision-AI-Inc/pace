<p align="center">
  <img src="https://raw.githubusercontent.com/Precision-AI-Inc/pace/main/docs/assets/logo.png" alt="Precision AI Logo" width="160"/>
</p>

# Precision AI Pace

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/Precision-AI-Inc/pace/blob/main/LICENSE.md)
[![PyPI](https://img.shields.io/pypi/v/precisionai-pace.svg?include_prereleases)](https://pypi.org/project/precisionai-pace/)
[![Python](https://img.shields.io/pypi/pyversions/precisionai-pace.svg?include_prereleases)](https://pypi.org/project/precisionai-pace/)

---

> Pace follows semantic versioning. Before 1.0, a minor version may change the Python API, and a new line format comes with a minor version: 0.1 writes `pace:2` and still reads `pace:1`.

Pace is a logger with progress reporting built in. Its output reads like any good
logger in a terminal, and every progress line also carries a short, versioned trailer
that a scheduler can read to estimate when the job will finish.

- **One way to log across projects**: the same levels, format and progress bars everywhere.
- **Progress for real ML work**: loops, manual counters, timed steps, phases, PyTorch Lightning and Hugging Face training.
- **Completion estimates**: the terminal shows time remaining, and a scheduler such as OrchAI gets the same estimate from the job's logs.
- **No dependencies** in the core, and Python 3.11 or later, so it fits into existing training images.

---

## Installation

```bash
pip install precisionai-pace
```

---

## Usage

```python
from precisionai.pace import capture_logging, logger, progress, track

capture_logging()  # route PyTorch's and Transformers' standard logging through Pace

logger.info("Loading model")
logger.success("Model ready")

for batch in track(loader, description="Inference", unit="batch"):
    infer(batch)

with progress(total=len(requests), description="Evaluation", unit="request") as run:
    for future in as_completed(futures):
        consume(future.result())
        run.advance()
```

Also available: `timed`, `phase` and `plan` for multi-stage jobs, `root=True` to mark
the scope that stands for the whole job, and `completed=` with `resumed=True` to resume
from a checkpoint. Settings come from `logger.configure(...)` or `PACE_*` environment
variables, so a scheduler can change them without code changes.

In a terminal, progress is one bar that redraws in place. In a container or a log file it
is a rate-limited line like this one:

```text
2026-11-03T17:36:42Z PROGRESS Train  1840/5000  36.8%  2.4 step/s · ~21m57s  [pace:2 train c=1840 t=5000 e=763.1 q=57 r=6bfc3dd5 root]
```

The part before `[pace:2 ...]` is for people. The bracketed trailer is a stable,
versioned format described in [docs/design.md](https://github.com/Precision-AI-Inc/pace/blob/main/docs/design.md); a program that reads it
should use `parse_line`, `LineSplitter` and `JobForecast` (see
[Reading Pace output](https://github.com/Precision-AI-Inc/pace/blob/main/docs/reading-output.md)).

---

## Documentation

| Page | Read it for |
|---|---|
| [Getting started](https://github.com/Precision-AI-Inc/pace/blob/main/docs/getting-started.md) | Install, first log lines, first progress loop, terminal vs container output. |
| [Logging](https://github.com/Precision-AI-Inc/pace/blob/main/docs/logging.md) | Levels, `exception`, `configure` and precedence, `capture_logging`, colour, encodings, ranks. |
| [Progress](https://github.com/Precision-AI-Inc/pace/blob/main/docs/progress.md) | `track`, `progress`, `timed`, `phase`, `plan`, nesting, resuming, root scope, emission rates, the terminal bar. |
| [Examples](https://github.com/Precision-AI-Inc/pace/blob/main/examples/README.md) | One runnable script per style of logging and progress: loops, epochs, resuming, phases, ranks, reading the output. |
| [Framework integrations](https://github.com/Precision-AI-Inc/pace/blob/main/docs/frameworks.md) | The PyTorch Lightning and Hugging Face Transformers callbacks. |
| [Migrating](https://github.com/Precision-AI-Inc/pace/blob/main/docs/migrating.md) | From `print`, `tqdm`, Loguru and `logging`. |
| [Configuration](https://github.com/Precision-AI-Inc/pace/blob/main/docs/configuration.md) | Every `PACE_*` variable and its `configure` equivalent. |
| [Reading Pace output](https://github.com/Precision-AI-Inc/pace/blob/main/docs/reading-output.md) | For programs that read logs: parsing, de-duplication, estimates, conformance fixtures. |
| [FAQ](https://github.com/Precision-AI-Inc/pace/blob/main/docs/faq.md) | Short answers. |
| [Specification](https://github.com/Precision-AI-Inc/pace/blob/main/docs/design.md) | The line format and the forecasting rules. |

The HTML build of these pages (and the module reference) is produced with Sphinx:

```bash
cd docs
make html       # HTML docs → docs/_build/html/index.html
make latexpdf   # PDF      → docs/_build/latex/documentation.pdf
```

Dependencies: `pip install -r docs/requirements.txt`

---

## Development

```bash
python -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"
pre-commit install

python -m pytest
```

Coverage must remain at or above **90%**, enforced by pytest and the pre-commit hook.
CI runs the tests on Python 3.11 through 3.13.

---

## Project layout

```
precisionai/pace/   # the package (core: standard library only)
docs/               # Sphinx (HTML + LaTeX/PDF), including design.md
skills/adopt-pace/  # Claude Code skill that adds Pace to a repository
examples/           # runnable examples
tests/              # mirrors the package structure
```

See [CLAUDE.md](https://github.com/Precision-AI-Inc/pace/blob/main/CLAUDE.md) for the coding standard.

---

## Releasing

Push a tag matching `v*.*.*` (e.g. `v0.1.0`) to trigger `.github/workflows/release.yml`,
which runs the test suite, builds the sdist and wheel, publishes to PyPI via trusted
publishing, and creates a GitHub Release. The version comes from the git tag via
`setuptools-scm`. Update `CHANGELOG.md` before tagging.

---

## Security

To report a security vulnerability, see [SECURITY.md](https://github.com/Precision-AI-Inc/pace/blob/main/SECURITY.md). Do not open a public issue.

---

## Contributing

See [CONTRIBUTING.md](https://github.com/Precision-AI-Inc/pace/blob/main/CONTRIBUTING.md) for setup and PR guidelines, and
[CODE_OF_CONDUCT.md](https://github.com/Precision-AI-Inc/pace/blob/main/CODE_OF_CONDUCT.md) for community expectations.

---

## License

[Apache 2.0](https://github.com/Precision-AI-Inc/pace/blob/main/LICENSE.md) © Precision AI
