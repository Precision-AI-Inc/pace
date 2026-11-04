# Pace — Project Standards

Apply these standards when writing, reviewing, or refactoring code in Pace. Read [docs/design.md](docs/design.md) before changing behaviour: it is the specification for the logger, the progress line format and the forecasting rules.

---

## Environment

- Python 3.11+ is supported (the oldest release that is not end-of-life), managed with `python -m venv .venv` (never conda)
- Install: `pip install -e ".[dev]"` → installs all deps including pre-commit
- Register hooks once: `pre-commit install`
- CI runs the tests on 3.11 through 3.13. Release 0.0.1 still covers 3.8 to 3.10.

---

## The rules that matter most

1. **The core has no runtime dependencies.** `precisionai.pace` imports only the standard library. It is installed into users' training images and into other packages, and must never add to them. A framework adapter imports its framework only inside its own module, behind an extra (`[lightning]`, `[transformers]`).
2. **The line format is a public contract.** Anything a consumer parses (the `[pace ...]` trailer) changes only as documented in `docs/design.md`: it has no version number, so a change may only add optional keys or words, which readers ignore until they know them. A format change updates `docs/design.md` and the fixtures in `tests/fixtures/protocol/` in the same PR. The readable text before the trailer is for people and may change freely; never make a consumer depend on it.
3. **Python 3.11 is the floor.** Do not use syntax or library calls newer than that. `X | None`, built-in generics, `match`, `Self`, `itertools.pairwise` and `dataclass(slots=True)` are fine.
4. **Never log secrets on the package's own initiative.** Pace never writes environment variables, configuration or credentials by itself, and labels stay short and are not repeated on every update.
5. **Parsing is defensive.** Log lines are untrusted input. The parser bounds every size and value, never evaluates anything, and ignores (never raises on) a malformed or unsupported line.

---

## Project layout

```
precisionai/pace/
  core.py, _env.py, _render.py, capture.py   # the logger; _env.py holds Settings, one table of PACE_ variable, check and default per setting
  scopes.py        # track, progress, timed, phase, plan
  protocol.py      # Record, the [pace ...] trailer, parse_line, LineSplitter
  forecast.py      # ScopeForecast, JobForecast
  lightning.py, transformers.py, _adapters.py   # framework callbacks (extras only)
  tqdm.py          # PaceTqdm: stands in for the tqdm class where a library takes one (stdlib only)
  version.py       # installed package version
  __init__.py      # re-exports only — no logic
docs/              # Sphinx (HTML + LaTeX/PDF); design.md is the specification
skills/adopt-pace/ # Claude Code skill that adds Pace to a repo; verify.py checks a log
examples/          # runnable examples
tests/             # mirrors package structure; fixtures/protocol/ holds format conformance cases
```

`precisionai` is a fixed, namespace-only top-level package (no logic, just an SPDX header) shared by Precision AI Python projects, which is what makes `precisionai.<namespace>.*` imports consistent across them.

---

## pyproject.toml — canonical config

```toml
[project]
dynamic = ["version"]   # version comes from the git tag via setuptools-scm, never hardcoded
requires-python = ">=3.11"
dependencies = []       # see "The rules that matter most"

[tool.setuptools_scm]
version_scheme = "no-guess-dev"
local_scheme = "no-local-version"
fallback_version = "0.0.0"

[tool.ruff]
line-length = 120
target-version = "py311"

[tool.ruff.lint]
select = ["E","W","F","I","UP","B","SIM","N","C90","D","PT","RUF","PL","ANN","PERF","S"]
ignore = [
    "E501",    # enforced by ruff-format
    "SIM108",  # ternary readability
    "D100","D104",          # module/package docstrings optional
    "D203","D213",          # pydocstyle conflicts — always ignore these two
    "PLR0913","PLR2004",    # arg count (progress scopes take several optional fields) + magic values in tests
    "ANN401",              # Any is allowed for genuinely dynamic types
]
[tool.ruff.lint.per-file-ignores]
"**/__init__.py" = ["F401"]
"tests/**"       = ["D","PLR","ANN","S"]
"docs/conf.py"   = ["E402","UP031","ANN","S"]

[tool.ruff.lint.pydocstyle]
convention = "numpy"          # enforces NumPy docstring style

[tool.ruff.lint.mccabe]
max-complexity = 10

[tool.ruff.lint.isort]
known-first-party = ["precisionai"]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
docstring-code-format = true  # formats code blocks inside docstrings

[tool.pytest.ini_options]
addopts = "--cov=precisionai --cov-report=term-missing --cov-fail-under=90"

[tool.coverage.report]
fail_under = 90
exclude_lines = ["pragma: no cover","if __name__ == .__main__.:",
                 "raise ImportError","except ImportError","if TYPE_CHECKING:"]

[tool.pyright]
pythonVersion = "3.11"
typeCheckingMode = "standard"
reportMissingImports = false
reportMissingModuleSource = false
```

---

## pre-commit

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v6.0.0
    hooks: [check-added-large-files (--maxkb=1800), check-yaml, check-json,
            check-toml, end-of-file-fixer, trailing-whitespace,
            check-merge-conflict, detect-private-key, debug-statements,
            fix-byte-order-marker]

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.15.17
    hooks: [ruff (--fix), ruff-format]

  - repo: local
    hooks:
      - id: pyright
        name: pyright
        entry: python -m pyright
        language: system
        types: [python]
        pass_filenames: false

  - repo: local
    hooks:
      - id: pytest
        entry: python -m pytest
        language: system
        args: [tests/, -q, --tb=short, --no-header]
        always_run: true
```

---

## Code style

### License header
- Every `.py` file starts with a two-line SPDX header, followed by a blank line before the module docstring (if any):
  ```python
  # Copyright 2026 Precision AI
  # SPDX-License-Identifier: Apache-2.0
  ```
- No "CONFIDENTIAL" or proprietary banners — this codebase is Apache 2.0 licensed; any such banner is a bug.

### Imports
- All imports at the **top of the file** — never inside functions
- Optional framework deps (adapters only): module-level `try/except ImportError` with a flag variable
- No silent fallbacks unless mathematically equivalent (document why)
- `__init__.py` re-exports only — no logic, no comments between import blocks
- `__all__` must be **alphabetically sorted**

```python
# optional dep pattern (adapter modules only — never in the core)
try:
    import lightning.pytorch as pl  # type: ignore[import]
    _LIGHTNING_AVAILABLE = True
except ImportError:
    _LIGHTNING_AVAILABLE = False

# inside the class/function that needs it
if not _LIGHTNING_AVAILABLE:
    raise ImportError("lightning is required: pip install precisionai-pace[lightning]") from None
```

### Docstrings
- **NumPy style** on all public functions, classes, and modules
- Module docstrings: plain English, no prefixes (no P0/P1/P2, no "original", no labels)
- One-line summary in **imperative mood** ("Compute …", "Return …", "Save …")
- Blank line between summary and extended description (D205)

### Type hints
- Required on **all** function signatures (public and private) — enforced by ruff (`ANN`) and pyright
- No `from __future__ import annotations`: write `X | None`, `list[int]` and `dict[str, int]` directly. A method that returns its own class is annotated `Self` (`typing.Self`), not the class name.
- Frozen value types are `@dataclass(frozen=True, slots=True)`. Do not turn `Event` or `ForecastState` into `StrEnum`: `str()` and f-string output of the public enums would change.
- Use `Any` from `typing` for genuinely dynamic types — don't use `object` when methods will be called on it
- pyright config lives in `[tool.pyright]` in `pyproject.toml` — never pass type-checker flags inline

### Comments
- Only when the **why** is non-obvious
- No section-divider comments that describe what the code already says
- No cryptic labels, no TODO/FIXME without a ticket reference

### Exception handling
- Always `raise X from err` or `raise X from None` inside `except` blocks (B904)
- Logging and progress calls never raise into user code because of a formatting or I/O problem; they degrade (drop the record, fall back to plain text) instead

### General
- `len()` returns `int` — never `int(len(...))`
- Loop variables not used in the body → rename to `_`
- `assert a and b` in tests → split into separate asserts
- `@pytest.fixture` not `@pytest.fixture()`
- `pytest.raises` always includes `match=` parameter
- No `# type: ignore[attr-defined]` when `Any` already covers the attribute
- Rates and intervals use `time.monotonic()`; wall-clock time is only for the printed timestamp

---

## README is the PyPI page

`README.md` is published as the PyPI project description, where relative paths do not resolve. Every link and image in it is an absolute `https://github.com/Precision-AI-Inc/pace/blob/main/...` or `https://raw.githubusercontent.com/Precision-AI-Inc/pace/main/...` URL (enforced by `tests/test_readme.py`). Those URLs only load for anonymous visitors once the repository is public.

---

## Docs (Sphinx)

```
docs/
  conf.py          # version from git tag, logo, LaTeX/fancyhdr, enumitem fix
  index.rst        # toctrees: user guide, reading output, reference
  design.md        # the specification — update it with any behaviour or format change
  modules.rst      # autodoc for all public submodules
  requirements.txt # sphinx, sphinx-rtd-theme, myst-parser, sphinx-autodoc-typehints
  Makefile         # make html | make latexpdf | make clean
  assets/logo.png
```

- `conf.py` copies root `README.md` → `docs/readme.md` at build time
- Version read from `git describe --tags --exact-match`, falls back to `0.0.0`

---

## Testing

- Mirror package structure: `tests/test_<module>.py`
- `tests/test_golden_output.py` pins every byte Pace writes for representative scripts; rewrite its files with `PACE_UPDATE_GOLDEN=1` only for a deliberate format change. `tests/test_protocol_properties.py` runs `hypothesis` property tests on the trailer and the splitter
- Shared fixtures in `tests/conftest.py`
- 90% coverage hard minimum — enforced by pytest and pre-commit
- Format conformance cases live in `tests/fixtures/protocol/` as input lines plus expected parse results; consumers run the same files
- Tests that depend on time use an injected clock, never `sleep`
- Integration tests marked `@pytest.mark.integration` and excluded from default runs

---

## Repository files

Besides the source layout above, the repository carries:

```
.gitattributes              # normalize line endings to LF; declare binary assets
CHANGELOG.md                # Keep a Changelog format + Semantic Versioning
CODE_OF_CONDUCT.md          # Contributor Covenant v2.1
MANIFEST.in                 # sdist packaging — include docs/license/changelog, exclude tests/examples
.github/
  dependabot.yml            # weekly pip + github-actions update checks
  ISSUE_TEMPLATE/
    bug_report.yml
    feature_request.yml
    config.yml               # blank_issues_enabled: false + link to SECURITY.md
  PULL_REQUEST_TEMPLATE.md
  workflows/
    ci.yml                   # dispatches to pre-commit.yml + unit-test.yml
    pre-commit.yml
    unit-test.yml
    release.yml               # tag push → build, PyPI publish, GitHub Release
```

## Releasing

- Versions are **never hardcoded** — `setuptools-scm` derives the package version from the current git tag (see the `pyproject.toml` block above). Tag with `vX.Y.Z`.
- Update `CHANGELOG.md` under `[Unreleased]` as you go; move those entries to a new `## [X.Y.Z] - YYYY-MM-DD` section when cutting a release.
- Pushing a `v*.*.*` tag triggers `.github/workflows/release.yml`: runs the full test matrix, builds the sdist/wheel, verifies the built version matches the tag, publishes to PyPI via trusted publishing (`id-token: write`, no stored API tokens), and creates a GitHub Release with generated notes.
- After a release is on PyPI, check OrchAI's example repo (`Precision-AI-Inc/orchai-examples`, formerly `orch-example-yolov5`): raise the `precisionai-pace>=` minimum in each `requirements.txt` and `Dockerfile` to the new version (a minimum, never a pin; raising it is what makes the nodes rebuild their cached images), and replace any workaround the release makes unnecessary. Open that as a PR in the example repo.
- CI checkout steps that build from git history (release builds, docs version detection) must use `fetch-depth: 0` — a shallow checkout has no tags for `setuptools-scm` or `git describe` to find.

---

## What to avoid

| Pattern | Instead |
|---|---|
| A new entry in `dependencies` | Put it behind an extra, imported only by the adapter module |
| Parsing the readable part of a progress line | Parse only the `[pace ...]` trailer |
| Changing or removing a trailer key in place | A new optional key or word; the trailer has no version number, so nothing is ever removed or redefined (see `docs/design.md`) |
| Syntax or library calls newer than 3.11 (`type` statements, `def f[T]()`) | `TypeVar` and 3.11 equivalents |
| `time.time()` for rates or intervals | `time.monotonic()` |
| `int(len(x))` | `len(x)` |
| `assert a and b` (tests) | two separate asserts |
| `@pytest.fixture()` | `@pytest.fixture` |
| `pytest.raises(ValueError)` | `pytest.raises(ValueError, match="…")` |
| `raise X` inside except | `raise X from None` or `raise X from err` |
| Deferred imports inside functions | Module-level try/except with flag |
| Cryptic prefixes (P0–P4, "original") | Plain descriptive names |
| `# type: ignore[attr-defined]` on `Any` | Remove — redundant |
| `# noqa` suppression | Fix the underlying issue |
| Silent fallback for optional dep | Raise `ImportError` with install hint |
