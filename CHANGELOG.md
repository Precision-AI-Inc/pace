# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Package scaffold for `precisionai-pace`, with no runtime dependencies and support for Python 3.8 and later.
- Design document for the logger, the progress line format and the completion-forecast rules.
- `logger` with `debug`, `info`, `success`, `warning`, `error`, `critical` and `exception`, writing to stdout by default, falling back to plain ASCII on terminals that cannot show Unicode, and never raising into your code.
- Progress reporting with `track`, `progress`, `timed`, `phase` and `plan`, written as readable lines that also carry a `[pace:1 ...]` trailer a scheduler can read.
- A live progress bar on terminals, and rate-limited progress lines everywhere else, so a long run writes a few lines a minute however many steps it has.
- `capture_logging()` to route the standard library's `logging` (and so PyTorch's and Transformers') through Pace.
- `parse_line` and `LineSplitter` for programs that read Pace output, with shared conformance fixtures.
- Time-remaining estimates (`ScopeForecast`, `JobForecast`) shared by the terminal display and by schedulers.
- Callbacks for PyTorch Lightning (`precisionai.pace.lightning.PaceCallback`) and Hugging Face Transformers and TRL (`precisionai.pace.transformers.PaceCallback`), installed with the `[lightning]`, `[transformers]` or `[all]` extras.
- `skills/adopt-pace/`, a Claude Code skill that adds Pace to a repository and checks the result with `verify.py`.
- User documentation: getting started, logging, progress, framework integrations, migration guides (`print`, `tqdm`, Loguru, `logging`), configuration, a guide for programs that read Pace output, and a FAQ.

[Unreleased]: https://github.com/Precision-AI-Inc/pace/commits/main
