# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `precisionai.pace.tqdm.PaceTqdm`, a stand-in for the `tqdm` class that reports to Pace instead of drawing a bar. Pass it where a library takes a progress-bar class, such as `LLM.generate(prompts, params, use_tqdm=PaceTqdm)` in vLLM, so a job that runs entirely inside one call still reports progress. It needs no `tqdm` install.
- `capture_logging(take_over=...)` names libraries, such as `"lightning"` and `"transformers"`, that print their own log lines. Their own console handlers are removed and their loggers pass records on, so each line is written once, in Pace's format, instead of twice or not at all. The docs also say how to bring vLLM's log lines in (`VLLM_CONFIGURE_LOGGING=0`).

### Changed

- Requires Python 3.11 or later; 0.0.1 still covers 3.8–3.10 and writes the same `pace:1` lines.

### Fixed

- Under `mpirun` (Open MPI) and MPICH or Intel MPI launchers, Pace now reads the process rank from `OMPI_COMM_WORLD_RANK` and `PMI_RANK`. Before, every MPI process counted as rank 0, so every one of them wrote progress. The global rank variables (`RANK`, the MPI ones, `SLURM_PROCID`) are now read before `LOCAL_RANK`, which restarts at 0 on every node.
- A reader fed both stdout and stderr no longer mistakes a scope's `start` line, read after that scope's updates, for a restart. Before, it threw away the progress already seen, and the copies of earlier updates from the other stream were counted again, which made the completion estimate jump.
- The Hugging Face Transformers callback no longer shows an absurd speed for an evaluation pass (for example `158 batch/s` on a pass that took six seconds). The Trainer reports a batch only after it has run, so the pass now starts with its first batch already counted, and the speed and "done in" time cover the batches after it.

## [0.0.1] - 2026-11-03

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
- Runnable examples, one per style of logging and progress (plain loops, epochs, concurrent work, phases and a plan, resuming, unknown totals, nested scopes, `logging` capture, distributed ranks, and reading the output).
- User documentation: getting started, logging, progress, framework integrations, migration guides (`print`, `tqdm`, Loguru, `logging`), configuration, a guide for programs that read Pace output, and a FAQ.

[Unreleased]: https://github.com/Precision-AI-Inc/pace/compare/v0.0.1...HEAD
[0.0.1]: https://github.com/Precision-AI-Inc/pace/releases/tag/v0.0.1
