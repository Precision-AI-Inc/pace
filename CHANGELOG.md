# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Progress lines are shorter, in a new line format: the trailer is `[pace ...]`, with no version number. It starts with the scope id (and `<parent` for a nested scope), names the event as a plain word (`start`, `end`, `fail`, `step`, `plan`; an update has none), and leaves out what a reader can work out: zeros, an end's total when it equals the count, the label after the start, and the sequence number except on updates and plans. The process run id (`r=`) is now on every line. The readable part drops the `0.0%` and `estimating…` from start lines, lines up the counts of a scope, and indents a nested scope under its parent. A real Ollama job's progress goes from five lines of 132 to 169 characters to four of 123 to 133. This release does not read the `[pace:1 ...]` lines of Pace 0.0.x, and 0.0.x does not read these, so a job and the program reading its output move to 0.1.0 together. With no version number, the format will from now on only gain keys and words, which older readers ignore.
- A `@timed` call or a `phase()` that is over before its first line is due (`PACE_MIN_INTERVAL`, 15 seconds by default) now writes one `STEP` line instead of a start line and an end line. One that takes longer has its start line written with the next line Pace writes after the interval, so it shows while it runs if anything else reports meanwhile; a long step with nothing else going on (a lone download) shows only when it ends. Pace starts no threads. A failing one still writes its start and a `fail` line. A terminal, and `PACE_MIN_INTERVAL=0`, write start and end lines as before.
- For code that uses the reading or writing API directly: `parse_line` reads only the new format, `Record.seq` is `None` on a start, end, fail or step (only updates and plans carry one), `Record` has defaults for `seq` and `elapsed`, `Event.STEP` is new, and `ScopeForecast.observe` and `JobForecast.observe` return whether they took the record. `format_trailer` writes the new format, and needs a run id on every record and a `seq` of at least 2 on an update (1 on a plan).

### Fixed

- A reader that captures both stdout and stderr, and reads a scope's start line from one only after its updates from the other, no longer takes the late start for a restart and resets the estimate. Every line now carries the run id, so a late copy is told apart from a real restart, including when the reader joined the job partway through.

## [0.0.2] - 2026-11-03

### Added

- `precisionai.pace.tqdm.PaceTqdm`, a stand-in for the `tqdm` class that reports to Pace instead of drawing a bar. Pass it where a library takes a progress-bar class, such as `LLM.generate(prompts, params, use_tqdm=PaceTqdm)` in vLLM, so a job that runs entirely inside one call still reports progress. It needs no `tqdm` install.
- `capture_logging(take_over=...)` names libraries, such as `"lightning"` and `"transformers"`, that print their own log lines. Their own console handlers are removed and their loggers pass records on, so each line is written once, in Pace's format, instead of twice or not at all. The docs also say how to bring vLLM's log lines in (`VLLM_CONFIGURE_LOGGING=0`).
- `examples/library_hooks.py`, a runnable example of `PaceTqdm` and `capture_logging(take_over=...)` with a stand-in library.

### Changed

- The `adopt-pace` skill is split so an agent loads less: `SKILL.md` is short and holds the workflow and an index, and each situation has its own small file under `patterns/`, read only when the repository needs it (`patterns.md` is gone). A new `patterns/model-server.md` covers jobs that start a model server, such as Ollama, and send it requests.
- Requires Python 3.11 or later; 0.0.1 still covers 3.8–3.10 and writes the same `pace:1` lines.

### Fixed

- Under `mpirun` (Open MPI) and MPICH or Intel MPI launchers, Pace now reads the process rank from `OMPI_COMM_WORLD_RANK` and `PMI_RANK`. Before, every MPI process counted as rank 0, so every one of them wrote progress. The global rank variables (`RANK`, the MPI ones, `SLURM_PROCID`) are now read before `LOCAL_RANK`, which restarts at 0 on every node.
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

[Unreleased]: https://github.com/Precision-AI-Inc/pace/compare/v0.0.2...HEAD
[0.0.2]: https://github.com/Precision-AI-Inc/pace/compare/v0.0.1...v0.0.2
[0.0.1]: https://github.com/Precision-AI-Inc/pace/releases/tag/v0.0.1
