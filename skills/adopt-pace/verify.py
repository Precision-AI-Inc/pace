#!/usr/bin/env python3
# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Check captured Pace output after adopting Pace in a repository.

Reads a log file (or standard input) written by a job that uses ``precisionai-pace``,
parses every progress line with :func:`precisionai.pace.parse_line`, and reports what a
scheduler would see: the scopes, which one is the root, whether the totals are known,
whether every scope was closed, and the estimator's view at the end of the log.

Usage::

    python skills/adopt-pace/verify.py run.log
    python my_job.py --smoke 2>&1 | python skills/adopt-pace/verify.py

The exit status is 0 when nothing is wrong, 1 when a check failed (or, with
``--strict``, when there is any warning), and 2 when there was nothing to read.
"""

import argparse
import re
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TextIO

from precisionai.pace import Event, ForecastState, JobEstimate, JobForecast, Record, parse_line

TRAILER_MARK = "[pace:"
MAX_LISTED_LINES = 5
LABEL_LIMIT = 64
_TIMESTAMP = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})Z\b")
_SECRET_WORDS = re.compile(r"secret|token|passw|api[_-]?key|bearer|credential", re.IGNORECASE)


@dataclass
class ScopeSummary:
    """What the log says about one scope.

    Parameters
    ----------
    scope : str
        Scope id.
    """

    scope: str
    label: str | None = None
    unit: str | None = None
    parent: str | None = None
    root: bool = False
    resumed: bool = False
    records: int = 0
    starts: int = 0
    completed: float | None = None
    total: float | None = None
    status: str = "open"
    first_index: int = 0
    closed_index: int | None = None
    last_index: int = 0


@dataclass
class Finding:
    """One thing worth telling the user.

    Parameters
    ----------
    level : str
        ``"error"`` (a check failed) or ``"warning"`` (probably a mistake, or a known gap).
    message : str
        What was found.
    """

    level: str
    message: str


@dataclass
class Report:
    """The result of :func:`analyze`.

    Parameters
    ----------
    lines : int
        Lines read.
    """

    lines: int = 0
    records: int = 0
    duplicates: int = 0
    malformed: list[int] = field(default_factory=list)
    scopes: dict[str, ScopeSummary] = field(default_factory=dict)
    plans: list[tuple[tuple[str, float], ...]] = field(default_factory=list)
    estimate: JobEstimate | None = None
    sources: set[str] = field(default_factory=set)
    reached_live: bool = False
    findings: list[Finding] = field(default_factory=list)

    @property
    def roots(self) -> list[str]:
        """Return the ids of the scopes marked as root."""
        return [name for name, scope in self.scopes.items() if scope.root]

    @property
    def errors(self) -> list[Finding]:
        """Return the findings that fail the check."""
        return [finding for finding in self.findings if finding.level == "error"]

    @property
    def warnings(self) -> list[Finding]:
        """Return the findings that are warnings."""
        return [finding for finding in self.findings if finding.level == "warning"]


def _timestamp(line: str) -> float | None:
    match = _TIMESTAMP.match(line)
    if match is None:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y-%m-%dT%H:%M:%S").replace(tzinfo=UTC).timestamp()
    except ValueError:
        return None


class _Reader:
    """Feed records into a report and a replay of the estimator."""

    def __init__(self, report: Report) -> None:
        self.report = report
        self.forecast = JobForecast()
        self.now = 0.0

    def read(self, index: int, line: str) -> None:
        """Take one line: parse it, and count it against the report."""
        stamp = _timestamp(line)
        if stamp is not None:
            self.now = stamp
        record = parse_line(line)
        if record is None:
            if TRAILER_MARK in line:
                self.report.malformed.append(index)
            return
        # The estimator ignores a copy read from the other stream and a line read late.
        if not self.forecast.observe(record, self.now):
            self.report.duplicates += 1
            return
        self.report.records += 1
        if record.event is Event.PLAN:
            self.report.plans.append(record.phases)
        else:
            self._summarise(index, record)
        self._watch_estimate()

    def _summarise(self, index: int, record: Record) -> None:
        scope = self.report.scopes.get(record.scope)
        if scope is None:
            scope = self.report.scopes[record.scope] = ScopeSummary(record.scope, first_index=index)
        scope.records += 1
        scope.last_index = index
        scope.root = scope.root or record.root
        scope.parent = record.parent or scope.parent
        if record.event in (Event.START, Event.STEP):
            scope.starts += 1
            scope.label = record.label or scope.label
            scope.unit = record.unit or scope.unit
            scope.resumed = scope.resumed or record.resumed
            scope.status = "open"
        if record.completed is not None:
            scope.completed = record.completed
        if record.total is not None:
            scope.total = record.total
        if record.event in (Event.END, Event.FAIL, Event.STEP):
            scope.status = "fail" if record.event is Event.FAIL else "end"
            scope.closed_index = index

    def _watch_estimate(self) -> None:
        estimate = self.forecast.snapshot(self.now)
        if estimate.state in (ForecastState.LIVE, ForecastState.STALE):
            self.report.reached_live = True
            if estimate.source:
                self.report.sources.add(estimate.source)

    def finish(self) -> None:
        """Take the estimator's final view and run the checks."""
        self.report.estimate = self.forecast.snapshot(self.now)
        _check(self.report)


def _check(report: Report) -> None:
    add = report.findings.append
    if report.malformed:
        listed = ", ".join(str(n) for n in report.malformed[:MAX_LISTED_LINES])
        add(
            Finding(
                "error", f"{len(report.malformed)} line(s) carry a [pace: trailer that does not parse (lines {listed})"
            )
        )
    if not report.scopes and not report.plans:
        add(Finding("error", "no progress records found: Pace never wrote to this output"))
        return
    _check_roots(report)
    for scope in report.scopes.values():
        _check_scope(report, scope)
    _check_plans(report)
    _check_root_covers_job(report)


def _check_roots(report: Report) -> None:
    roots = report.roots
    top = [name for name, scope in report.scopes.items() if scope.parent is None]
    if len(roots) > 1:
        report.findings.append(
            Finding("error", f"{len(roots)} scopes are marked root ({', '.join(roots)}); mark exactly one")
        )
    elif not roots and not report.plans and len(top) == 1:
        report.findings.append(
            Finding("warning", f"scope {top[0]!r} is the only top-level scope but is not marked root=True")
        )
    elif not roots and not report.plans and len(top) > 1:
        report.findings.append(
            Finding(
                "warning",
                "several top-level scopes, no root and no plan: the scheduler can only estimate the current phase",
            )
        )


def _check_scope(report: Report, scope: ScopeSummary) -> None:
    add = report.findings.append
    if scope.status == "open":
        add(Finding("error", f"scope {scope.scope!r} was started but never ended or failed (no end record)"))
    elif scope.status == "fail":
        add(Finding("warning", f"scope {scope.scope!r} ended with fail"))
    if scope.total is None:
        level = "warning"
        what = "the root scope" if scope.root else f"scope {scope.scope!r}"
        add(Finding(level, f"{what} has no total (reported as unknown), so it cannot be estimated"))
    label = scope.label or ""
    if _SECRET_WORDS.search(label):
        add(Finding("warning", f"scope {scope.scope!r}: the label {label!r} looks like it may contain a secret"))
    if len(label) >= LABEL_LIMIT:
        add(Finding("warning", f"scope {scope.scope!r}: the label is {len(label)} characters; keep labels short"))


def _check_plans(report: Report) -> None:
    for phases in report.plans[-1:]:
        planned = [name for name, _ in phases]
        for name in planned:
            if name not in report.scopes:
                report.findings.append(
                    Finding("error", f"the plan lists {name!r} but no scope has that id (check the name or scope=)")
                )
        _check_plan_covers_job(report, planned)


def _check_plan_covers_job(report: Report, planned: list[str]) -> None:
    closed = [report.scopes[name].closed_index for name in planned if name in report.scopes]
    if not closed or any(index is None for index in closed):
        return
    plan_end = max(index for index in closed if index is not None)
    late = [
        scope.scope
        for scope in report.scopes.values()
        if scope.parent is None and not scope.root and scope.scope not in planned and scope.last_index > plan_end
    ]
    if late:
        report.findings.append(
            Finding(
                "warning",
                f"{', '.join(late)} ran after the last planned phase ended but is not in the plan: "
                "the scheduler treats the job as complete once every planned phase has ended",
            )
        )


def _check_root_covers_job(report: Report) -> None:
    roots = report.roots
    if len(roots) != 1:
        return
    root = report.scopes[roots[0]]
    if root.closed_index is None:
        return
    later = [
        scope.scope
        for scope in report.scopes.values()
        if scope.scope != root.scope and scope.parent is None and scope.last_index > root.closed_index
    ]
    if later:
        report.findings.append(
            Finding(
                "warning",
                f"root {root.scope!r} ended while {', '.join(later)} was still to run or running: "
                "the scheduler treats the job as complete when the root ends",
            )
        )


def analyze(lines: Iterable[str]) -> Report:
    """Parse captured output and check it.

    Parameters
    ----------
    lines : iterable of str
        The lines of the captured output, in order. In-place bar redraws (carriage
        returns) are split into lines first, as the scheduler's reader does.

    Returns
    -------
    Report
        Scopes, the estimator's view at the end of the log, and the findings.
    """
    report = Report()
    reader = _Reader(report)
    for text in lines:
        for line in text.splitlines() or [""]:
            report.lines += 1
            reader.read(report.lines, line)
    reader.finish()
    return report


def _number(value: float | None) -> str:
    if value is None:
        return "?"
    return f"{value:g}"


def _seconds(value: float | None) -> str:
    if value is None:
        return "-"
    minutes, seconds = divmod(round(value), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    return f"{minutes}m{seconds:02d}s" if minutes else f"{seconds}s"


def _scope_row(scope: ScopeSummary) -> str:
    flags = ", ".join(flag for flag, on in (("root", scope.root), ("resumed", scope.resumed)) if on)
    parent = f"in {scope.parent}" if scope.parent else "top-level"
    done = f"{_number(scope.completed)}/{_number(scope.total)} {scope.unit or ''}".strip()
    return f"  {scope.scope:<20} {done:<22} {scope.status:<5} {parent:<18} {flags}".rstrip()


def format_report(report: Report) -> str:
    """Return the report as text for a person.

    Parameters
    ----------
    report : Report
        The result of :func:`analyze`.

    Returns
    -------
    str
        A short summary: counts, scopes, plan, the scheduler's view, findings, result.
    """
    out = [f"read {report.lines} lines: {report.records} Pace records, {len(report.scopes)} scopes"]
    if report.duplicates:
        out.append(
            f"  ({report.duplicates} record(s) ignored as copies or late lines, as a consumer reading both streams would)"
        )
    out.extend(_scope_row(scope) for scope in report.scopes.values())
    if report.plans:
        out.append("plan: " + ", ".join(f"{name}:{weight:g}" for name, weight in report.plans[-1]))
    out.append("root: " + (", ".join(report.roots) if report.roots else "none"))
    out.append(_estimate_line(report))
    out.extend(f"{finding.level.upper():<7} {finding.message}" for finding in report.findings)
    out.append(_result_line(report))
    return "\n".join(out)


def _estimate_line(report: Report) -> str:
    estimate = report.estimate
    if estimate is None:
        return "scheduler view at the end of the log: none"
    text = f"scheduler view at the end of the log: {estimate.state.value}"
    if estimate.state is ForecastState.COMPLETE:
        text += " (the job is recognised as finished)"
    elif estimate.state is ForecastState.FAILED:
        text += " (the job is recognised as failed)"
    elif estimate.remaining_seconds is not None:
        text += f", {_seconds(estimate.remaining_seconds)} left from {estimate.source}"
    live = (
        "reached a live estimate"
        if report.reached_live
        else "never reached a live estimate (a run under a minute is too short to)"
    )
    return f"{text}; {live}"


def _result_line(report: Report) -> str:
    if report.errors:
        return f"FAIL: {len(report.errors)} error(s), {len(report.warnings)} warning(s)"
    return f"PASS: {len(report.warnings)} warning(s)"


def main(argv: Sequence[str] | None = None, stdin: TextIO | None = None, stdout: TextIO | None = None) -> int:
    """Run the check from the command line.

    Parameters
    ----------
    argv : sequence of str, optional
        Arguments; ``sys.argv[1:]`` by default. File names, or ``-`` (or nothing) for standard input.
    stdin : file-like, optional
        Replacement for standard input.
    stdout : file-like, optional
        Replacement for standard output.

    Returns
    -------
    int
        0 when nothing is wrong, 1 on an error (or any warning with ``--strict``), 2 when there is nothing to read.
    """
    parser = argparse.ArgumentParser(description="Check Pace progress output captured from a job.")
    parser.add_argument("files", nargs="*", help="log files; '-' or none reads standard input")
    parser.add_argument("--strict", action="store_true", help="treat warnings as failures")
    args = parser.parse_args(argv)
    out = stdout or sys.stdout
    lines: list[str] = []
    try:
        for name in args.files or ["-"]:
            lines.extend(_read(name, stdin or sys.stdin))
    except OSError as err:
        print(f"cannot read input: {err}", file=out)
        return 2
    report = analyze(lines)
    print(format_report(report), file=out)
    if report.lines == 0:
        return 2
    return 1 if report.errors or (args.strict and report.warnings) else 0


def _read(name: str, stdin: TextIO) -> list[str]:
    if name == "-":
        return stdin.read().splitlines()
    with open(name, encoding="utf-8", errors="replace", newline="") as handle:
        return handle.read().splitlines()


if __name__ == "__main__":
    sys.exit(main())
