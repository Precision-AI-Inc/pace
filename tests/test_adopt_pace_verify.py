"""Tests for the adopt-pace skill's verify.py, loaded by path because skills/ is not a package."""

import importlib.util
import io
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from precisionai.pace import Event, ForecastState, Record, format_trailer, phase, plan, progress, track
from precisionai.pace.core import Logger
from tests.conftest import FakeClock, FakeStream

VERIFY_PATH = Path(__file__).resolve().parent.parent / "skills" / "adopt-pace" / "verify.py"


def load_verify() -> ModuleType:
    spec = importlib.util.spec_from_file_location("adopt_pace_verify", VERIFY_PATH)
    assert spec
    assert isinstance(spec.loader, SourceFileLoader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


verify = load_verify()


def line(record: Record, stamp: str = "2026-11-03T10:00:00Z") -> str:
    return f"{stamp} PROGRESS text  {format_trailer(record)}"


def start(scope: str = "train", **kwargs: object) -> Record:
    values: dict = {"scope": scope, "event": Event.START, "elapsed": 0.0, "completed": 0, "total": 10}
    values.update({"unit": "step", "label": "Train", "run": "abc"})
    values.update(kwargs)
    return Record(**values)


def update(scope: str = "train", seq: int = 2, completed: float = 5, **kwargs: object) -> Record:
    values: dict = {"scope": scope, "event": Event.UPDATE, "seq": seq, "elapsed": float(seq), "completed": completed}
    values.update({"total": 10, "run": "abc"})
    values.update(kwargs)
    return Record(**values)


def end(scope: str = "train", seq: int = 9, event: Event = Event.END, **kwargs: object) -> Record:
    values: dict = {"scope": scope, "event": event, "elapsed": float(seq), "completed": 10, "total": 10, "run": "abc"}
    values.update(kwargs)
    return Record(**values)


def run_lines(*records: Record) -> list[str]:
    return [line(record) for record in records]


def messages(report: Any, level: str) -> list[str]:
    return [finding.message for finding in report.findings if finding.level == level]


def test_clean_run_with_a_root_passes() -> None:
    report = verify.analyze(run_lines(start(root=True), update(root=True), end(root=True)))

    assert report.roots == ["train"]
    assert report.records == 3
    assert not report.errors
    assert not report.warnings
    assert report.scopes["train"].status == "end"
    assert report.scopes["train"].completed == 10
    assert report.estimate.state is ForecastState.COMPLETE
    assert "PASS: 0 warning(s)" in verify.format_report(report)


def test_output_without_a_trailer_is_an_error() -> None:
    report = verify.analyze(["just text", "more text"])

    assert messages(report, "error") == ["no progress records found: Pace never wrote to this output"]
    assert "FAIL: 1 error(s)" in verify.format_report(report)


def test_a_trailer_that_does_not_parse_is_reported_with_its_line_number() -> None:
    lines = [*run_lines(start(root=True)), "2026-11-03T10:00:00Z PROGRESS broken  [pace:1 s=train k=bogus]"]
    lines.extend(run_lines(end(root=True)))

    report = verify.analyze(lines)

    assert report.malformed == [2]
    assert any("does not parse (lines 2)" in text for text in messages(report, "error"))


def test_two_roots_is_an_error() -> None:
    report = verify.analyze(
        run_lines(start("a", root=True), start("b", root=True), end("a", root=True), end("b", root=True))
    )

    assert any("2 scopes are marked root (a, b)" in text for text in messages(report, "error"))


def test_a_single_top_level_scope_without_root_is_a_warning() -> None:
    report = verify.analyze(run_lines(start(), end()))

    assert not report.errors
    assert any("not marked root=True" in text for text in messages(report, "warning"))


def test_several_scopes_without_root_or_plan_is_a_warning() -> None:
    report = verify.analyze(run_lines(start("a"), end("a"), start("b"), end("b")))

    assert any("only estimate the current phase" in text for text in messages(report, "warning"))


def test_a_scope_that_never_ends_is_an_error() -> None:
    report = verify.analyze(run_lines(start(root=True), update(root=True)))

    assert any("never ended or failed" in text for text in messages(report, "error"))
    assert report.scopes["train"].status == "open"


def test_a_failed_scope_is_a_warning_and_the_job_reads_failed() -> None:
    report = verify.analyze(run_lines(start(root=True), end(root=True, event=Event.FAIL, completed=4)))

    assert any("ended with fail" in text for text in messages(report, "warning"))
    assert report.estimate.state is ForecastState.FAILED
    assert "recognised as failed" in verify.format_report(report)


def test_an_unknown_total_is_a_warning() -> None:
    report = verify.analyze(
        run_lines(
            start(root=True, total=None),
            end(root=True, total=None),
        )
    )

    assert any("the root scope has no total" in text for text in messages(report, "warning"))
    assert "?" in verify.format_report(report)


def test_a_non_root_scope_with_unknown_total_names_the_scope() -> None:
    report = verify.analyze(run_lines(start("load", total=None), end("load", total=None)))

    assert any("scope 'load' has no total" in text for text in messages(report, "warning"))


def test_a_secret_looking_or_long_label_is_a_warning() -> None:
    report = verify.analyze(
        run_lines(
            start("a", root=True, label="Train with api_key"),
            end("a", root=True),
            start("b", label="x" * 64),
            end("b"),
        )
    )

    warnings = messages(report, "warning")
    assert any("may contain a secret" in text for text in warnings)
    assert any("64 characters" in text for text in warnings)


def test_a_plan_whose_names_match_the_scopes_is_clean() -> None:
    phases = (("prep", 0.5), ("train", 0.5))
    lines = run_lines(
        Record(scope="plan", event=Event.PLAN, seq=1, phases=phases, run="abc"),
        start("prep"),
        end("prep"),
        start("train"),
        end("train"),
    )

    report = verify.analyze(lines)

    assert not report.errors
    assert not report.warnings
    assert report.estimate.state is ForecastState.COMPLETE
    assert "plan: prep:0.5, train:0.5" in verify.format_report(report)


def test_a_plan_name_with_no_scope_is_an_error() -> None:
    phases = (("prep", 0.5), ("trian", 0.5))
    lines = run_lines(
        Record(scope="plan", event=Event.PLAN, seq=1, phases=phases, run="abc"),
        start("prep"),
        end("prep"),
    )

    report = verify.analyze(lines)

    assert any("'trian' but no scope has that id" in text for text in messages(report, "error"))


def test_unplanned_work_after_the_last_planned_phase_is_a_warning() -> None:
    phases = (("train", 1.0),)
    lines = run_lines(
        Record(scope="plan", event=Event.PLAN, seq=1, phases=phases, run="abc"),
        start("train"),
        end("train"),
        start("export"),
        end("export"),
    )

    report = verify.analyze(lines)

    assert any("export ran after the last planned phase ended" in text for text in messages(report, "warning"))


def test_work_after_the_root_ended_is_a_warning() -> None:
    report = verify.analyze(run_lines(start("main", root=True), end("main", root=True), start("export"), end("export")))

    assert any("root 'main' ended while export was still" in text for text in messages(report, "warning"))


def test_work_still_running_when_the_root_ended_is_a_warning() -> None:
    report = verify.analyze(run_lines(start("main", root=True), start("upload"), end("main", root=True), end("upload")))

    assert any("root 'main' ended while upload was still" in text for text in messages(report, "warning"))


def test_nested_scopes_after_the_root_ended_are_not_flagged() -> None:
    report = verify.analyze(
        run_lines(
            start("main", root=True), start("val", parent="main"), end("val", parent="main"), end("main", root=True)
        )
    )

    assert not any("ended before" in text for text in messages(report, "warning"))


def test_duplicate_records_are_ignored() -> None:
    lines = run_lines(start(root=True), start(root=True), end(root=True))

    report = verify.analyze(lines)

    assert report.duplicates == 1
    assert report.records == 2
    assert "1 duplicate record(s) ignored" in verify.format_report(report)


def test_a_record_read_late_is_ignored() -> None:
    report = verify.analyze(
        run_lines(start(root=True), update(seq=5, root=True), update(seq=3, root=True), end(root=True))
    )

    assert report.duplicates == 1
    assert not report.errors


def test_a_late_start_after_the_updates_is_a_duplicate_not_a_restart() -> None:
    # Read from both streams, one after the other: every line comes twice, the start late.
    lines = run_lines(start(root=True), update(root=True), end(root=True))

    report = verify.analyze(lines + lines)

    assert report.duplicates == 3
    assert report.scopes["train"].starts == 1
    assert report.estimate.state is ForecastState.COMPLETE


def test_a_restart_with_a_new_run_id_is_not_a_duplicate_or_disorder() -> None:
    lines = run_lines(
        start(root=True, run="first"),
        end(root=True, event=Event.FAIL, run="first"),
        start(root=True, run="second", resumed=True, completed=5),
        end(root=True, run="second"),
    )

    report = verify.analyze(lines)

    assert report.duplicates == 0
    assert not report.errors
    assert report.scopes["train"].starts == 2
    assert report.scopes["train"].resumed
    assert "root, resumed" in verify.format_report(report)


def test_carriage_return_redraws_are_split_into_lines() -> None:
    text = "\r".join(run_lines(start(root=True), end(root=True)))

    report = verify.analyze([text])

    assert report.records == 2


def test_blank_and_ansi_coloured_lines_are_handled() -> None:
    coloured = "\x1b[32m" + line(start(root=True)) + "\x1b[0m"

    report = verify.analyze(["", coloured, line(end(root=True))])

    assert report.lines == 3
    assert report.records == 2


def test_a_bad_timestamp_is_ignored() -> None:
    report = verify.analyze([line(start(root=True), stamp="2026-13-45T99:00:00Z"), line(end(root=True), stamp="none")])

    assert report.records == 2
    assert report.estimate.state is ForecastState.COMPLETE


def test_a_long_enough_run_reaches_a_live_estimate() -> None:
    lines = [
        line(start(root=True, total=100), "2026-11-03T10:00:00Z"),
        line(update(seq=2, completed=10, elapsed=30.0, total=100, root=True), "2026-11-03T10:00:30Z"),
        line(update(seq=3, completed=20, elapsed=60.0, total=100, root=True), "2026-11-03T10:01:00Z"),
        line(update(seq=4, completed=30, elapsed=90.0, total=100, root=True), "2026-11-03T10:01:30Z"),
        line(update(seq=5, completed=40, elapsed=120.0, total=100, root=True), "2026-11-03T10:02:00Z"),
    ]

    report = verify.analyze(lines)
    text = verify.format_report(report)

    assert report.reached_live
    assert report.sources == {"root"}
    assert report.estimate.state is ForecastState.LIVE
    assert "left from root" in text
    assert "reached a live estimate" in text


def test_a_short_run_says_it_never_went_live() -> None:
    text = verify.format_report(verify.analyze(run_lines(start(root=True), update(root=True))))

    assert "never reached a live estimate" in text


def test_real_logger_output_passes(log: Logger, stream: FakeStream, clock: FakeClock) -> None:
    plan(["prepare", "train"], logger=log)
    with progress(4, "Prepare", "file", scope="prepare", logger=log) as run:
        for _ in range(4):
            clock.advance(20)
            run.advance()
    for _ in track(range(6), "Train", "step", scope="train", logger=log):
        clock.advance(20)

    report = verify.analyze(stream.lines())

    assert not report.errors
    assert not report.warnings
    assert set(report.scopes) == {"prepare", "train"}
    assert report.estimate.state is ForecastState.COMPLETE


def test_real_logger_phase_wrapper_is_fine(log: Logger, stream: FakeStream) -> None:
    plan(["train"], logger=log)
    with phase("train", logger=log):
        for _ in track(range(3), "Steps", "step", logger=log):
            pass

    report = verify.analyze(stream.lines())

    assert not messages(report, "error")
    assert report.estimate.state is ForecastState.COMPLETE


@pytest.mark.parametrize(
    ("seconds", "text"),
    [(None, "-"), (5, "5s"), (125, "2m05s"), (3725, "1h02m")],
)
def test_durations_are_formatted_compactly(seconds: float | None, text: str) -> None:
    assert verify._seconds(seconds) == text


def test_main_reads_a_file_and_returns_zero_on_pass(tmp_path: Path) -> None:
    log_file = tmp_path / "run.log"
    log_file.write_text("\n".join(run_lines(start(root=True), end(root=True))) + "\n", encoding="utf-8")
    out = io.StringIO()

    code = verify.main([str(log_file)], stdout=out)

    assert code == 0
    assert "PASS" in out.getvalue()


def test_main_reads_standard_input_and_returns_one_on_error() -> None:
    out = io.StringIO()

    code = verify.main([], stdin=io.StringIO("\n".join(run_lines(start(root=True))) + "\n"), stdout=out)

    assert code == 1
    assert "FAIL" in out.getvalue()


def test_main_strict_fails_on_a_warning() -> None:
    stdin = io.StringIO("\n".join(run_lines(start(), end())) + "\n")

    assert verify.main(["-", "--strict"], stdin=stdin, stdout=io.StringIO()) == 1
    stdin = io.StringIO("\n".join(run_lines(start(), end())) + "\n")
    assert verify.main(["-"], stdin=stdin, stdout=io.StringIO()) == 0


def test_main_returns_two_for_a_missing_file_or_empty_input(tmp_path: Path) -> None:
    out = io.StringIO()

    assert verify.main([str(tmp_path / "missing.log")], stdout=out) == 2
    assert "cannot read input" in out.getvalue()
    assert verify.main([], stdin=io.StringIO(""), stdout=io.StringIO()) == 2


def test_a_restart_without_run_ids_is_not_hidden_as_a_duplicate() -> None:
    lines = [
        "[pace:1 s=train root=1 k=start c=0 t=10 u=step e=0 q=1]",
        "[pace:1 s=train root=1 k=upd c=5 t=10 e=2 q=2]",
        "[pace:1 s=train root=1 k=start c=0 t=10 u=step e=0 q=1]",
        "[pace:1 s=train root=1 k=upd c=5 t=10 e=2 q=2]",
        "[pace:1 s=train root=1 k=end c=10 t=10 e=3 q=3]",
    ]

    report = verify.analyze(lines)

    assert report.duplicates == 0
    assert report.scopes["train"].starts == 2


def test_the_same_start_read_twice_without_run_ids_is_a_duplicate() -> None:
    report = verify.analyze(
        [
            "[pace:1 s=train root=1 k=start c=0 t=10 u=step e=0 q=1]",
            "[pace:1 s=train root=1 k=start c=0 t=10 u=step e=0 q=1]",
            "[pace:1 s=train root=1 k=end c=10 t=10 e=2 q=2]",
        ]
    )

    assert report.duplicates == 1
