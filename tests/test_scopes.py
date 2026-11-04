# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for progress scopes: events, emission rate, nesting, ids, ranks and the terminal bar."""

import io
import re
import threading
import time

import pytest

from precisionai.pace import scopes
from precisionai.pace.core import Logger
from precisionai.pace.core import logger as default_logger
from precisionai.pace.forecast import Forecast, ForecastState
from precisionai.pace.protocol import MAX_LINE_BYTES, MAX_NUMBER, Record, parse_line
from precisionai.pace.scopes import (
    Phase,
    Scope,
    derive_scope_id,
    phase,
    plan,
    progress,
    timed,
    track,
    validate_scope_id,
)
from tests.conftest import FakeClock, FakeStream

TRAILER = re.compile(r"\[pace:2 [^\]]+\]$")
KINDS = {"start", "end", "fail", "step", "plan"}


def fields(line: str) -> dict[str, str]:
    """Return a trailer as a dict: ``s``, ``p`` and ``k`` (``upd`` when there is no event word), each flag as ``1``, then the fields."""
    match = TRAILER.search(line)
    assert match, f"no trailer on {line!r}"
    head, *items = match.group(0)[len("[pace:2 ") : -1].split(" ")
    scope, _, parent = head.partition("<")
    out = {"s": scope, "k": "upd"} | ({"p": parent} if parent else {})
    for item in items:
        key, sep, value = item.partition("=")
        if sep:
            out[key] = value
        elif key in KINDS:
            out["k"] = key
        else:
            out[key] = "1"
    return out


def records(stream: FakeStream) -> list[dict[str, str]]:
    return [fields(line) for line in stream.lines()]


def kinds(stream: FakeStream) -> list[str]:
    return [item["k"] for item in records(stream)]


@pytest.fixture(autouse=True)
def spy_on_trailers(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    produced: list[str] = []
    real = scopes.format_trailer

    def spy(record: Record) -> str:
        text = real(record)
        produced.append(text)
        return text

    monkeypatch.setattr(scopes, "format_trailer", spy)
    return produced


def test_progress_writes_start_update_end(log: Logger, clock: FakeClock, stream: FakeStream) -> None:
    with progress(100, "Train", "step", logger=log) as run:
        clock.advance(20)
        run.advance(30)
    assert kinds(stream) == ["start", "upd", "end"]
    start, update, end = records(stream)
    assert start == {"s": "train", "k": "start", "t": "100", "u": "step", "n": "Train", "r": log.run_id}
    assert update == {"s": "train", "k": "upd", "c": "30", "t": "100", "e": "20", "q": "2", "r": log.run_id}
    assert end == {"s": "train", "k": "end", "c": "30", "t": "100", "e": "20", "r": log.run_id}


def test_every_progress_line_ends_with_a_trailer_format_trailer_produced(
    log: Logger, clock: FakeClock, stream: FakeStream, spy_on_trailers: list[str]
) -> None:
    log.configure(min_interval=0, heartbeat=0)
    with progress(10, "A trailer check", logger=log, root=True) as run:
        for _ in range(10):
            clock.advance(5)
            run.advance()
    plan(["a", "b"], logger=log)
    lines = stream.lines()
    assert len(lines) == len(spy_on_trailers) > 5
    for line, trailer in zip(lines, spy_on_trailers, strict=True):
        assert TRAILER.search(line)
        assert line.endswith(trailer)
        assert trailer.isascii()
        assert "\x1b" not in trailer


def test_line_text_shows_percent_count_rate_and_estimate(log: Logger, clock: FakeClock, stream: FakeStream) -> None:
    # The estimator needs 60 s, 3 intervals and 1% of the work before it is Live.
    with progress(5000, "Train", "step", logger=log) as run:
        for done in (400, 800, 1200, 1600):
            clock.advance(20)
            run.update(done)
        clock.advance(20)
        run.update(1840)
    update = stream.lines()[5]
    # Four intervals at 20 step/s, then 12: the weighted average gives 0.7 x 20 + 0.3 x 12 = 17.6,
    # and 3160 steps left / 17.6 = 3m00s.
    assert update.startswith("2025-11-03T17:18:22Z PROGRESS Train  1840/5000  36.8%  17.6 step/s · ~3m00s  [pace:2 ")


def test_failure_writes_fail_and_reraises(log: Logger, stream: FakeStream) -> None:
    with pytest.raises(RuntimeError, match="boom"), progress(10, "Job", logger=log):
        raise RuntimeError("boom")
    assert kinds(stream) == ["start", "fail"]
    assert "failed after" in stream.lines()[-1]


def test_unknown_total_writes_question_mark_and_no_percent(log: Logger, clock: FakeClock, stream: FakeStream) -> None:
    with progress(None, "Stream", "row", logger=log) as run:
        clock.advance(200)
        run.advance(7)
    start, update, _ = records(stream)
    assert start["t"] == "?"
    assert update["t"] == "?"
    assert "%" not in stream.lines()[1].split("[pace")[0]


def test_resumed_scope_starts_above_zero(log: Logger, stream: FakeStream) -> None:
    with progress(100, "Resume", logger=log, completed=40, resumed=True):
        pass
    start = records(stream)[0]
    assert start["c"] == "40"
    assert start["res"] == "1"


def test_root_and_parent_are_written_on_every_record(log: Logger, clock: FakeClock, stream: FakeStream) -> None:
    with progress(10, "Whole", logger=log, root=True), progress(10, "Inner", logger=log) as inner:
        clock.advance(60)
        inner.advance(5)
    by_scope = {}
    for item in records(stream):
        by_scope.setdefault(item["s"], []).append(item)
    assert all(item["root"] == "1" for item in by_scope["whole"])
    assert all(item["p"] == "whole" for item in by_scope["inner"])
    assert all("p" not in item for item in by_scope["whole"])


def test_update_before_start_starts_the_scope(log: Logger, stream: FakeStream) -> None:
    handle = progress(3, "Manual", logger=log)
    handle.advance()
    handle.end()
    handle.end()
    handle.fail()
    assert kinds(stream) == ["start", "end"]


def test_updates_after_the_end_are_ignored(log: Logger, clock: FakeClock, stream: FakeStream) -> None:
    handle = progress(3, "Done", logger=log)
    handle.start()
    handle.end()
    clock.advance(100)
    handle.advance()
    assert kinds(stream) == ["start", "end"]


def test_invalid_numbers_are_ignored_not_raised(log: Logger, stream: FakeStream) -> None:
    handle = progress(-5, "Odd", logger=log, completed=float("nan"))
    handle.advance(float("inf"))
    handle.update(-3)
    handle.update("many")  # type: ignore[arg-type]
    handle.advance(None)  # type: ignore[arg-type]
    handle.set_total(0)
    handle.end()
    start, end = records(stream)
    assert start["t"] == end["t"] == "?"
    assert "c" not in end


def test_set_total_changes_the_total_and_unknown_total_can_become_known(
    log: Logger, clock: FakeClock, stream: FakeStream
) -> None:
    with progress(None, "Late total", logger=log) as run:
        run.set_total(200)
        clock.advance(20)
        run.advance(10)
        run.set_total(None)
    assert [item.get("t") for item in records(stream)] == ["?", "200", "?"]


def test_update_can_go_backwards(log: Logger, clock: FakeClock, stream: FakeStream) -> None:
    with progress(100, "Back", logger=log) as run:
        run.update(50)
        clock.advance(20)
        run.update(10)
    assert records(stream)[-1]["c"] == "10"


def test_description_with_newlines_cannot_break_the_line(log: Logger, stream: FakeStream) -> None:
    with progress(1, "Bad\nname [pace:1 s=x k=end]", logger=log):
        pass
    for line in stream.lines():
        assert line.count("[pace:") == 1
    assert len(stream.lines()) == 2


def test_nothing_raises_when_the_stream_fails(log: Logger) -> None:
    class Broken:
        encoding = "utf-8"

        def write(self, text: str) -> int:
            raise OSError("gone")

        def flush(self) -> None:
            raise OSError("gone")

    log.configure(stream=Broken())
    with progress(2, "Broken", logger=log) as run:
        run.advance()


# ---- emission rate ---------------------------------------------------------------------------


def test_updates_wait_for_the_minimum_interval(log: Logger, clock: FakeClock, stream: FakeStream) -> None:
    with progress(1000, "Rate", logger=log) as run:
        for _ in range(14):
            clock.advance(1)
            run.advance(100)
        assert kinds(stream) == ["start"]
        clock.advance(1)
        run.advance(1)
        assert kinds(stream) == ["start", "upd"]


def test_updates_need_one_percent_more_progress(log: Logger, clock: FakeClock, stream: FakeStream) -> None:
    with progress(1000, "Percent", logger=log) as run:
        clock.advance(30)
        run.advance(9)
        assert kinds(stream) == ["start"]
        clock.advance(0.1)
        run.advance(1)
        assert kinds(stream) == ["start", "upd"]


def test_heartbeat_forces_a_line_without_progress(log: Logger, clock: FakeClock, stream: FakeStream) -> None:
    with progress(1000, "Heartbeat", logger=log) as run:
        clock.advance(119)
        run.advance(0)
        assert kinds(stream) == ["start"]
        clock.advance(1)
        run.advance(0)
        assert kinds(stream) == ["start", "upd"]


def test_unknown_total_writes_only_heartbeats(log: Logger, clock: FakeClock, stream: FakeStream) -> None:
    with progress(None, "Unbounded", logger=log) as run:
        clock.advance(60)
        run.advance(1_000_000)
        assert kinds(stream) == ["start"]
        clock.advance(60)
        run.advance()
        assert kinds(stream) == ["start", "upd"]


def test_interval_and_heartbeat_come_from_the_environment(
    log: Logger, clock: FakeClock, stream: FakeStream, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PACE_MIN_INTERVAL", "2")
    monkeypatch.setenv("PACE_HEARTBEAT", "5")
    with progress(1000, "Env", logger=log) as run:
        clock.advance(3)
        run.advance(0)
        assert kinds(stream) == ["start"]
        clock.advance(2)
        run.advance(0)
        assert kinds(stream) == ["start", "upd"]


def test_explicit_interval_beats_the_environment(
    log: Logger, clock: FakeClock, stream: FakeStream, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PACE_MIN_INTERVAL", "1000")
    log.configure(min_interval=0, heartbeat=0)
    with progress(1000, "Explicit", logger=log) as run:
        run.advance()
        run.advance()
    assert kinds(stream) == ["start", "upd", "upd", "end"]


def test_invalid_environment_interval_uses_the_default(
    log: Logger, clock: FakeClock, stream: FakeStream, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PACE_MIN_INTERVAL", "soon")
    with progress(1000, "Default", logger=log) as run:
        clock.advance(14)
        run.advance(500)
    assert kinds(stream) == ["start", "end"]


def test_a_million_fast_updates_write_few_lines(log: Logger, clock: FakeClock, stream: FakeStream) -> None:
    with progress(1_000_000, "Tight loop", logger=log) as run:
        for _ in range(2000):
            clock.advance(0.01)
            run.advance(500)
    assert len(stream.lines()) < 10


# ---- track, timed, phase, plan ---------------------------------------------------------------


def test_track_uses_len_and_ends_when_exhausted(log: Logger, stream: FakeStream) -> None:
    assert list(track([1, 2, 3], "Items", "item", logger=log)) == [1, 2, 3]
    start, end = records(stream)[0], records(stream)[-1]
    assert start["t"] == "3"
    assert end["k"] == "end"
    assert end["c"] == "3"


def test_track_without_len_has_unknown_total(log: Logger, stream: FakeStream) -> None:
    assert list(track(iter(range(2)), "Gen", logger=log)) == [0, 1]
    assert records(stream)[0]["t"] == "?"


def test_track_total_argument_wins(log: Logger, stream: FakeStream) -> None:
    list(track(range(3), "Explicit", total=10, logger=log))
    assert records(stream)[0]["t"] == "10"


def test_track_starts_when_iteration_begins(log: Logger, stream: FakeStream) -> None:
    items = track(range(3), "Lazy", logger=log)
    assert stream.lines() == []
    next(items)
    assert kinds(stream) == ["start"]
    items.close()  # type: ignore[attr-defined]


def test_track_break_writes_end_with_the_amount_reached(log: Logger, stream: FakeStream) -> None:
    for item in track(range(10), "Early stop", logger=log):
        if item == 4:
            break
    assert kinds(stream) == ["start", "end"]
    assert records(stream)[-1]["c"] == "4"


def test_track_iterable_error_writes_fail(log: Logger, stream: FakeStream) -> None:
    def source():
        yield 1
        raise ValueError("bad batch")

    with pytest.raises(ValueError, match="bad batch"):
        list(track(source(), "Failing", logger=log))
    assert kinds(stream) == ["start", "fail"]
    assert "1/3" not in stream.lines()[-1]
    assert "Failing  1  failed after" in stream.lines()[-1]


def test_track_scope_is_the_parent_of_scopes_in_the_loop(log: Logger, stream: FakeStream) -> None:
    for _ in track(range(1), "Epochs", logger=log):
        with progress(1, "Batches", logger=log):
            pass
    inner = [item for item in records(stream) if item["s"] == "batches"]
    assert {item["p"] for item in inner} == {"epochs"}
    with progress(1, "After", logger=log):
        pass
    assert all("p" not in item for item in records(stream) if item["s"] == "after")


def test_timed_with_description(log: Logger, clock: FakeClock, stream: FakeStream) -> None:
    @timed("Load model", logger=log)
    def load(path: str) -> str:
        clock.advance(7)
        return path

    assert load("weights") == "weights"
    assert load.__name__ == "load"
    # It ended before its first line was due (PACE_MIN_INTERVAL, 15 s), so it is one line.
    (step,) = records(stream)
    assert step == {"s": "load-model", "k": "step", "e": "7", "n": "Load%20model", "r": log.run_id}
    assert stream.lines()[0].startswith("2025-11-03T17:16:49Z STEP     Load model  done in 7s  [pace:2 ")


def test_timed_without_a_minimum_interval_writes_start_and_end(
    log: Logger, clock: FakeClock, stream: FakeStream
) -> None:
    log.configure(min_interval=0)

    @timed("Load model", logger=log)
    def load() -> None:
        clock.advance(7)

    load()
    start, end = records(stream)
    assert (start["s"], start["k"], start["t"]) == ("load-model", "start", "1")
    assert (end["k"], end["c"], end["e"]) == ("end", "1", "7")


def test_steps_nest_under_a_running_loop(log: Logger, clock: FakeClock, stream: FakeStream) -> None:
    @timed("Save", logger=log)
    def save() -> None:
        clock.advance(2)

    for _ in track(range(2), "Train", "step", logger=log):
        clock.advance(60)
        save()
    lines = stream.lines()
    assert [item["k"] for item in records(stream)] == ["start", "step", "upd", "step", "upd", "end"]
    assert [item.get("p") for item in records(stream) if item["k"] == "step"] == ["train", "train"]
    assert " STEP     ├ Save  done in 2s  [pace:2 save<train step e=2 n=Save r=" in lines[1]


def test_a_failing_step_writes_its_start_and_fail(log: Logger, clock: FakeClock, stream: FakeStream) -> None:
    @timed("Explode", logger=log)
    def explode() -> None:
        clock.advance(3)
        raise KeyError("x")

    with pytest.raises(KeyError, match="x"):
        explode()
    start, fail = records(stream)
    assert start == {"s": "explode", "k": "start", "t": "1", "u": "call", "e": "3", "n": "Explode", "r": log.run_id}
    assert fail == {"s": "explode", "k": "fail", "t": "1", "e": "3", "r": log.run_id}


def test_a_child_line_writes_the_pending_start_of_its_phase_first(
    log: Logger, clock: FakeClock, stream: FakeStream
) -> None:
    with phase("Load", logger=log):
        clock.advance(2)
        with progress(3, "Shards", "shard", logger=log) as shards:
            shards.advance(3)
    assert [(item["s"], item["k"], item.get("e")) for item in records(stream)] == [
        ("load", "start", "2"),
        ("shards", "start", None),
        ("shards", "end", None),
        ("load", "end", "2"),
    ]


def test_a_step_still_running_when_its_line_is_due_writes_its_start_then(
    log: Logger, clock: FakeClock, stream: FakeStream
) -> None:
    download = Phase("Download", unit="call", logger=log)
    download.start()
    assert stream.lines() == []
    clock.advance(16)
    # The logger's timer runs this once PACE_MIN_INTERVAL has passed in real time.
    log.run_due(time.monotonic() + 20)
    clock.advance(4)
    download.completed = 1
    download.end()
    assert [(item["k"], item.get("e")) for item in records(stream)] == [("start", "16"), ("end", "20")]


def test_the_logger_timer_writes_a_pending_start_on_its_own(clock: FakeClock) -> None:
    written = threading.Event()

    class Signal(FakeStream):
        def write(self, text: str) -> int:
            written.set()
            return super().write(text)

    stream = Signal()
    log = Logger(clock=clock.clock())
    log.configure(stream=stream, min_interval=0.01)
    download = Phase("Download", logger=log)
    download.start()
    assert written.wait(timeout=10)
    download.end()
    assert kinds(stream) == ["start", "end"]


def test_a_step_cancelled_before_its_timer_writes_nothing_twice(log: Logger, stream: FakeStream) -> None:
    with phase("Quick", logger=log):
        pass
    assert log.run_due(time.monotonic() + 100) is None
    assert kinds(stream) == ["step"]


def test_a_terminal_writes_a_phase_as_before(clock: FakeClock) -> None:
    tty = FakeStream(tty=True)
    out = Logger(clock=clock.clock())
    out.configure(stream=tty, color="never")
    with phase("Quick", logger=out):
        clock.advance(1)
    (line,) = tty.lines()[-1:]
    assert fields(line)["k"] == "end"


def test_timed_without_arguments_uses_the_function_name(stream: FakeStream) -> None:
    default_logger.configure(stream=stream)

    @timed
    def warm_up() -> None:
        return None

    warm_up()
    assert records(stream)[0]["s"] == "warm_up"
    assert kinds(stream) == ["step"]


def test_timed_fails_and_reraises(log: Logger, stream: FakeStream) -> None:
    @timed("Explode", logger=log)
    def explode() -> None:
        raise KeyError("x")

    with pytest.raises(KeyError, match="x"):
        explode()
    assert kinds(stream) == ["start", "fail"]


def test_timed_second_call_gets_a_new_scope_id(log: Logger, stream: FakeStream) -> None:
    @timed("Twice", logger=log)
    def twice() -> None:
        return None

    twice()
    twice()
    assert [item["s"] for item in records(stream) if item["k"] == "step"] == ["twice", "twice-2"]


def test_phase_is_the_parent_of_nested_scopes_and_nesting_restores(log: Logger, stream: FakeStream) -> None:
    with phase("Preprocess", weight=0.05, logger=log) as outer:
        assert isinstance(outer, Phase)
        with phase("Tokenise", logger=log), progress(5, "Shards", logger=log):
            pass
        with progress(5, "Second", logger=log):
            pass
    parents = {item["s"]: item.get("p") for item in records(stream)}
    assert parents == {"preprocess": None, "tokenise": "preprocess", "shards": "tokenise", "second": "preprocess"}
    ends = {item["s"]: item for item in records(stream) if item["k"] == "end"}
    assert ends["preprocess"]["c"] == "1"


def test_phase_failure_is_not_marked_complete(log: Logger, stream: FakeStream) -> None:
    with pytest.raises(RuntimeError, match="stop"), phase("Doomed", logger=log):
        raise RuntimeError("stop")
    assert records(stream)[-1]["k"] == "fail"
    assert "0/1  failed after" in stream.lines()[-1]


def test_scope_opened_in_a_new_thread_has_no_parent(log: Logger, stream: FakeStream) -> None:
    def work() -> None:
        with progress(1, "Threaded", logger=log):
            pass

    with phase("Main", logger=log):
        thread = threading.Thread(target=work)
        thread.start()
        thread.join()
    threaded = [item for item in records(stream) if item["s"] == "threaded"]
    assert all("p" not in item for item in threaded)


def test_plan_lists_phases_with_weights(log: Logger, stream: FakeStream) -> None:
    prep = phase("Preprocess", 0.05, logger=log)
    plan([prep, ("Train", 0.9), "Evaluate"], logger=log)
    (line,) = stream.lines()
    item = fields(line)
    assert item["k"] == "plan"
    assert item["ph"] == "preprocess:0.05,train:0.9,evaluate:0.05"
    assert item["q"] == "1"
    assert "c" not in item
    assert line.split("[pace")[0].endswith("Plan  preprocess 5%, train 90%, evaluate 5%  ")


def test_plan_without_weights_shares_equally(log: Logger, stream: FakeStream) -> None:
    plan(["a", "b", "c", "d"], logger=log)
    assert fields(stream.lines()[0])["ph"] == "a:0.25,b:0.25,c:0.25,d:0.25"


def test_plan_names_match_the_phases_that_follow(log: Logger, stream: FakeStream) -> None:
    plan(["Fine tune"], logger=log)
    with phase("Fine tune", logger=log):
        pass
    assert fields(stream.lines()[0])["ph"].split(":")[0] == records(stream)[1]["s"] == "fine-tune"


def test_plan_with_repeated_names_matches_the_phase_scopes(log: Logger, stream: FakeStream) -> None:
    plan(["Train", ("Train", 0.5)], logger=log)
    with phase("Train", logger=log):
        pass
    with phase("Train", logger=log):
        pass
    planned = [pair.split(":")[0] for pair in fields(stream.lines()[0])["ph"].split(",")]
    started = [item["s"] for item in records(stream) if item["k"] in ("start", "step")]
    assert planned == started == ["train", "train-2"]


def test_repeated_plans_share_a_scope_in_q_order(log: Logger, stream: FakeStream) -> None:
    plan(["a"], logger=log)
    plan(["a", "b"], logger=log)
    assert [(item["s"], item["q"]) for item in records(stream)] == [("plan", "1"), ("plan", "2")]


def test_a_plan_of_long_names_stays_within_the_parser_limit(log: Logger, stream: FakeStream) -> None:
    plan([f"phase-{index:02d}-" + "x" * 55 for index in range(32)], logger=log)
    (line,) = stream.lines()
    assert len(line.encode("utf-8")) <= MAX_LINE_BYTES
    record = parse_line(line)
    assert record is not None
    assert len(record.phases) == 32


def test_a_rank_change_applies_to_the_next_line(
    log: Logger, clock: FakeClock, stream: FakeStream, monkeypatch: pytest.MonkeyPatch
) -> None:
    with progress(100, "Train", "step", logger=log) as run:
        monkeypatch.setenv("RANK", "1")
        clock.advance(20)
        run.advance(50)
        monkeypatch.setenv("RANK", "0")
    assert kinds(stream) == ["start", "end"]


# ---- scope ids -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("description", "expected"),
    [
        ("Load model", "load-model"),
        ("  Train / eval!  ", "train-eval"),
        ("", "scope"),
        ("???", "scope"),
        ("v1.2_final", "v1.2_final"),
        ("x" * 100, "x" * 48),
        ("café au lait", "caf-au-lait"),
    ],
)
def test_derived_ids(description: str, expected: str) -> None:
    assert derive_scope_id(description) == expected


@pytest.mark.parametrize(
    ("scope", "expected"),
    [("Train.v2", "Train.v2"), ("my scope", "my-scope"), ("a" * 70, "a" * 64), ("///", "scope")],
)
def test_explicit_ids_are_validated_and_sanitised(scope: str, expected: str) -> None:
    assert validate_scope_id(scope) == expected


def test_explicit_id_is_used_and_made_unique(log: Logger, stream: FakeStream) -> None:
    with progress(1, "Whatever", scope="job.main", logger=log):
        pass
    with progress(1, "Whatever", scope="job.main", logger=log):
        pass
    assert {item["s"] for item in records(stream)} == {"job.main", "job.main-2"}


def test_same_description_gets_distinct_ids(log: Logger, stream: FakeStream) -> None:
    for _ in range(3):
        with progress(1, "Epoch", logger=log):
            pass
    assert [item["s"] for item in records(stream) if item["k"] == "start"] == ["epoch", "epoch-2", "epoch-3"]


def test_scope_class_is_exported_for_adapters(log: Logger, stream: FakeStream) -> None:
    handle = Scope("Adapter", 10, "step", logger=log)
    handle.start()
    handle.fail()
    assert kinds(stream) == ["start", "fail"]


# ---- threads ---------------------------------------------------------------------------------


def test_advance_from_many_threads_keeps_seq_strictly_increasing(log: Logger, stream: FakeStream) -> None:
    log.configure(min_interval=0, heartbeat=0)
    handle = progress(None, "Parallel", logger=log)
    handle.start()

    def work() -> None:
        for _ in range(50):
            handle.advance()

    threads = [threading.Thread(target=work) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    handle.end()
    lines = records(stream)
    seqs = [int(item["q"]) for item in lines[1:-1]]
    assert (lines[0]["k"], lines[-1]["k"]) == ("start", "end")
    assert seqs == list(range(2, len(lines)))
    assert lines[-1]["c"] == "400"


# ---- ranks -----------------------------------------------------------------------------------


@pytest.mark.parametrize("variable", ["RANK", "OMPI_COMM_WORLD_RANK", "PMI_RANK", "SLURM_PROCID", "LOCAL_RANK"])
def test_only_rank_zero_writes_progress_but_every_rank_logs(
    log: Logger, stream: FakeStream, monkeypatch: pytest.MonkeyPatch, variable: str
) -> None:
    monkeypatch.setenv(variable, "3")
    with progress(2, "Sharded", logger=log) as run:
        run.advance()
    plan(["a"], logger=log)
    log.info("still logged")
    assert [line.split()[1] for line in stream.lines()] == ["INFO"]


def test_rank_variables_are_read_in_priority_order(
    log: Logger, stream: FakeStream, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "5")
    with progress(1, "Global rank wins", logger=log):
        pass
    assert kinds(stream) == ["start", "end"]


def test_all_ranks_override(log: Logger, stream: FakeStream, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("PACE_ALL_RANKS", "1")
    with progress(1, "Everyone", logger=log):
        pass
    assert kinds(stream) == ["start", "end"]


def test_all_ranks_configure_beats_environment(
    log: Logger, stream: FakeStream, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("PACE_ALL_RANKS", "1")
    log.configure(all_ranks=False)
    with progress(1, "Nobody", logger=log):
        pass
    assert stream.lines() == []


# ---- characters and terminal -----------------------------------------------------------------


def test_non_utf8_stream_gets_ascii_text_and_an_intact_trailer(log: Logger, clock: FakeClock) -> None:
    raw = io.BytesIO()
    legacy = io.TextIOWrapper(raw, encoding="cp1252")
    log.configure(stream=legacy)
    with progress(100, "Entraînement \U0001f680", "step", logger=log) as run:
        clock.advance(30)
        run.advance(50)
    legacy.flush()
    text = raw.getvalue().decode("cp1252")
    lines = text.splitlines()
    assert len(lines) == 3
    assert "·" not in text
    assert "…" not in text
    assert "  50/100  50.0%  estimating..." in lines[1]
    assert "Entraînement ?" in lines[0]
    for line in lines:
        assert TRAILER.search(line)


def test_utf8_stream_gets_unicode_glyphs(log: Logger, clock: FakeClock, stream: FakeStream) -> None:
    with progress(10, "Glyphs", logger=log), progress(10, "Inner", logger=log) as inner:
        clock.advance(30)
        inner.advance()
    assert "PROGRESS ├ Inner   1/10  10.0%  estimating…" in stream.lines()[2]


def test_stream_without_encoding_gets_ascii(log: Logger, clock: FakeClock) -> None:
    sink = io.StringIO()
    log.configure(stream=sink)
    with progress(10, "Plain", logger=log), progress(10, "Inner", logger=log) as inner:
        clock.advance(30)
        inner.advance()
    assert "PROGRESS - Inner   1/10  10.0%  estimating..." in sink.getvalue()


def test_lines_show_what_each_event_needs(log: Logger, clock: FakeClock, stream: FakeStream) -> None:
    with progress(128, "Generate", "request", logger=log) as run:
        with progress(4, "Batch", "prompt", logger=log) as batch:
            batch.advance(4)
        clock.advance(20)
        run.advance(66)
        clock.advance(20)
        run.advance(62)
    texts = [line[len("2025-11-03T17:16:42Z ") : line.index("  [pace:2 ")] for line in stream.lines()]
    assert texts == [
        "PROGRESS Generate    0/128 request",
        "PROGRESS ├ Batch  0/4 prompt",
        "PROGRESS ├ Batch  4/4  done in 0s",
        "PROGRESS Generate   66/128  51.6%  estimating…",
        "PROGRESS Generate  128/128  100.0%  estimating…",
        "PROGRESS Generate  128/128  done in 40s · 3.2 request/s",
    ]


def test_terminal_redraws_one_bar_line_without_a_trailer(clock: FakeClock) -> None:
    tty = FakeStream(tty=True)
    out = Logger(clock=clock.clock())
    out.configure(stream=tty)
    with progress(100, "Bar", "step", logger=out) as run:
        clock.advance(1)
        run.advance(25)
        clock.advance(1)
        run.advance(25)
    parts = tty.getvalue().split("\r")
    bars = [part for part in parts if part.strip() and "[pace" not in part]
    assert len(bars) == 3
    assert not any("\n" in bar for bar in bars)
    assert "Bar " in bars[1]
    assert "50.0%" in bars[2]
    assert "\u2501" in bars[2]
    final = parts[-1].strip()
    assert fields(final)["k"] == "end"
    assert "done in 2s" in final


def test_terminal_redraw_is_limited_to_ten_per_second(clock: FakeClock) -> None:
    tty = FakeStream(tty=True)
    out = Logger(clock=clock.clock())
    out.configure(stream=tty)
    with progress(1000, "Fast", logger=out) as run:
        for _ in range(100):
            clock.advance(0.01)
            run.advance()
    redraws = tty.getvalue().count("\r")
    assert 8 <= redraws <= 14


def test_log_lines_do_not_collide_with_the_bar(clock: FakeClock) -> None:
    tty = FakeStream(tty=True)
    out = Logger(clock=clock.clock())
    out.configure(stream=tty, color="never")
    with progress(10, "Busy", logger=out) as run:
        clock.advance(1)
        run.advance(5)
        out.info("hello")
        clock.advance(1)
        run.advance(1)
    text = tty.getvalue()
    before, _, after = text.partition("hello")
    assert before.rstrip().endswith("INFO")
    erase = before.rsplit("\r", 2)
    assert set(erase[-2]) == {" "}
    assert "6/10 it" in after


def test_terminal_ascii_bar_on_legacy_encoding(clock: FakeClock) -> None:
    tty = FakeStream(tty=True, encoding="cp1252")
    out = Logger(clock=clock.clock())
    out.configure(stream=tty)
    with progress(4, "Legacy", logger=out) as run:
        clock.advance(1)
        run.advance(2)
    assert "##########----------" in tty.getvalue()
    assert "━" not in tty.getvalue()


def test_bar_can_be_turned_off_so_a_scheduler_still_sees_trailers(
    clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    tty = FakeStream(tty=True)
    out = Logger(clock=clock.clock())
    out.configure(stream=tty)
    monkeypatch.setenv("PACE_BAR", "never")
    with progress(10, "Piped by Nomad", logger=out):
        pass
    assert "\r" not in tty.getvalue()
    assert [item["k"] for item in [fields(line) for line in tty.lines()]] == ["start", "end"]


def test_terminal_plan_is_a_normal_line(clock: FakeClock) -> None:
    tty = FakeStream(tty=True)
    out = Logger(clock=clock.clock())
    out.configure(stream=tty)
    plan(["a", "b"], logger=out)
    assert fields(tty.lines()[0])["k"] == "plan"


def test_long_bar_line_is_cut_to_the_terminal_width(clock: FakeClock, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COLUMNS", "30")
    tty = FakeStream(tty=True)
    out = Logger(clock=clock.clock())
    out.configure(stream=tty)
    with progress(10, "A rather long description of a job", logger=out):
        pass
    first = tty.getvalue().split("\r")[1]
    assert len(first) <= 29
    assert first.endswith("…")


# ---- estimate display ------------------------------------------------------------------------


class StubForecast:
    result = Forecast(ForecastState.UNKNOWN)

    def __init__(self, scope: str) -> None:
        self.scope = scope

    def observe(self, record: Record, received_at: float) -> None:
        return None

    def snapshot(self, now: float) -> Forecast:
        return self.result


@pytest.mark.parametrize(
    ("forecast", "shown"),
    [
        (Forecast(ForecastState.UNKNOWN), "no estimate"),
        (Forecast(ForecastState.LEARNING), "estimating…"),
        (Forecast(ForecastState.BOOTSTRAP, 90.0, 1.0), "~1m30s (guess)"),
        (Forecast(ForecastState.LIVE, 3725.0, 2.0), "~1h02m"),
        (Forecast(ForecastState.STALE, 45.0, 2.0), "~45s (stale)"),
        (Forecast(ForecastState.LIVE, None, None), "no estimate"),
        (Forecast(ForecastState.COMPLETE, 0.0), "no estimate"),
    ],
)
def test_every_forecast_state_has_a_display(
    log: Logger, stream: FakeStream, monkeypatch: pytest.MonkeyPatch, forecast: Forecast, shown: str
) -> None:
    StubForecast.result = forecast
    monkeypatch.setattr(scopes, "ScopeForecast", StubForecast)
    log.configure(min_interval=0)
    handle = progress(10, "States", "step", logger=log)
    handle.start()
    handle.advance()
    assert shown in stream.lines()[-1]


def test_a_number_passed_as_the_unit_is_an_error(log: Logger) -> None:
    # tqdm(iterable, desc, total) ordering moved over to track() unchanged.
    with pytest.raises(TypeError, match="pass the total as total="):
        track(range(3), "Train", 3000, logger=log)  # type: ignore[arg-type]


def test_a_total_too_large_for_a_trailer_is_reported_as_unknown(log: Logger, stream: FakeStream) -> None:
    with progress(1e16, "Huge", "item", logger=log) as run:
        run.advance(5)
    start = parse_line(stream.lines()[0])
    assert start is not None
    assert start.total is None


def test_a_count_pushed_past_the_trailer_limit_is_capped(log: Logger, clock: FakeClock, stream: FakeStream) -> None:
    with progress(None, "Huge", "item", logger=log) as run:
        run.update(1e16)
    end = parse_line(stream.lines()[-1])
    assert end is not None
    assert end.completed == MAX_NUMBER
