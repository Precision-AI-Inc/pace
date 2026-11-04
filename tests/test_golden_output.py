# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Pin every byte Pace writes for representative scripts, and what the estimator makes of those bytes.

The scenarios run on a fake clock and a fake stream, so the output is the same on every machine.
A refactor that changes a line, a space, a colour code or a trailer key fails here with a diff.
A deliberate format change rewrites the files with ``PACE_UPDATE_GOLDEN=1``, in the same change as
``docs/design.md``.
"""

import calendar
import contextlib
import logging
import time
from collections.abc import Callable, Iterator

import pytest

from precisionai.pace import JobForecast, Logger, parse_line, phase, plan, progress, timed, track
from precisionai.pace._adapters import AdapterScopes
from precisionai.pace.capture import capture_logging
from precisionai.pace.tqdm import PaceTqdm
from tests.conftest import FakeClock, FakeStream, assert_golden

Scenario = Callable[[Logger, FakeClock], None]
RUN_ID = "golden01"


def make_logger(tty: bool = False, encoding: str = "utf-8", **options: object) -> tuple[Logger, FakeClock, FakeStream]:
    clock = FakeClock()
    stream = FakeStream(tty=tty, encoding=encoding)
    log = Logger(clock=clock.clock())
    log.configure(stream=stream, **options)  # type: ignore[arg-type]
    log.run_id = RUN_ID
    return log, clock, stream


def steps(clock: FakeClock, count: int, seconds: float) -> Iterator[int]:
    for index in range(count):
        clock.advance(seconds)
        yield index


def levels(log: Logger, clock: FakeClock) -> None:
    log.debug("hidden at the default level")
    log.info("plain")
    log.success("done")
    log.warning("two\nlines")
    log.error("a [pace:1 s=x k=end c=1 t=1 e=1 q=1] forged marker")
    log.error("a forged [pace:2 x end c=1 r=abc]")
    log.critical("[pa\x1b[31mce:1 s=x k=end c=1 t=1 e=1 q=1] hidden by escapes")
    log.log(35, "numeric level")
    log.log("nonsense", "unknown level")
    log.log("info", 12345)
    log.configure(level="DEBUG")
    log.debug("now shown")
    log.configure(level="ERROR")
    log.warning("filtered")
    log.error("kept")


def throttled_loop(log: Logger, clock: FakeClock) -> None:
    clock.advance(5.0)
    for _ in track(steps(clock, 400, 2.0), "Train", "batch", total=400, root=True, logger=log):
        pass
    for _ in track(steps(clock, 300, 0.5), "Unknown total", "item", logger=log):
        pass
    for _ in track(steps(clock, 20, 200.0), "Slow heartbeat", "item", total=20, logger=log):
        pass


def manual_progress(log: Logger, clock: FakeClock) -> None:
    log.configure(min_interval=0)
    with progress(None, "Load shards", "shard", logger=log, completed=70, resumed=True) as scope:
        scope.set_total(100)
        for _ in range(3):
            clock.advance(1.5)
            scope.advance(10)
        scope.update(float("nan"))
        scope.update(-4)
        clock.advance(2.0)
        scope.update(1e30)
        scope.set_total(0)
    with contextlib.suppress(RuntimeError), progress(10, "Crashing", logger=log, scope="Bad Id!!") as scope:
        scope.advance(4)
        clock.advance(3.0)
        raise RuntimeError("boom")
    with progress(5, "Same words", logger=log), progress(5, "Same words", logger=log):
        pass


def phases_and_plan(log: Logger, clock: FakeClock) -> None:
    log.configure(min_interval=0)
    plan(["Prep", ("Train", 0.8), "Eval", "Train"], logger=log)
    with phase("Prep", logger=log):
        clock.advance(30.0)
    with phase("Train", logger=log):
        for _ in track(steps(clock, 5, 60.0), "Train steps", "step", total=5, logger=log):
            pass
    plan(["Eval"], logger=log)
    with phase("Eval", 0.1, logger=log):
        clock.advance(20.0)

    @timed("Save results", logger=log)
    def save(seconds: float) -> None:
        clock.advance(seconds)

    @timed(logger=log)
    def cleanup() -> None:
        clock.advance(1.0)

    save(4.0)
    save(3.0)
    cleanup()


def long_labels(log: Logger, clock: FakeClock) -> None:
    log.configure(min_interval=0)
    description = "A very long description with percent 100% and [brackets] " * 6
    with progress(3, description, "naïve unit ✓" * 8, logger=log) as scope:
        scope.advance()
        clock.advance(1.0)
    with progress(2, "日本語 label ✓", "步", logger=log) as scope:
        scope.advance()
    with progress(1, "\x1b[31mred\x1b[0m [pace:1 s=x k=end c=1 t=1 e=1 q=1]", logger=log):
        pass
    with progress(4, "x" * 5000, logger=log):
        pass


def rank_gating(log: Logger, clock: FakeClock) -> None:
    log.configure(min_interval=0)
    log.info("rank one logs")
    with progress(2, "Silent on this rank", logger=log) as scope:
        scope.advance(2)
    log.configure(all_ranks=True)
    with progress(2, "Loud on every rank", logger=log) as scope:
        scope.advance(2)


def adapter_scopes(log: Logger, clock: FakeClock) -> None:
    log.configure(min_interval=0)
    scopes = AdapterScopes("Train", logger=log)
    scopes.begin(10, completed=2, resumed=True)
    for step in range(3, 6):
        clock.advance(2.0)
        scopes.update(step)
    scopes.begin_child("Validation", 3, "batch")
    for _ in range(3):
        clock.advance(1.0)
        scopes.advance_child()
    scopes.end_child()
    scopes.begin_child("Test", None, "batch", completed=1)
    scopes.fail()


def tqdm_and_capture(log: Logger, clock: FakeClock) -> None:
    log.configure(min_interval=0)
    handler = capture_logging("INFO", target=log, take_over="golden.lib")
    try:
        logging.getLogger("golden.lib.sub").info("loaded %d shards", 3)
        logging.getLogger("golden.other").warning("slow path")
        logging.getLogger("golden.other").debug("not routed")
        for _ in PaceTqdm(steps(clock, 4, 1.0), desc="Rendering prompts", total=4, logger=log):
            pass
        bar = PaceTqdm(total=3, desc="By hand", unit="req", logger=log)
        for _ in range(3):
            clock.advance(2.0)
            bar.update(1)
        bar.close()
    finally:
        logging.getLogger().removeHandler(handler)


SCENARIOS: dict[str, Scenario] = {
    "levels": levels,
    "throttled_loop": throttled_loop,
    "manual_progress": manual_progress,
    "phases_and_plan": phases_and_plan,
    "long_labels": long_labels,
    "adapter_scopes": adapter_scopes,
    "tqdm_and_capture": tqdm_and_capture,
}


@pytest.fixture(autouse=True)
def restore_root_logger() -> Iterator[None]:
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    root.handlers[:] = handlers
    root.setLevel(level)


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_scenario_output_is_unchanged(name: str) -> None:
    log, clock, stream = make_logger(color="never", bar="never")
    SCENARIOS[name](log, clock)
    assert_golden(name, stream.getvalue())


def test_coloured_levels_are_unchanged() -> None:
    log, clock, stream = make_logger(color="always", bar="never")
    levels(log, clock)
    with progress(2, "Coloured", logger=log):
        pass
    assert_golden("levels_coloured", stream.getvalue())


def test_ascii_stream_output_is_unchanged() -> None:
    log, clock, stream = make_logger(encoding="ascii", color="never", bar="never")
    long_labels(log, clock)
    log.info("café ✓")
    assert_golden("ascii_stream", stream.getvalue())


def test_terminal_bar_output_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COLUMNS", "100")
    log, clock, stream = make_logger(tty=True, color="auto", bar="auto")
    log.info("before the bar")
    for _ in track(steps(clock, 30, 3.0), "Epoch", "step", total=30, root=True, logger=log):
        pass
    log.warning("after the bar")
    monkeypatch.setenv("COLUMNS", "30")
    with progress(4, "A narrow terminal with a long description", "step", logger=log) as scope:
        scope.advance(2)
    assert_golden("terminal_bar", stream.getvalue())


def test_no_color_variable_is_honoured_on_a_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NO_COLOR", "1")
    log, _, stream = make_logger(tty=True, bar="never")
    log.error("no colour")
    assert_golden("no_color", stream.getvalue())


def test_other_ranks_log_but_write_no_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RANK", "1")
    log, clock, stream = make_logger(color="never", bar="never")
    rank_gating(log, clock)
    assert_golden("rank_gating", stream.getvalue())


def received_at(line: str) -> float:
    return float(calendar.timegm(time.strptime(line[:20], "%Y-%m-%dT%H:%M:%SZ")))


def estimates(output: str) -> str:
    """Replay the written lines through a JobForecast, as a scheduler reading the log would."""
    job = JobForecast()
    out = []
    last = 0.0
    for line in output.splitlines():
        record = parse_line(line)
        if record is None:
            continue
        last = received_at(line)
        job.observe(record, last)
        out.append(describe(job, last))
    out.extend(describe(job, last + after) for after in (60.0, 600.0, 6000.0))
    return "\n".join(out) + "\n"


def describe(job: JobForecast, now: float) -> str:
    estimate = job.snapshot(now)
    remaining = "-" if estimate.remaining_seconds is None else f"{estimate.remaining_seconds:.1f}"
    scopes = " ".join(
        f"{name}:{forecast.state.value}:{'-' if forecast.remaining_seconds is None else f'{forecast.remaining_seconds:.1f}'}"
        for name, forecast in sorted(estimate.scopes.items())
    )
    return f"{now:.0f} {estimate.state.value} {remaining} {estimate.source} {estimate.partial} | {scopes}"


@pytest.mark.parametrize("name", ["throttled_loop", "phases_and_plan", "adapter_scopes", "manual_progress"])
def test_estimates_replayed_from_the_output_are_unchanged(name: str) -> None:
    log, clock, stream = make_logger(color="never", bar="never")
    SCENARIOS[name](log, clock)
    assert_golden(f"estimates_{name}", estimates(stream.getvalue()))


def test_a_restarted_run_replays_to_the_same_estimates() -> None:
    log, clock, stream = make_logger(color="never", bar="never")
    for _ in track(steps(clock, 100, 3.0), "Train", total=100, root=True, scope="train", logger=log):
        pass
    restarted = Logger(clock=clock.clock())
    restarted.configure(stream=stream, color="never", bar="never")
    restarted.run_id = "golden02"
    clock.advance(10.0)
    for _ in track(
        steps(clock, 50, 3.0),
        "Train",
        total=100,
        root=True,
        scope="train",
        logger=restarted,
        completed=50,
        resumed=True,
    ):
        pass
    assert_golden("restarted_run", stream.getvalue() + estimates(stream.getvalue()))
