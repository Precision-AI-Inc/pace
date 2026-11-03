# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for routing standard logging through the Pace logger."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import pytest

from precisionai.pace.capture import PaceHandler, capture_logging
from precisionai.pace.core import Logger
from tests.conftest import FakeStream


@pytest.fixture(autouse=True)
def restore_root_logger() -> Iterator[None]:
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    root.handlers[:] = handlers
    root.setLevel(level)


def test_importing_pace_does_not_touch_the_root_logger() -> None:
    assert not any(isinstance(handler, PaceHandler) for handler in logging.getLogger().handlers)


def test_records_are_routed_through_the_pace_logger(log: Logger, stream: FakeStream) -> None:
    capture_logging("INFO", target=log)
    logging.getLogger("somelib.trainer").info("loaded %d shards", 3)
    logging.getLogger("tensorlib").warning("slow path")
    assert [line.split(" ", 2)[1:] for line in stream.lines()] == [
        ["INFO", "    loaded 3 shards"],
        ["WARNING", " slow path"],
    ]


def test_root_level_is_lowered_so_records_reach_the_handler(log: Logger) -> None:
    logging.getLogger().setLevel(logging.WARNING)
    capture_logging("DEBUG", target=log)
    assert logging.getLogger().level == logging.DEBUG


def test_root_level_is_not_raised(log: Logger) -> None:
    logging.getLogger().setLevel(logging.DEBUG)
    capture_logging("ERROR", target=log)
    assert logging.getLogger().level == logging.DEBUG


def test_level_filters_records(log: Logger, stream: FakeStream) -> None:
    capture_logging("WARNING", target=log)
    logging.getLogger("lib").info("hidden")
    logging.getLogger("lib").error("shown")
    assert len(stream.lines()) == 1


def test_calling_twice_installs_one_handler_and_updates_the_level(log: Logger, stream: FakeStream) -> None:
    first = capture_logging("INFO", target=log)
    second = capture_logging("ERROR", target=log)
    assert first is second
    assert sum(isinstance(handler, PaceHandler) for handler in logging.getLogger().handlers) == 1
    logging.getLogger("lib").warning("hidden")
    assert stream.lines() == []


def test_exceptions_include_the_traceback(log: Logger, stream: FakeStream) -> None:
    capture_logging("INFO", target=log)
    try:
        raise ValueError("bad")
    except ValueError:
        logging.getLogger("lib").exception("failed")
    assert "ValueError: bad" in stream.getvalue()


def test_custom_numeric_levels_map_to_the_nearest_pace_level(log: Logger, stream: FakeStream) -> None:
    capture_logging("DEBUG", target=log)
    logging.getLogger("lib").log(25, "ok")
    assert stream.lines()[0].split()[1] == "SUCCESS"


def test_invalid_level_raises() -> None:
    with pytest.raises(ValueError, match="unknown level"):
        capture_logging("loud")


def test_a_failing_format_does_not_propagate(log: Logger) -> None:
    handler = capture_logging("INFO", target=log)

    class Hostile:
        def __str__(self) -> str:
            raise RuntimeError("no")

    logging.raiseExceptions = False
    try:
        logging.getLogger("lib").info("%s", Hostile())
    finally:
        logging.raiseExceptions = True
    assert handler in logging.getLogger().handlers


def test_default_target_is_the_module_logger(capsys: pytest.CaptureFixture[str]) -> None:
    capture_logging("INFO")
    logging.getLogger("lib").info("through the default")
    assert "through the default" in capsys.readouterr().out


LIBRARY = "fakelib"
LIBRARY_LOGGERS = (LIBRARY, f"{LIBRARY}.core", f"{LIBRARY}.fabric")


@pytest.fixture
def stubborn_library(capsys: pytest.CaptureFixture[str]) -> Iterator[list[logging.Logger]]:
    """Loggers set up the way Lightning's are: a console handler of their own and no propagation."""
    loggers = [logging.getLogger(name) for name in LIBRARY_LOGGERS]
    for each in loggers:
        each.setLevel(logging.INFO)
        each.addHandler(logging.StreamHandler())
        each.propagate = False
    yield loggers
    for each in loggers:
        each.handlers.clear()
        each.propagate = True
        each.setLevel(logging.NOTSET)


def test_a_library_that_does_not_propagate_never_reaches_pace_by_default(
    log: Logger, stream: FakeStream, stubborn_library: list[logging.Logger]
) -> None:
    capture_logging("INFO", target=log)
    logging.getLogger(f"{LIBRARY}.core").info("lost")
    assert stream.lines() == []


def test_take_over_makes_each_line_appear_once_in_pace_format(
    log: Logger,
    stream: FakeStream,
    stubborn_library: list[logging.Logger],
    capsys: pytest.CaptureFixture[str],
) -> None:
    capture_logging("INFO", target=log, take_over=LIBRARY)
    logging.getLogger(LIBRARY).info("from the parent")
    logging.getLogger(f"{LIBRARY}.core").info("from a child")
    logging.getLogger(f"{LIBRARY}.fabric").warning("from another")
    assert [line.split(" ", 2)[1:] for line in stream.lines()] == [
        ["INFO", "    from the parent"],
        ["INFO", "    from a child"],
        ["WARNING", " from another"],
    ]
    assert capsys.readouterr().err == ""
    assert all(each.propagate and not each.handlers for each in stubborn_library)


def test_take_over_accepts_several_names(
    log: Logger, stream: FakeStream, stubborn_library: list[logging.Logger]
) -> None:
    capture_logging("INFO", target=log, take_over=[f"{LIBRARY}.core", f"{LIBRARY}.fabric"])
    logging.getLogger(f"{LIBRARY}.core").info("one")
    logging.getLogger(f"{LIBRARY}.fabric").info("two")
    assert len(stream.lines()) == 2


def test_take_over_also_frees_the_parents_that_would_swallow_the_record(
    log: Logger, stream: FakeStream, stubborn_library: list[logging.Logger], capsys: pytest.CaptureFixture[str]
) -> None:
    capture_logging("INFO", target=log, take_over=f"{LIBRARY}.core")
    logging.getLogger(f"{LIBRARY}.core").info("once")
    assert len(stream.lines()) == 1
    assert capsys.readouterr().err == ""
    assert not stubborn_library[0].handlers
    assert stubborn_library[0].propagate
    assert stubborn_library[2].handlers


def test_take_over_keeps_file_and_custom_handlers(
    log: Logger, stream: FakeStream, stubborn_library: list[logging.Logger], tmp_path: Path
) -> None:
    kept = logging.FileHandler(tmp_path / "library.log", encoding="utf-8")
    custom = logging.NullHandler()
    stubborn_library[0].addHandler(kept)
    stubborn_library[0].addHandler(custom)
    try:
        capture_logging("INFO", target=log, take_over=LIBRARY)
        assert stubborn_library[0].handlers == [kept, custom]
        logging.getLogger(LIBRARY).info("both")
        assert "both" in (tmp_path / "library.log").read_text(encoding="utf-8")
        assert len(stream.lines()) == 1
    finally:
        kept.close()


def test_take_over_does_not_touch_a_logger_with_a_similar_name(
    log: Logger, stubborn_library: list[logging.Logger]
) -> None:
    sibling = logging.getLogger(f"{LIBRARY}x")
    sibling.addHandler(logging.StreamHandler())
    sibling.propagate = False
    try:
        capture_logging("INFO", target=log, take_over=LIBRARY)
        assert sibling.handlers
        assert not sibling.propagate
    finally:
        sibling.handlers.clear()
        sibling.propagate = True


def test_take_over_can_be_repeated_for_loggers_created_later(log: Logger) -> None:
    capture_logging("INFO", target=log, take_over="latelib")
    late = logging.getLogger("latelib.worker")
    late.addHandler(logging.StreamHandler())
    late.propagate = False
    try:
        capture_logging("INFO", target=log, take_over="latelib")
        assert late.propagate
        assert not late.handlers
    finally:
        late.handlers.clear()
        late.propagate = True


def test_a_bad_take_over_changes_nothing(stubborn_library: list[logging.Logger]) -> None:
    root_handlers = list(logging.getLogger().handlers)
    with pytest.raises(ValueError, match="take_over needs logger names"):
        capture_logging("INFO", take_over=[LIBRARY, ""])
    with pytest.raises(ValueError, match="take_over needs logger names"):
        capture_logging("INFO", take_over=[1])  # type: ignore[list-item]
    assert logging.getLogger().handlers == root_handlers
    assert all(each.handlers and not each.propagate for each in stubborn_library)
