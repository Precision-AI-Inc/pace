# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for routing standard logging through the Pace logger."""

from __future__ import annotations

import logging
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
