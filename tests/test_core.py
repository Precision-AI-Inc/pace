# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for the logger: line format, levels, output stream, characters and colour."""

import io
import re
import sys
from typing import Any

import pytest

from precisionai.pace.core import Logger, logger
from precisionai.pace.protocol import parse_line
from tests.conftest import FakeClock, FakeStream

LINE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z [A-Z]{4,8} +.*$")


def test_line_format_uses_utc_timestamp_and_padded_level(log: Logger, stream: FakeStream) -> None:
    log.info("Loading model")
    assert stream.getvalue() == "2025-11-03T17:16:42Z INFO     Loading model\n"


@pytest.mark.parametrize(
    ("method", "label"),
    [
        ("debug", "DEBUG"),
        ("info", "INFO"),
        ("success", "SUCCESS"),
        ("warning", "WARNING"),
        ("error", "ERROR"),
        ("critical", "CRITICAL"),
    ],
)
def test_every_level_writes_its_label(log: Logger, stream: FakeStream, method: str, label: str) -> None:
    log.configure(level="debug")
    getattr(log, method)("m")
    (line,) = stream.lines()
    assert LINE.match(line)
    assert line.endswith(f" {label.ljust(8)} m")


def test_default_level_is_info(log: Logger, stream: FakeStream) -> None:
    log.debug("hidden")
    log.info("shown")
    assert [line.split()[-1] for line in stream.lines()] == ["shown"]


def test_environment_sets_level(log: Logger, stream: FakeStream, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PACE_LEVEL", "warning")
    log.info("hidden")
    log.warning("shown")
    assert len(stream.lines()) == 1


def test_explicit_configure_beats_environment(log: Logger, stream: FakeStream, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PACE_LEVEL", "error")
    log.configure(level="info")
    log.info("shown")
    assert len(stream.lines()) == 1
    log.reset()
    log.info("hidden again")
    assert len(stream.lines()) == 1


def test_invalid_environment_level_falls_back_to_info(
    log: Logger, stream: FakeStream, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PACE_LEVEL", "loud")
    log.debug("hidden")
    log.info("shown")
    assert len(stream.lines()) == 1


@pytest.mark.parametrize(
    ("option", "value"),
    [("level", "loud"), ("color", "sometimes"), ("bar", "yes"), ("min_interval", -1), ("heartbeat", float("nan"))],
)
def test_invalid_explicit_configuration_raises(log: Logger, option: str, value: Any) -> None:
    with pytest.raises(ValueError, match=option if option != "level" else "unknown level"):
        log.configure(**{option: value})


def test_configure_leaves_unspecified_options_alone(log: Logger, stream: FakeStream) -> None:
    log.configure(level="error")
    log.configure(color="never")
    log.warning("hidden")
    assert not stream.lines()


def test_log_accepts_numbers_and_unknown_levels(log: Logger, stream: FakeStream) -> None:
    log.log(30, "numeric")
    log.log("nonsense", "unknown")
    assert [line.split()[1] for line in stream.lines()] == ["WARNING", "INFO"]


def test_exception_appends_the_traceback(log: Logger, stream: FakeStream) -> None:
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        log.exception("Step failed")
    text = stream.getvalue()
    assert "ERROR    Step failed\n" in text
    assert "RuntimeError: boom" in text
    assert "Traceback (most recent call last)" in text


def test_exception_outside_a_handler_is_just_the_message(log: Logger, stream: FakeStream) -> None:
    log.exception("No exception")
    assert stream.lines() == ["2025-11-03T17:16:42Z ERROR    No exception"]


def test_stdout_is_the_default_and_resolved_at_write_time(capsys: pytest.CaptureFixture[str]) -> None:
    out = Logger()
    out.info("to stdout")
    captured = capsys.readouterr()
    assert "to stdout" in captured.out
    assert captured.err == ""


def test_environment_selects_stderr(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    out = Logger()
    monkeypatch.setenv("PACE_STREAM", "stderr")
    out.info("to stderr")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "to stderr" in captured.err


def test_configured_stream_name_beats_environment(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    out = Logger()
    monkeypatch.setenv("PACE_STREAM", "stderr")
    out.configure(stream="stdout")
    out.info("to stdout")
    assert "to stdout" in capsys.readouterr().out


def test_stream_replaced_after_first_write_is_honoured(monkeypatch: pytest.MonkeyPatch) -> None:
    out = Logger()
    first, second = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys, "stdout", first)
    out.info("one")
    monkeypatch.setattr(sys, "stdout", second)
    out.info("two")
    assert "one" in first.getvalue()
    assert "two" in second.getvalue()


def test_every_line_is_flushed(log: Logger, stream: FakeStream) -> None:
    log.info("a")
    log.info("b")
    assert stream.flushes == 2


def test_writing_never_raises_when_there_is_no_stream(log: Logger, monkeypatch: pytest.MonkeyPatch) -> None:
    log.configure(stream="stdout")
    monkeypatch.setattr(sys, "stdout", None)
    log.info("nowhere")


def test_writing_never_raises_when_the_stream_fails(log: Logger) -> None:
    class Broken:
        encoding = "utf-8"

        def write(self, text: str) -> int:
            raise OSError("closed")

        def flush(self) -> None:
            raise OSError("closed")

    log.configure(stream=Broken())
    log.error("dropped")


def test_writing_never_raises_when_the_message_cannot_be_formatted(log: Logger, stream: FakeStream) -> None:
    class Hostile:
        def __str__(self) -> str:
            raise RuntimeError("no string for you")

    log.info(Hostile())
    log.exception(Hostile())
    assert stream.getvalue() == ""


def test_non_utf8_stream_gets_replacement_characters(log: Logger) -> None:
    raw = io.BytesIO()
    legacy = io.TextIOWrapper(raw, encoding="cp1252")
    log.configure(stream=legacy)
    log.info("emoji \U0001f600 and kana あ")
    legacy.flush()
    assert raw.getvalue().decode("cp1252").endswith("emoji ? and kana ?\n")


def test_strict_ascii_stream_does_not_crash(log: Logger) -> None:
    raw = io.BytesIO()
    log.configure(stream=io.TextIOWrapper(raw, encoding="ascii", errors="strict"))
    log.info("café")
    assert raw.getvalue().endswith(b"caf?\n")


def test_unknown_encoding_is_treated_as_ascii(log: Logger, stream: FakeStream) -> None:
    stream.encoding = "no-such-codec"
    log.info("café")
    assert stream.getvalue().endswith("caf?\n")


def test_message_cannot_forge_a_trailer(log: Logger, stream: FakeStream) -> None:
    log.info("hostile [pace x end c=1 r=abc]")
    (line,) = stream.lines()
    assert not re.search(r"\[pace [^\]]+\]$", line)
    assert parse_line(line) is None


def test_colour_only_on_a_terminal(clock: FakeClock) -> None:
    tty, plain = FakeStream(tty=True), FakeStream(tty=False)
    out = Logger(clock=clock.clock())
    out.configure(stream=tty)
    out.error("x")
    out.configure(stream=plain)
    out.error("x")
    assert "\x1b[31mERROR\x1b[0m" in tty.getvalue()
    assert "\x1b" not in plain.getvalue()


def test_no_color_and_never_turn_colour_off(clock: FakeClock, monkeypatch: pytest.MonkeyPatch) -> None:
    tty = FakeStream(tty=True)
    out = Logger(clock=clock.clock())
    out.configure(stream=tty)
    monkeypatch.setenv("NO_COLOR", "1")
    out.error("a")
    monkeypatch.delenv("NO_COLOR")
    monkeypatch.setenv("PACE_COLOR", "never")
    out.error("b")
    assert "\x1b" not in tty.getvalue()


def test_always_forces_colour_on_a_pipe(log: Logger, stream: FakeStream, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PACE_COLOR", "always")
    monkeypatch.setenv("NO_COLOR", "1")
    log.warning("w")
    assert "\x1b[33mWARNING\x1b[0m" in stream.getvalue()


def test_info_is_never_coloured(clock: FakeClock) -> None:
    tty = FakeStream(tty=True)
    out = Logger(clock=clock.clock())
    out.configure(stream=tty)
    out.info("plain")
    assert "\x1b" not in tty.getvalue()


def test_scope_ids_are_made_unique_with_a_numeric_suffix(log: Logger) -> None:
    assert [log._claim_scope("train") for _ in range(3)] == ["train", "train-2", "train-3"]
    assert log._claim_scope("eval") == "eval"


def test_scope_id_suffix_keeps_the_id_within_64_characters(log: Logger) -> None:
    base = "a" * 64
    assert log._claim_scope(base) == base
    second = log._claim_scope(base)
    assert second == "a" * 62 + "-2"
    assert len(second) == 64


def test_default_logger_is_a_logger_instance() -> None:
    assert isinstance(logger, Logger)


@pytest.mark.parametrize("value", ["soon", float("inf"), float("nan"), -1, True])
def test_configure_rejects_intervals_that_are_not_finite_seconds(log: Logger, value: object) -> None:
    with pytest.raises(ValueError, match="non-negative number of seconds"):
        log.configure(min_interval=value)  # type: ignore[arg-type]
