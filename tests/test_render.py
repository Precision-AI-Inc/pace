# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for the readable part of output lines."""

from __future__ import annotations

import pytest

from precisionai.pace import _render as render
from precisionai.pace.forecast import Forecast, ForecastState
from precisionai.pace.protocol import Event, Record


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0, "0s"),
        (44.4, "44s"),
        (59.6, "1m00s"),
        (1317, "21m57s"),
        (3600, "1h00m"),
        (3725, "1h02m"),
        (90_000, "1d1h"),
        (-5, "0s"),
    ],
)
def test_format_duration(seconds: float, expected: str) -> None:
    assert render.format_duration(seconds) == expected


@pytest.mark.parametrize(
    ("rate", "expected"),
    [(2.4, "2.4 step/s"), (0.0312, "0.0312 step/s"), (12.34, "12.3 step/s"), (1234.5, "1234 step/s")],
)
def test_format_rate(rate: float, expected: str) -> None:
    assert render.format_rate(rate, "step") == expected


@pytest.mark.parametrize(("value", "expected"), [(1840.0, "1840"), (0.0, "0"), (2.5, "2.5"), (1 / 3, "0.3333")])
def test_format_count(value: float, expected: str) -> None:
    assert render.format_count(value) == expected


def test_bar_is_fixed_width_and_clamped() -> None:
    for fraction in (-1, 0, 0.5, 1, 7):
        assert len(render.format_bar(fraction, render.ASCII)) == render.BAR_WIDTH
    assert render.format_bar(0.5, render.ASCII) == "#" * 10 + "-" * 10
    assert render.format_bar(9, render.UNICODE) == "━" * 20


def test_charset_follows_the_encoding() -> None:
    assert render.charset_for("UTF-8") is render.UNICODE
    assert render.charset_for("utf8") is render.UNICODE
    assert render.charset_for("cp1252") is render.ASCII
    assert render.charset_for("no-such-codec") is render.ASCII


def test_encode_safely_replaces_and_survives_unknown_codecs() -> None:
    assert render.encode_safely("a·b", "ascii") == "a?b"
    assert render.encode_safely("a·b", "utf-8") == "a·b"
    assert render.encode_safely("a·b", "no-such-codec") == "a?b"
    assert render.encode_safely("lone \ud800 surrogate", "utf-8") == "lone ? surrogate"


def test_timestamp_is_iso_utc_to_the_second() -> None:
    assert render.format_timestamp(1_762_190_202.9) == "2025-11-03T17:16:42Z"


def test_clean_text_makes_one_line_and_cuts() -> None:
    assert render.clean_text("a\n\tb\x00c  d") == "a b c d"
    assert render.clean_text("x" * 200) == "x" * render.MAX_DESCRIPTION_CHARS
    assert render.clean_text("see [pace:1 s=a]") == "see [ pace:1 s=a]"


def test_log_line_pads_the_label_and_neutralises_markers() -> None:
    assert render.log_line("T", "INFO", "hi", False) == "T INFO     hi"
    assert render.log_line("T", "CRITICAL", "hi", False) == "T CRITICAL hi"
    assert render.log_line("T", "ERROR", "x", True) == "T \x1b[31mERROR\x1b[0m    x"


def test_colorize_leaves_uncoloured_labels_alone() -> None:
    assert render.colorize("INFO", True) == "INFO"
    assert render.colorize("ERROR", False) == "ERROR"


def test_plan_text_shows_normalised_shares() -> None:
    assert render.plan_text((("a", 1.0), ("b", 3.0))) == "Plan  a 25%, b 75%"
    assert render.plan_text(()) == "Plan  "


def test_fit_cuts_with_an_ellipsis() -> None:
    assert render.fit("short", 10) == "short"
    assert render.fit("0123456789", 6) == "012..."
    assert render.fit("0123456789", 6, render.UNICODE) == "01234…"
    assert render.fit("0123456789", 0) == ""


def record(event: Event, completed: float = 50, total: float | None = 100, elapsed: float = 10) -> Record:
    return Record(scope="job", event=event, seq=1, elapsed=elapsed, completed=completed, total=total)


@pytest.mark.parametrize(
    ("event", "forecast", "expected"),
    [
        (Event.END, Forecast(ForecastState.COMPLETE, 0.0), "done in 10s"),
        (Event.FAIL, Forecast(ForecastState.FAILED), "failed after 10s"),
        (Event.UPDATE, Forecast(ForecastState.LIVE, 120.0, 1.0), "~2m00s"),
        (Event.UPDATE, Forecast(ForecastState.STALE, 120.0, 1.0), "~2m00s (stale)"),
        (Event.UPDATE, Forecast(ForecastState.BOOTSTRAP, 120.0), "~2m00s (guess)"),
        (Event.UPDATE, Forecast(ForecastState.LEARNING), "estimating..."),
        (Event.UPDATE, Forecast(ForecastState.LEARNING, 5.0), "estimating..."),
        (Event.UPDATE, Forecast(ForecastState.UNKNOWN), "no estimate"),
        (Event.UPDATE, Forecast(ForecastState.LIVE), "no estimate"),
    ],
)
def test_estimate_text(event: Event, forecast: Forecast, expected: str) -> None:
    assert render.estimate_text(record(event), forecast, render.ASCII) == expected


def test_progress_text_with_a_known_total() -> None:
    forecast = Forecast(ForecastState.LIVE, 30.0, 5.0)
    text = render.progress_text(record(Event.UPDATE), forecast, "Train", "step", render.UNICODE)
    assert text == "Train  50.0%  50/100 step · 5 step/s · ~30s"


def test_progress_text_with_an_unknown_total_has_no_percent() -> None:
    item = record(Event.UPDATE, completed=7, total=None)
    text = render.progress_text(item, Forecast(ForecastState.UNKNOWN), "Rows", "row", render.ASCII)
    assert text == "Rows  7 row | no estimate"


def test_progress_text_falls_back_to_the_scope_id_and_a_default_unit() -> None:
    forecast = Forecast(ForecastState.LIVE, 1.0, 2.0)
    text = render.progress_text(record(Event.UPDATE), forecast, " ", "", render.ASCII)
    assert text.startswith("job  50.0%  50/100 | 2 it/s")


def test_progress_text_with_a_bar() -> None:
    text = render.progress_text(
        record(Event.UPDATE), Forecast(ForecastState.LEARNING), "X", "u", render.ASCII, bar=True
    )
    assert text == "X  " + "#" * 10 + "-" * 10 + "  50.0%  50/100 u | estimating..."
