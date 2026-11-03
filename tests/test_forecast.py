# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.pace.forecast (interface baseline)."""

from __future__ import annotations

import pytest

from precisionai.pace.forecast import ForecastState, ScopeForecast
from precisionai.pace.protocol import Event, Record


def _rec(event: Event, seq: int, elapsed: float, completed: float | None = None, **kw: object) -> Record:
    return Record(scope="train", event=event, seq=seq, elapsed=elapsed, completed=completed, **kw)  # type: ignore[arg-type]


def test_unknown_before_any_record() -> None:
    assert ScopeForecast("train").snapshot(0.0).state is ForecastState.UNKNOWN


def test_unknown_without_total() -> None:
    fc = ScopeForecast("train")
    fc.observe(_rec(Event.START, 1, 0.0, 0, unit="step"), 0.0)
    assert fc.snapshot(0.0).state is ForecastState.UNKNOWN


def test_learning_before_any_work_done() -> None:
    fc = ScopeForecast("train")
    fc.observe(_rec(Event.START, 1, 0.0, 0, total=100, unit="step"), 0.0)
    assert fc.snapshot(0.0).state is ForecastState.LEARNING


def test_estimates_remaining_from_rate() -> None:
    fc = ScopeForecast("train")
    fc.observe(_rec(Event.START, 1, 0.0, 0, total=100, unit="step"), 0.0)
    fc.observe(_rec(Event.UPDATE, 2, 10.0, 25, total=100), 10.0)
    snap = fc.snapshot(10.0)
    assert snap.state is ForecastState.LIVE
    assert snap.remaining_seconds == pytest.approx(30.0)
    assert snap.unit == "step"


def test_resume_measures_rate_from_resumed_point() -> None:
    fc = ScopeForecast("train")
    fc.observe(_rec(Event.START, 1, 0.0, 50, total=100, resumed=True), 0.0)
    fc.observe(_rec(Event.UPDATE, 2, 10.0, 60, total=100), 10.0)
    assert fc.snapshot(10.0).remaining_seconds == pytest.approx(40.0)


def test_end_and_fail_are_terminal() -> None:
    done = ScopeForecast("train")
    done.observe(_rec(Event.END, 5, 40.0, 100, total=100), 40.0)
    assert done.snapshot(41.0).state is ForecastState.COMPLETE
    failed = ScopeForecast("train")
    failed.observe(_rec(Event.FAIL, 5, 40.0), 40.0)
    assert failed.snapshot(41.0).state is ForecastState.FAILED


def test_ignores_other_scopes() -> None:
    fc = ScopeForecast("train")
    fc.observe(Record(scope="eval", event=Event.END, seq=1, elapsed=1.0, completed=1, total=1), 1.0)
    assert fc.snapshot(1.0).state is ForecastState.UNKNOWN
