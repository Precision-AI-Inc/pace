# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Estimate remaining time from progress records.

The same code serves the terminal's "time remaining" and a scheduler reading a
job's logs, so the two never disagree. The rules are specified in
``docs/design.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from precisionai.pace.protocol import Event, Record


class ForecastState(str, Enum):
    """How much a forecast can be trusted."""

    UNKNOWN = "unknown"
    BOOTSTRAP = "bootstrap"
    LEARNING = "learning"
    LIVE = "live"
    STALE = "stale"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass(frozen=True)
class Forecast:
    """A point-in-time estimate for one scope.

    Parameters
    ----------
    state : ForecastState
        How much the estimate can be trusted.
    remaining_seconds : float or None
        Estimated seconds until the scope finishes; ``None`` when there is no estimate.
    rate : float or None
        Units of work per second used for the estimate.
    completed : float or None
        Latest completed amount.
    total : float or None
        Latest total, or ``None`` when unknown.
    unit : str or None
        Unit of ``completed`` and ``total``.
    """

    state: ForecastState
    remaining_seconds: float | None = None
    rate: float | None = None
    completed: float | None = None
    total: float | None = None
    unit: str | None = None


class ScopeForecast:
    """Track one scope's records and estimate its remaining time.

    Parameters
    ----------
    scope : str
        Scope id this forecast follows. Records for other scopes are ignored.
    """

    def __init__(self, scope: str) -> None:
        self.scope = scope
        self._unit: str | None = None
        self._completed: float | None = None
        self._total: float | None = None
        self._start_completed = 0.0
        self._elapsed = 0.0
        self._ended: Event | None = None

    def observe(self, record: Record, received_at: float) -> None:
        """Take one record into account.

        Parameters
        ----------
        record : Record
            A record for this scope.
        received_at : float
            When the record was received, in seconds on the caller's clock. Used
            to decide staleness; must be the same clock passed to :meth:`snapshot`.
        """
        if record.scope != self.scope:
            return
        if record.event is Event.START:
            self._start_completed = record.completed or 0.0
            self._unit = record.unit or self._unit
            self._ended = None
        if record.event in (Event.END, Event.FAIL):
            self._ended = record.event
        if record.completed is not None:
            self._completed = record.completed
        if record.event in (Event.START, Event.UPDATE, Event.END):
            self._total = record.total
        self._elapsed = record.elapsed

    def snapshot(self, now: float) -> Forecast:
        """Return the estimate as of ``now``.

        Parameters
        ----------
        now : float
            Current time on the clock used for ``received_at``.

        Returns
        -------
        Forecast
            The current estimate.
        """
        common = {"completed": self._completed, "total": self._total, "unit": self._unit}
        if self._ended is Event.END:
            return Forecast(ForecastState.COMPLETE, remaining_seconds=0.0, **common)
        if self._ended is Event.FAIL:
            return Forecast(ForecastState.FAILED, **common)
        if self._completed is None or self._total is None:
            return Forecast(ForecastState.UNKNOWN, **common)
        done = self._completed - self._start_completed
        if done <= 0 or self._elapsed <= 0:
            return Forecast(ForecastState.LEARNING, **common)
        rate = done / self._elapsed
        remaining = max(self._total - self._completed, 0.0) / rate
        return Forecast(ForecastState.LIVE, remaining_seconds=remaining, rate=rate, **common)
