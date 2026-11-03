# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Estimate remaining time from progress records.

The same code serves the terminal's "time remaining" and a scheduler reading a
job's logs, so the two never disagree. The rules are specified in
``docs/design.md``.

:class:`ScopeForecast` follows one scope; :class:`JobForecast` combines every
scope of one process run into a single :class:`JobEstimate`.
"""

from __future__ import annotations

import math
import statistics
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
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
class ForecastConfig:
    """Thresholds and smoothing parameters of the estimator.

    The first five are the *initial* thresholds of ``docs/design.md``; the rest
    shape the robust rate.

    Parameters
    ----------
    min_observed_seconds : float
        Learning becomes Live only after this many seconds of ``elapsed`` in the segment.
    min_intervals : int
        ... and after this many measured intervals.
    min_fraction : float
        ... and after this fraction of the total was done within the segment.
    heartbeat_seconds : float
        The logger's heartbeat; a scope is never Stale before ``2 x`` this.
    stale_factor : float
        A Live scope is Stale after ``stale_factor x`` its median record gap (and at least
        two heartbeats) without a record.
    ewma_alpha : float
        Weight of a new interval rate in the exponentially weighted average. The first
        intervals use ``1 / n`` instead, so the start is a plain mean.
    outlier_factor : float
        An interval rate more than this many times above, or below, the recent median is
        an outlier and does not enter the average.
    reference_window : int
        How many accepted interval rates make up the "recent median".
    min_reference : int
        Outlier tests start once this many rates are accepted; until then every rate is.
    shift_intervals : int
        This many consecutive outliers on the same side are a sustained change, not a
        blip: the average is re-seeded from them.
    """

    min_observed_seconds: float = 60.0
    min_intervals: int = 3
    min_fraction: float = 0.01
    heartbeat_seconds: float = 120.0
    stale_factor: float = 3.0
    ewma_alpha: float = 0.3
    outlier_factor: float = 3.0
    reference_window: int = 9
    min_reference: int = 3
    shift_intervals: int = 3


DEFAULT_CONFIG = ForecastConfig()


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


class _RateEstimator:
    """Exponentially weighted interval rate that discounts outliers.

    An interval rate is an outlier when it is more than ``outlier_factor`` times above or
    below the median of the recently accepted rates. Outliers are held back, so one
    evaluation pause or checkpoint (a single very slow interval) leaves the average
    alone. When ``shift_intervals`` outliers in a row fall on the same side, the speed
    really changed: the average is re-seeded from those rates.
    """

    def __init__(self, config: ForecastConfig) -> None:
        self._config = config
        self._reference: deque[float] = deque(maxlen=config.reference_window)
        self._pending: list[float] = []
        self._pending_side = 0
        self._count = 0
        self.value: float | None = None
        self.intervals = 0

    def add(self, rate: float) -> None:
        """Take one interval rate into account."""
        self.intervals += 1
        side = self._side(rate)
        if side == 0:
            self._blend(rate)
            self._pending = []
            self._pending_side = 0
            return
        if side != self._pending_side:
            self._pending = []
            self._pending_side = side
        self._pending.append(rate)
        if len(self._pending) >= self._config.shift_intervals:
            self._shift()

    def _side(self, rate: float) -> int:
        if len(self._reference) < self._config.min_reference:
            return 0
        median = statistics.median(self._reference)
        if median <= 0:
            return 0
        if rate < median / self._config.outlier_factor:
            return -1
        if rate > median * self._config.outlier_factor:
            return 1
        return 0

    def _blend(self, rate: float) -> None:
        self._reference.append(rate)
        self._count += 1
        if self.value is None:
            self.value = rate
            return
        alpha = max(self._config.ewma_alpha, 1.0 / self._count)
        self.value += alpha * (rate - self.value)

    def _shift(self) -> None:
        pending = self._pending
        self._pending = []
        self._pending_side = 0
        self._reference.clear()
        self._count = 0
        self.value = None
        for rate in pending:
            self._blend(rate)


class ScopeForecast:
    """Track one scope's records and estimate its remaining time.

    Parameters
    ----------
    scope : str
        Scope id this forecast follows. Records for other scopes are ignored.
    config : ForecastConfig, optional
        Thresholds and smoothing parameters.
    """

    def __init__(self, scope: str, *, config: ForecastConfig = DEFAULT_CONFIG) -> None:
        self.scope = scope
        self._config = config
        self._unit: str | None = None
        self._completed: float | None = None
        self._total: float | None = None
        self._parent: str | None = None
        self._root = False
        self._elapsed = 0.0
        self._ended: Event | None = None
        self._reached_live = False
        self._run: str | None = None
        self._begin_segment()

    @property
    def parent(self) -> str | None:
        """Parent scope id, once any record has named one."""
        return self._parent

    @property
    def root(self) -> bool:
        """Whether any record marked this scope as standing for the whole job."""
        return self._root

    @property
    def elapsed(self) -> float:
        """Seconds since the current segment started, per the latest record."""
        return self._elapsed

    @property
    def reached_live(self) -> bool:
        """Whether this scope has ever been Live, in any segment."""
        return self._reached_live

    def observe(self, record: Record, received_at: float) -> None:
        """Take one record into account.

        A record whose ``seq`` is not above the highest seen in the current segment is
        ignored (a duplicate from the other stream). A ``start`` after the scope has
        reported beyond ``q=1`` begins a new segment.

        Parameters
        ----------
        record : Record
            A record for this scope.
        received_at : float
            When the record was received, in seconds on the caller's clock. Used
            to decide staleness; must be the same clock passed to :meth:`snapshot`.
        """
        if record.scope != self.scope or record.event is Event.PLAN:
            return
        if record.event is Event.START and self._is_restart(record):
            self._begin_segment()
        elif record.seq <= self._max_seq:
            return
        if self._ended is not None and record.event is not Event.START:
            return
        self._max_seq = record.seq
        self._receipts.append(received_at)
        self._take(record)

    def snapshot(self, now: float) -> Forecast:
        """Return the estimate as of ``now``.

        While Live, the remaining time counts down from the latest record by the time
        received since it. A Stale scope keeps the estimate from its latest record.

        Parameters
        ----------
        now : float
            Current time on the clock used for ``received_at``.

        Returns
        -------
        Forecast
            The current estimate.
        """
        rate = self._estimator.value
        common = {"completed": self._completed, "total": self._total, "unit": self._unit}
        if self._ended is Event.END:
            return Forecast(ForecastState.COMPLETE, remaining_seconds=0.0, rate=rate, **common)
        if self._ended is Event.FAIL:
            return Forecast(ForecastState.FAILED, rate=rate, **common)
        if self._completed is None or self._total is None:
            return Forecast(ForecastState.UNKNOWN, **common)
        if not self._live or rate is None:
            return Forecast(ForecastState.LEARNING, **common)
        remaining = self._remaining(rate)
        if self._is_stale(now):
            return Forecast(ForecastState.STALE, remaining_seconds=remaining, rate=rate, **common)
        if remaining is not None:
            remaining = max(remaining - max(now - self._receipts[-1], 0.0), 0.0)
        return Forecast(ForecastState.LIVE, remaining_seconds=remaining, rate=rate, **common)

    def _begin_segment(self) -> None:
        self._estimator = _RateEstimator(self._config)
        self._max_seq = 0
        self._receipts: deque[float] = deque(maxlen=self._config.reference_window + 1)
        self._baseline_elapsed = 0.0
        self._baseline_completed = 0.0
        self._previous: tuple[float, float] | None = None
        self._live = False
        self._ended = None
        self._elapsed = 0.0
        # A restarted process may place the scope differently; its records say where.
        self._parent = None
        self._root = False

    def _is_restart(self, record: Record) -> bool:
        # With run ids, a start from another run is a restart and one from the same run is
        # the same line read twice. Without them, a start after q=1 is taken as a restart.
        if record.run is not None and self._run is not None:
            return record.run != self._run
        return self._max_seq > 1

    def _take(self, record: Record) -> None:
        if record.event is Event.START and record.run is not None:
            self._run = record.run
        self._parent = record.parent or self._parent
        self._root = self._root or record.root
        self._unit = record.unit or self._unit
        self._elapsed = record.elapsed
        if record.event is Event.START:
            self._ended = None
        if record.event in (Event.START, Event.UPDATE, Event.END):
            self._total = record.total
        if record.completed is not None:
            self._completed = record.completed
            self._measure(record)
        if record.event in (Event.END, Event.FAIL):
            self._ended = record.event
        elif self._ready():
            self._live = True
            self._reached_live = True

    def _measure(self, record: Record) -> None:
        completed = record.completed or 0.0
        if self._previous is None:
            self._previous = (record.elapsed, completed)
            self._baseline_elapsed = record.elapsed
            self._baseline_completed = completed
            return
        if record.event is Event.END:
            return
        seconds = record.elapsed - self._previous[0]
        done = completed - self._previous[1]
        if seconds <= 0:
            return
        if done < 0:
            self._previous = (record.elapsed, completed)
            return
        # No progress yet: keep the old baseline so the next interval spans the stall
        # and a slow step is measured at its true speed rather than skipped.
        if done == 0:
            return
        rate = done / seconds
        if math.isfinite(rate):
            self._estimator.add(rate)
            self._previous = (record.elapsed, completed)

    def _ready(self) -> bool:
        cfg = self._config
        if self._estimator.value is None or not self._total or self._total <= 0 or self._completed is None:
            return False
        fraction = (self._completed - self._baseline_completed) / self._total
        return (
            self._elapsed - self._baseline_elapsed >= cfg.min_observed_seconds
            and self._estimator.intervals >= cfg.min_intervals
            and fraction >= cfg.min_fraction
        )

    def _is_stale(self, now: float) -> bool:
        receipts = list(self._receipts)
        gaps = [later - earlier for earlier, later in zip(receipts, receipts[1:])]
        median_gap = statistics.median(gaps) if gaps else 0.0
        limit = max(self._config.stale_factor * median_gap, 2 * self._config.heartbeat_seconds)
        return now - receipts[-1] > limit

    def _remaining(self, rate: float) -> float | None:
        if rate <= 0 or self._total is None or self._completed is None:
            return None
        remaining = max(self._total - self._completed, 0.0) / rate
        return remaining if math.isfinite(remaining) else None


_ENDED_STATES = (ForecastState.COMPLETE, ForecastState.FAILED)


@dataclass(frozen=True)
class JobEstimate:
    """A point-in-time estimate for one job (one process run).

    Parameters
    ----------
    state : ForecastState
        How much the estimate can be trusted.
    remaining_seconds : float or None
        Estimated seconds until the job finishes; ``None`` when there is no estimate.
    source : str or None
        Where the estimate comes from: ``"root"``, ``"plan"``, ``"concurrent"`` or
        ``"user"`` (the expected runtime the user supplied). ``None`` without an estimate.
    partial : bool
        Whether the estimate covers only the phases known so far (no root scope and no
        complete plan).
    scopes : dict of str to Forecast
        Every scope's own forecast. A fresh dict per snapshot; because of it a
        :class:`JobEstimate` is not hashable.
    """

    state: ForecastState
    remaining_seconds: float | None = None
    source: str | None = None
    partial: bool = False
    scopes: dict[str, Forecast] = field(default_factory=dict)


@dataclass
class _PlanTally:
    """Running totals while walking a plan."""

    remaining: float = 0.0
    known_seconds: float = 0.0
    known_weight: float = 0.0
    unstarted_weight: float = 0.0
    stale: bool = False


class JobForecast:
    """Combine every scope of one job into one estimate.

    Precedence: the ``root`` scope when there is one; otherwise the latest ``plan``
    (finished phases count 0, the active phase its own estimate, phases not yet started
    ``weight x seconds-per-weight`` where seconds-per-weight is the elapsed plus
    remaining time of the finished and active phases over their weights); otherwise the
    longest remaining time among top-level scopes. Nested scopes (those with a parent)
    are never added on top of their parent.

    Parameters
    ----------
    expected_seconds : float, optional
        The user's expected runtime. Used only until some scope is Live, then never again.
    config : ForecastConfig, optional
        Thresholds and smoothing parameters, shared by every scope.
    """

    def __init__(self, *, expected_seconds: float | None = None, config: ForecastConfig = DEFAULT_CONFIG) -> None:
        self._expected = expected_seconds
        self._config = config
        self._scopes: dict[str, ScopeForecast] = {}
        self._plan: tuple[tuple[str, float], ...] = ()
        self._plan_seq: dict[str, int] = {}
        self._first_received: float | None = None
        self._ever_live = False

    def observe(self, record: Record, received_at: float) -> None:
        """Route one record to its scope's forecast, creating it on first sight.

        Parameters
        ----------
        record : Record
            Any record of the job.
        received_at : float
            When the record was received, on the clock passed to :meth:`snapshot`.
        """
        if self._first_received is None:
            self._first_received = received_at
        if record.event is Event.PLAN:
            # A plan read late from the other stream must not replace a newer one.
            if record.phases and record.seq > self._plan_seq.get(record.scope, 0):
                self._plan_seq[record.scope] = record.seq
                self._plan = record.phases
            return
        scope = self._scopes.get(record.scope)
        if scope is None:
            scope = self._scopes[record.scope] = ScopeForecast(record.scope, config=self._config)
        scope.observe(record, received_at)
        self._ever_live = self._ever_live or scope.reached_live

    def snapshot(self, now: float) -> JobEstimate:
        """Return the job's estimate as of ``now``.

        Parameters
        ----------
        now : float
            Current time on the clock used for ``received_at``.

        Returns
        -------
        JobEstimate
            The current estimate.
        """
        forecasts = {name: scope.snapshot(now) for name, scope in self._scopes.items()}
        ended = self._ended_state(forecasts)
        if ended is None:
            forecasts = self._with_child_estimates(forecasts)
        if ended is not None:
            remaining = 0.0 if ended is ForecastState.COMPLETE else None
            return JobEstimate(ended, remaining_seconds=remaining, scopes=forecasts)
        estimate = self._pick(forecasts) or self._without_estimate(now, forecasts)
        if estimate.source in ("root", "plan") and self._rests_on_partial():
            estimate = replace(estimate, partial=True)
        return replace(estimate, scopes=forecasts)

    def _with_child_estimates(self, forecasts: Mapping[str, Forecast]) -> dict[str, Forecast]:
        """Give a running scope with no estimate of its own one built from its children.

        A ``phase()`` counts as one unit of work, and a parent such as ``Epochs`` (one unit
        per child) reports too rarely to have a rate of its own early on; the scopes inside
        them do. A counted parent's estimate is the running child's remaining time plus,
        for each unit still to come, the duration of one child. A parent with no count of
        its own (a phase, or no total) cannot know how many children are still to come, so
        its estimate is only the running child's and is marked partial.
        """
        effective = dict(forecasts)
        self._partial_scopes: set[str] = set()
        # One pass per level of nesting: repeat until no scope gains an estimate.
        for _ in range(len(effective)):
            if not self._borrow_once(effective):
                break
        return effective

    def _borrow_once(self, effective: dict[str, Forecast]) -> bool:
        changed = False
        for name, forecast in list(effective.items()):
            if forecast.state in _ENDED_STATES or self._has_estimate(forecast):
                continue
            children = [child for child, scope in self._scopes.items() if scope.parent == name]
            borrowed = self._from_children(forecast, children, effective)
            if borrowed is not None:
                effective[name] = borrowed[0]
                if borrowed[1]:
                    self._partial_scopes.add(name)
                changed = True
        return changed

    def _from_children(
        self, parent: Forecast, children: list[str], effective: Mapping[str, Forecast]
    ) -> tuple[Forecast, bool] | None:
        """Return the parent's estimate from its children and whether it is partial, if any."""
        running = [child for child in children if self._has_estimate(effective[child])]
        finished = [child for child in children if effective[child].state is ForecastState.COMPLETE]
        counted = parent.total is not None and parent.total > 1 and parent.completed is not None
        if running:
            longest = max(running, key=lambda child: effective[child].remaining_seconds or 0.0)
            now = effective[longest]
            remaining = now.remaining_seconds or 0.0
            per_unit = self._scopes[longest].elapsed + remaining
            stale = any(effective[child].state is ForecastState.STALE for child in running)
        elif counted and finished:
            # Between two children: the next has not reported yet, so use the finished ones.
            remaining = 0.0
            per_unit = sum(self._scopes[child].elapsed for child in finished) / len(finished)
            stale = False
        else:
            return None
        partial = not counted
        if counted:
            units_after = max((parent.total or 0.0) - (parent.completed or 0.0) - (1 if running else 0), 0.0)
            remaining += units_after * per_unit
        state = ForecastState.STALE if stale else ForecastState.LIVE
        return replace(parent, state=state, remaining_seconds=remaining), partial

    def _rests_on_partial(self) -> bool:
        partial = getattr(self, "_partial_scopes", set())
        root = self._root()
        if root is not None:
            return root in partial
        return any(name in partial for name, _ in self._plan)

    def _top_level(self) -> list[str]:
        return [name for name, scope in self._scopes.items() if scope.parent is None or scope.root]

    def _root(self) -> str | None:
        return next((name for name, scope in self._scopes.items() if scope.root), None)

    def _ended_state(self, forecasts: Mapping[str, Forecast]) -> ForecastState | None:
        root = self._root()
        if root is not None:
            state = forecasts[root].state
            return state if state in (ForecastState.COMPLETE, ForecastState.FAILED) else None
        top = self._top_level()
        if any(forecasts[name].state is ForecastState.FAILED for name in top):
            return ForecastState.FAILED
        # Without a root, only a finished plan proves the job is done: between two
        # unplanned phases every scope seen so far has ended, but more work may follow.
        if not self._plan:
            return None
        phases_done = all(
            name in forecasts and forecasts[name].state is ForecastState.COMPLETE for name, _ in self._plan
        )
        if phases_done and all(forecasts[name].state is ForecastState.COMPLETE for name in top):
            return ForecastState.COMPLETE
        return None

    def _pick(self, forecasts: Mapping[str, Forecast]) -> JobEstimate | None:
        root = self._root()
        if root is not None:
            return self._from_root(forecasts[root])
        return self._from_plan(forecasts) or self._from_concurrent(forecasts)

    @staticmethod
    def _has_estimate(forecast: Forecast) -> bool:
        return forecast.state in (ForecastState.LIVE, ForecastState.STALE) and forecast.remaining_seconds is not None

    def _from_root(self, forecast: Forecast) -> JobEstimate | None:
        if not self._has_estimate(forecast):
            return None
        return JobEstimate(forecast.state, forecast.remaining_seconds, source="root")

    def _from_plan(self, forecasts: Mapping[str, Forecast]) -> JobEstimate | None:
        if not self._plan:
            return None
        tally = _PlanTally()
        for name, weight in self._plan:
            forecast = forecasts.get(name)
            if forecast is None:
                tally.unstarted_weight += weight
            elif not self._tally_phase(tally, weight, forecast, self._scopes[name].elapsed):
                return None
        if tally.unstarted_weight > 0:
            if tally.known_weight <= 0 or tally.known_seconds <= 0:
                return None
            tally.remaining += tally.unstarted_weight * tally.known_seconds / tally.known_weight
        state = ForecastState.STALE if tally.stale else ForecastState.LIVE
        return JobEstimate(state, tally.remaining, source="plan")

    def _tally_phase(self, tally: _PlanTally, weight: float, forecast: Forecast, elapsed: float) -> bool:
        if forecast.state is ForecastState.COMPLETE:
            tally.known_seconds += elapsed
        elif self._has_estimate(forecast) and forecast.remaining_seconds is not None:
            tally.remaining += forecast.remaining_seconds
            tally.known_seconds += elapsed + forecast.remaining_seconds
            tally.stale = tally.stale or forecast.state is ForecastState.STALE
        else:
            return False
        tally.known_weight += weight
        return True

    def _from_concurrent(self, forecasts: Mapping[str, Forecast]) -> JobEstimate | None:
        running = [forecasts[name] for name in self._top_level() if self._has_estimate(forecasts[name])]
        if not running:
            return None
        longest = max(running, key=lambda forecast: forecast.remaining_seconds or 0.0)
        stale = any(forecast.state is ForecastState.STALE for forecast in running)
        state = ForecastState.STALE if stale else ForecastState.LIVE
        return JobEstimate(state, longest.remaining_seconds, source="concurrent", partial=True)

    def _without_estimate(self, now: float, forecasts: Mapping[str, Forecast]) -> JobEstimate:
        if self._expected is not None and not self._ever_live:
            waited = 0.0 if self._first_received is None else max(now - self._first_received, 0.0)
            return JobEstimate(ForecastState.BOOTSTRAP, max(self._expected - waited, 0.0), source="user")
        if any(forecast.state is ForecastState.LEARNING for forecast in forecasts.values()):
            return JobEstimate(ForecastState.LEARNING)
        return JobEstimate(ForecastState.UNKNOWN)
