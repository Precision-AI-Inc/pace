# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Estimate remaining time from progress records.

The same code serves the terminal's "time remaining" and a scheduler reading a
job's logs, so the two never disagree. The rules are specified in
``docs/design.md``.

:class:`ScopeForecast` follows one scope; :class:`JobForecast` combines every
scope of one process run into a single :class:`JobEstimate`.
"""

import math
import statistics
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from itertools import pairwise

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


@dataclass(frozen=True, slots=True)
class ForecastConfig:
    """Thresholds and smoothing parameters of the estimator.

    The first five are the default thresholds of ``docs/design.md``; the rest
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


@dataclass(frozen=True, slots=True)
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


# Enough for any real run of restarts; bounded because run ids come from untrusted lines.
_RUNS_REMEMBERED = 16


def _order(record: Record) -> float:
    """Return where ``record`` falls among its run's records: by ``seq``, or first for a start and last for an end without one."""
    if record.seq is not None:
        return record.seq
    return 0 if record.event is Event.START else math.inf


class _Runs:
    """The run id records come from now, and the ones left behind by a restart."""

    def __init__(self) -> None:
        self.current: str | None = None
        self._left: deque[str] = deque(maxlen=_RUNS_REMEMBERED)

    def left_behind(self, run: str | None) -> bool:
        """Return whether ``run`` is one a restart has replaced, so its records are late copies."""
        return run is not None and run != self.current and run in self._left

    def leave(self) -> None:
        """Leave the current run behind; the next record's run becomes current."""
        if self.current is not None:
            self._left.append(self.current)
        self.current = None


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
        self._runs = _Runs()
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

    @property
    def last_received(self) -> float | None:
        """When the latest record of the current segment was received, if any."""
        return self._receipts[-1] if self._receipts else None

    def elapsed_at(self, now: float) -> float:
        """Return the seconds since the current segment started, as of ``now``.

        A running scope's elapsed time counts on from its latest record by the time
        received since it, so a ``phase()``, which writes only when it starts and ends,
        has the right elapsed time while the work inside it runs. An ended scope keeps
        the elapsed time of its last record, and so does a scope that is Stale on its own
        records, which also keeps the estimate of its last record.

        Parameters
        ----------
        now : float
            Current time on the clock used for ``received_at``.

        Returns
        -------
        float
            Seconds since the segment started.
        """
        if self._ended is not None or not self._receipts or (self._live and self._is_stale(now)):
            return self._elapsed
        return self._elapsed + max(now - self._receipts[-1], 0.0)

    def seconds_per_unit(self) -> float | None:
        """Return the mean seconds per unit over the current segment's records.

        Measured from the segment's first record with a count (or the latest record where
        the count went backwards) to its latest, so a resumed scope counts only the units
        done in this run.

        Returns
        -------
        float or None
            Seconds per unit, or ``None`` before the segment has made any progress.
        """
        if self._completed is None:
            return None
        units = self._completed - self._pace_origin[1]
        seconds = self._elapsed - self._pace_origin[0]
        if units <= 0 or seconds <= 0:
            return None
        return seconds / units

    def observe(self, record: Record, received_at: float) -> bool:
        """Take one record into account.

        A record from a run id not seen before begins a new segment (the process
        restarted); one from a run this scope has left behind is ignored. Within a run, a
        record that does not come after the latest one taken is ignored: a copy read from
        the other stream, or a ``start`` read late. Records are ordered by ``seq``; a
        ``pace:2`` start, which has none, comes first and an end, fail or step last. A
        ``pace:1`` ``start`` without a run id after the scope reported beyond ``q=1`` also
        begins a new segment.

        Parameters
        ----------
        record : Record
            A record for this scope.
        received_at : float
            When the record was received, in seconds on the caller's clock. Used
            to decide staleness; must be the same clock passed to :meth:`snapshot`.

        Returns
        -------
        bool
            Whether the record was taken, rather than ignored.
        """
        if record.scope != self.scope or record.event is Event.PLAN or self._runs.left_behind(record.run):
            return False
        if self._is_restart(record):
            self._runs.leave()
            self._begin_segment()
        key = _order(record)
        if self._last_key is not None and key <= self._last_key:
            return False
        if self._ended is not None and record.event is not Event.START:
            return False
        self._last_key = key
        self._receipts.append(received_at)
        self._take(record)
        return True

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
        self._last_key: float | None = None
        self._receipts: deque[float] = deque(maxlen=self._config.reference_window + 1)
        self._baseline_elapsed = 0.0
        self._baseline_completed = 0.0
        self._pace_origin = (0.0, 0.0)
        self._previous: tuple[float, float] | None = None
        self._live = False
        self._ended = None
        self._elapsed = 0.0
        # A restarted process may place the scope differently; its records say where.
        self._parent = None
        self._root = False

    def _is_restart(self, record: Record) -> bool:
        if self._last_key is None:
            return False
        if record.run is not None:
            return record.run != self._runs.current
        # A pace:1 start without a run id: after q=1 it can only be a restart.
        return record.event is Event.START and self._last_key > 1

    def _take(self, record: Record) -> None:
        self._runs.current = record.run or self._runs.current
        self._parent = record.parent or self._parent
        self._root = self._root or record.root
        self._unit = record.unit or self._unit
        self._elapsed = record.elapsed
        if record.event in (Event.START, Event.STEP):
            self._ended = None
        if record.event in (Event.START, Event.UPDATE, Event.END, Event.STEP):
            self._total = record.total
        if record.completed is not None:
            self._completed = record.completed
            self._measure(record)
        if record.event in (Event.END, Event.FAIL, Event.STEP):
            self._ended = Event.FAIL if record.event is Event.FAIL else Event.END
        elif self._ready():
            self._live = True
            self._reached_live = True

    def _measure(self, record: Record) -> None:
        completed = record.completed or 0.0
        if self._previous is None:
            self._previous = (record.elapsed, completed)
            self._baseline_elapsed = record.elapsed
            self._baseline_completed = completed
            self._pace_origin = (record.elapsed, completed)
            return
        if record.event is Event.END:
            return
        seconds = record.elapsed - self._previous[0]
        done = completed - self._previous[1]
        if seconds <= 0:
            return
        if done < 0:
            self._previous = (record.elapsed, completed)
            self._pace_origin = (record.elapsed, completed)
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

    def stale_limit(self) -> float:
        """Return how long the scope may go without a record before it is Stale, in seconds.

        Returns
        -------
        float
            ``stale_factor`` times the median gap between its recent records, and at least
            twice the heartbeat.
        """
        receipts = list(self._receipts)
        gaps = [later - earlier for earlier, later in pairwise(receipts)]
        median_gap = statistics.median(gaps) if gaps else 0.0
        return max(self._config.stale_factor * median_gap, 2 * self._config.heartbeat_seconds)

    def _is_stale(self, now: float) -> bool:
        return now - self._receipts[-1] > self.stale_limit()

    def _remaining(self, rate: float) -> float | None:
        if rate <= 0 or self._total is None or self._completed is None:
            return None
        remaining = max(self._total - self._completed, 0.0) / rate
        return remaining if math.isfinite(remaining) else None


_ENDED_STATES = (ForecastState.COMPLETE, ForecastState.FAILED)

# The writer reports a counted scope at least every 1% of its total, so a parent of up
# to 100 units writes every advance; a larger one may skip some.
_EVERY_ADVANCE_WRITTEN = 100


@dataclass(frozen=True, slots=True)
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
        complete plan), or rests on a running ``phase()`` timed only by the loop running
        inside it, after which more loops may follow.
    scopes : dict of str to Forecast
        Every scope's own forecast. A fresh dict per snapshot; because of it a
        :class:`JobEstimate` is not hashable.
    """

    state: ForecastState
    remaining_seconds: float | None = None
    source: str | None = None
    partial: bool = False
    scopes: dict[str, Forecast] = field(default_factory=dict)


@dataclass(slots=True)
class _Borrowed:
    """What one snapshot's borrowing from children decided, passed down so ``JobForecast`` keeps no per-snapshot state.

    Attributes
    ----------
    forecasts : dict of str to Forecast
        Every scope's forecast, a parent without an estimate of its own given one built from its children.
    partial : set of str
        Scopes whose borrowed estimate covers only the child running now, because they cannot know how
        many children are still to come.
    frozen_at : dict of str to float
        For a borrowed estimate that went Stale: the time its source last reported. Its clock stops
        there, as a Stale scope's own estimate does.
    """

    forecasts: dict[str, Forecast]
    partial: set[str] = field(default_factory=set)
    frozen_at: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
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
        self._plan_runs: dict[str, _Runs] = {}
        self._first_received: float | None = None
        self._ever_live = False

    def observe(self, record: Record, received_at: float) -> bool:
        """Route one record to its scope's forecast, creating it on first sight.

        Parameters
        ----------
        record : Record
            Any record of the job.
        received_at : float
            When the record was received, on the clock passed to :meth:`snapshot`.

        Returns
        -------
        bool
            Whether the record was taken; ``False`` for a copy read from the other stream,
            a record read late, or a plan that replaces nothing.
        """
        if self._first_received is None:
            self._first_received = received_at
        if record.event is Event.PLAN:
            return self._take_plan(record)
        scope = self._scopes.get(record.scope)
        if scope is None:
            scope = self._scopes[record.scope] = ScopeForecast(record.scope, config=self._config)
        taken = scope.observe(record, received_at)
        self._ever_live = self._ever_live or scope.reached_live
        return taken

    def _take_plan(self, record: Record) -> bool:
        # A plan read late from the other stream must not replace a newer one of its run,
        # and a plan from a restarted run replaces any plan of the run before it.
        runs = self._plan_runs.setdefault(record.scope, _Runs())
        if not record.phases or runs.left_behind(record.run):
            return False
        if record.run != runs.current and record.scope in self._plan_seq:
            runs.leave()
            del self._plan_seq[record.scope]
        if (record.seq or 0) <= self._plan_seq.get(record.scope, 0):
            return False
        runs.current = record.run
        self._plan_seq[record.scope] = record.seq or 0
        self._plan = record.phases
        return True

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
        if ended is not None:
            remaining = 0.0 if ended is ForecastState.COMPLETE else None
            return JobEstimate(ended, remaining_seconds=remaining, scopes=forecasts)
        borrowed = self._with_child_estimates(forecasts, now)
        estimate = self._pick(borrowed, now) or self._without_estimate(now, borrowed.forecasts)
        if estimate.source in ("root", "plan") and self._rests_on_partial(borrowed):
            estimate = replace(estimate, partial=True)
        return replace(estimate, scopes=borrowed.forecasts)

    def _with_child_estimates(self, forecasts: Mapping[str, Forecast], now: float) -> _Borrowed:
        """Give a running scope with no estimate of its own one built from its children.

        A ``phase()`` counts as one unit of work, and a parent such as ``Epochs`` reports
        too rarely to have a rate of its own early on; the scopes inside them do. A counted
        parent's estimate is the time one unit takes times the units still to come, less
        the time already spent in the running unit, and never less than the running
        child's remaining time. Once a unit has finished, the parent's own records time a
        unit, however many children it holds; before that, the running unit counts as the
        time it has taken so far plus the running child's remaining time. A parent with no
        count of its own (a phase, or no total) cannot know how many children are still to
        come, so its estimate is only the running child's and is marked partial.
        """
        borrowed = _Borrowed(dict(forecasts))
        # One pass per level of nesting: repeat until no scope gains an estimate.
        for _ in range(len(borrowed.forecasts)):
            if not self._borrow_once(borrowed, now):
                break
        return borrowed

    def _borrow_once(self, borrowed: _Borrowed, now: float) -> bool:
        changed = False
        for name, forecast in list(borrowed.forecasts.items()):
            if forecast.state in _ENDED_STATES or self._has_estimate(forecast):
                continue
            children = [child for child, scope in self._scopes.items() if scope.parent == name]
            from_children = self._from_children(name, children, borrowed, now)
            if from_children is not None:
                borrowed.forecasts[name] = from_children[0]
                if from_children[1]:
                    borrowed.partial.add(name)
                changed = True
        return changed

    def _from_children(
        self, name: str, children: list[str], borrowed: _Borrowed, now: float
    ) -> tuple[Forecast, bool] | None:
        """Return the parent's estimate from its children and whether it is partial, if any."""
        effective = borrowed.forecasts
        parent = effective[name]
        running = [child for child in children if self._has_estimate(effective[child])]
        finished = [child for child in children if effective[child].state is ForecastState.COMPLETE]
        counted = parent.total is not None and parent.total > 1 and parent.completed is not None
        if not running and not (counted and finished):
            return None
        child_left = max((effective[child].remaining_seconds or 0.0 for child in running), default=0.0)
        stale_children = [child for child in running if effective[child].state is ForecastState.STALE]
        state = ForecastState.STALE if stale_children else ForecastState.LIVE
        if stale_children:
            now = min(self._reported_at(child, borrowed, now) for child in stale_children)
            borrowed.frozen_at[name] = now
        elif not running:
            # Between two children, the parent and every scope under it (in any state)
            # speak for the job; when none has reported within the parent's limit, it is Stale.
            reported = max(self._reported_at(scope, borrowed, now) for scope in (name, *self._descendants(name)))
            if now - reported > self._scopes[name].stale_limit():
                state = ForecastState.STALE
                now = borrowed.frozen_at[name] = reported
        if not counted:
            return replace(parent, state=state, remaining_seconds=child_left), True
        scope = self._scopes[name]
        per_unit = scope.seconds_per_unit()
        if per_unit is None and (parent.total or 0.0) <= _EVERY_ADVANCE_WRITTEN:
            # No unit finished yet: this one has taken at least its time so far plus
            # what the running child has left.
            in_unit = scope.elapsed_at(now)
            per_unit = in_unit + child_left
        elif per_unit is None:
            # A larger parent need not write every advance, and its units are often steps
            # rather than children (a model load inside a root over 5000 steps), so no
            # child says how long a unit takes. Wait for the parent's own pace.
            return None
        else:
            # Units left and the time since are both measured from the parent's latest record.
            in_unit = scope.elapsed_at(now) - scope.elapsed
        units_left = max((parent.total or 0.0) - (parent.completed or 0.0), 0.0)
        remaining = max(units_left * per_unit - in_unit, child_left)
        return replace(parent, state=state, remaining_seconds=remaining), False

    def _descendants(self, name: str) -> list[str]:
        found: list[str] = []
        frontier = [name]
        while frontier:
            current = frontier.pop()
            for child, scope in self._scopes.items():
                if scope.parent == current and child != name and child not in found:
                    found.append(child)
                    frontier.append(child)
        return found

    def _reported_at(self, name: str, borrowed: _Borrowed, now: float) -> float:
        """Return when a scope's estimate was last refreshed: its latest record, or its freeze."""
        frozen = borrowed.frozen_at.get(name)
        if frozen is not None:
            return frozen
        received = self._scopes[name].last_received
        return received if received is not None else now

    def _rests_on_partial(self, borrowed: _Borrowed) -> bool:
        root = self._root()
        if root is not None:
            return root in borrowed.partial
        return any(name in borrowed.partial for name, _ in self._plan)

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

    def _pick(self, borrowed: _Borrowed, now: float) -> JobEstimate | None:
        root = self._root()
        if root is not None:
            return self._from_root(borrowed.forecasts[root])
        return self._from_plan(borrowed, now) or self._from_concurrent(borrowed.forecasts)

    @staticmethod
    def _has_estimate(forecast: Forecast) -> bool:
        return forecast.state in (ForecastState.LIVE, ForecastState.STALE) and forecast.remaining_seconds is not None

    def _from_root(self, forecast: Forecast) -> JobEstimate | None:
        if not self._has_estimate(forecast):
            return None
        return JobEstimate(forecast.state, forecast.remaining_seconds, source="root")

    def _from_plan(self, borrowed: _Borrowed, now: float) -> JobEstimate | None:
        if not self._plan:
            return None
        tally = _PlanTally()
        for name, weight in self._plan:
            forecast = borrowed.forecasts.get(name)
            if forecast is None:
                tally.unstarted_weight += weight
            elif not self._tally_phase(tally, weight, forecast, self._elapsed_for_plan(name, borrowed, now)):
                return None
        if tally.unstarted_weight > 0:
            if tally.known_weight <= 0 or tally.known_seconds <= 0:
                return None
            tally.remaining += tally.unstarted_weight * tally.known_seconds / tally.known_weight
        state = ForecastState.STALE if tally.stale else ForecastState.LIVE
        return JobEstimate(state, tally.remaining, source="plan")

    def _elapsed_for_plan(self, name: str, borrowed: _Borrowed, now: float) -> float:
        frozen = borrowed.frozen_at.get(name)
        return self._scopes[name].elapsed_at(now if frozen is None else frozen)

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
