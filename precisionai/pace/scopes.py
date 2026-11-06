# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Progress scopes: ``progress``, ``track``, ``timed``, ``phase`` and ``plan``.

A scope is one piece of work with a start, a completed amount, an optional total and
an end. Each progress line it writes carries the ``[pace ...]`` trailer built by
:func:`precisionai.pace.protocol.format_trailer`; the text before it is for people.

Output rules (see ``docs/design.md``):

* On a terminal a single in-place bar line is redrawn about ten times a second, with no
  trailer. ``start`` draws it; ``end`` and ``fail`` print a normal line with a trailer.
* Otherwise ``start``, ``end``, ``fail`` and ``plan`` are written at once, except that a
  one-off step or a child scope holds back its ``start`` and, if it ends before its first
  line is due, writes one ``step`` line instead. ``upd`` is written only after ``PACE_MIN_INTERVAL`` seconds and either 1% more of the total or
  ``PACE_HEARTBEAT`` seconds (with an unknown total, the heartbeat alone).
* Only rank 0 writes progress unless ``PACE_ALL_RANKS=1``.

Scopes nest through a context variable: a scope opened inside ``progress``, ``phase``
or an iterated ``track`` records that scope as its parent. Threads and ``asyncio`` tasks
start with the context they were created in, so a scope opened in a new thread has no
parent unless the context is copied.
"""

import contextlib
import functools
import math
import re
from collections.abc import Callable, Iterable, Iterator, Sequence, Sized
from contextvars import ContextVar, Token
from typing import Any, Self, TypeVar

from precisionai.pace._render import (
    PROGRESS_LABEL,
    STEP_LABEL,
    colorize,
    format_timestamp,
    plan_text,
    progress_text,
)
from precisionai.pace.core import Logger, _Target
from precisionai.pace.core import logger as default_logger
from precisionai.pace.forecast import ScopeForecast
from precisionai.pace.protocol import (
    _MAX_LINE_BYTES,
    _MAX_NUMBER,
    _SCOPE_PATTERN,
    Event,
    Record,
    _is_bare_step,
    format_trailer,
)

__all__ = ["Phase", "Scope", "phase", "plan", "progress", "timed", "track"]

_REDRAW_INTERVAL = 0.1
_MAX_DERIVED_ID_CHARS = 48
_MAX_ID_CHARS = 64
_FALLBACK_ID = "scope"
_UPDATE_FRACTION = 0.01

_UNSAFE_ID_CHARS = re.compile(r"[^A-Za-z0-9._-]+")
_PARENT: ContextVar["Scope | None"] = ContextVar("pace_parent_scope", default=None)

_T = TypeVar("_T")


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _count(value: Any) -> float:
    number = _number(value)
    return 0.0 if number is None or number < 0 else min(number, _MAX_NUMBER)


def _total(value: Any) -> float | None:
    number = _number(value)
    # A total no trailer can carry is reported as unknown rather than dropped.
    return None if number is None or number <= 0 or number > _MAX_NUMBER else number


def _derive_scope_id(description: str) -> str:
    """Return a scope id from a description: lower case, anything outside the id alphabet as ``-``, at most 48 characters."""
    cleaned = _UNSAFE_ID_CHARS.sub("-", description.lower())[:_MAX_DERIVED_ID_CHARS].strip("-")
    return cleaned or _FALLBACK_ID


def _validate_scope_id(scope: str) -> str:
    """Return ``scope`` if it is a valid scope id, otherwise a sanitised version of it."""
    if _SCOPE_PATTERN.fullmatch(scope):
        return scope
    cleaned = _UNSAFE_ID_CHARS.sub("-", scope)[:_MAX_ID_CHARS].strip("-")
    return cleaned or _FALLBACK_ID


class Scope:
    """One unit of work being reported, and the handle the caller advances.

    Use it as a context manager (``progress`` does), or call :meth:`start` and
    :meth:`end` or :meth:`fail` yourself. Updates before :meth:`start` start the scope.
    Methods never raise because of an output problem and are safe to call from several
    threads.

    Parameters
    ----------
    description : str
        Short human label, also the source of the scope id when ``scope`` is not given.
    total : float, optional
        Total amount; ``None`` (or a non-positive number) means unknown.
    unit : str
        Unit of the counts, such as ``step`` or ``batch``.
    scope : str, optional
        Explicit scope id; an invalid one is sanitised. Made unique within the process
        with a numeric suffix.
    root : bool
        Mark this scope as standing for the whole job.
    completed : float
        Amount already done when the scope starts.
    resumed : bool
        Whether the scope resumes earlier work, so ``completed`` starts above zero.
    logger : Logger, optional
        Output to use; the module-level logger by default.
    """

    # A one-off step (a phase or a timed call) that ends before its first line is due writes a
    # bare step line of one unit; a child scope does the same with its counts and unit.
    _ONE_OFF = False

    def __init__(
        self,
        description: str = "",
        total: float | None = None,
        unit: str = "it",
        *,
        scope: str | None = None,
        root: bool = False,
        completed: float = 0.0,
        resumed: bool = False,
        logger: Logger | None = None,
    ) -> None:
        # tqdm takes (iterable, desc, total) and track takes (iterable, description, unit),
        # so a migrated call can pass the total as the unit. Every record of such a scope
        # would then be dropped silently; fail loudly at the call instead.
        if not isinstance(unit, str):
            raise TypeError(f"unit must be a string such as 'step', got {unit!r}; pass the total as total=...")
        self._out = logger or default_logger
        base = _derive_scope_id(description) if scope is None else _validate_scope_id(scope)
        self.id = self._out._claim_scope(base)
        self.description = description
        self.unit = unit
        self.root = bool(root)
        self.resumed = bool(resumed)
        self.completed = _count(completed)
        self.total = _total(total)
        self.parent: str | None = None
        self.depth = 0
        self._parent_scope: Scope | None = None
        self._pending = False
        self._due = 0.0
        self._seq = 0
        self._started_at = 0.0
        self._start_completed = 0.0
        self._state = "new"
        self._last_out = 0.0
        self._last_completed = 0.0
        self._next_check = 0.0
        self._forecast = ScopeForecast(self.id)
        self._token: Token[Scope | None] | None = None

    def start(self) -> None:
        """Write the ``start`` record. Does nothing if the scope already started."""
        with contextlib.suppress(Exception), self._out._lock:
            if self._state != "new":
                return
            now = self._out.clock.monotonic()
            self._started_at = now
            self._start_completed = self.completed
            self._parent_scope = _PARENT.get()
            if self._parent_scope is not None:
                self.parent = self._parent_scope.id
                self.depth = self._parent_scope.depth + 1
            self._state = "open"
            delay = self._start_delay()
            if delay > 0:
                # It may end before a line is due; then its start and end are one line. Its
                # updates wait as if the start had been written.
                self._pending = True
                self._last_out = now
                self._due = now + delay
                self._out._hold(self, self._due, self._write_pending_start)
            else:
                self._emit(Event.START, now)

    def advance(self, n: float = 1) -> None:
        """Add ``n`` to the completed amount."""
        with contextlib.suppress(Exception), self._out._lock:
            self._set_completed(self.completed + _count(n))

    def update(self, completed: float) -> None:
        """Set the completed amount; a value that is not a non-negative number is ignored."""
        with contextlib.suppress(Exception), self._out._lock:
            number = _number(completed)
            if number is not None and number >= 0:
                self._set_completed(number)

    def set_total(self, total: float | None) -> None:
        """Change the total; ``None`` or a non-positive number makes it unknown."""
        with contextlib.suppress(Exception), self._out._lock:
            self.total = _total(total)
            self._after_change()

    def declare(self, phases: tuple[tuple[str, float], ...]) -> None:
        """Write a ``plan`` record from this scope and close it; used by :func:`plan`."""
        with contextlib.suppress(Exception), self._out._lock:
            now = self._out.clock.monotonic()
            self._started_at = now
            self._state = "closed"
            self._emit(Event.PLAN, now, phases=phases)

    def end(self) -> None:
        """Write the ``end`` record. Does nothing if the scope has already ended."""
        self._finish(Event.END)

    def fail(self) -> None:
        """Write the ``fail`` record. Does nothing if the scope has already ended."""
        self._finish(Event.FAIL)

    def __enter__(self) -> Self:
        """Start the scope and make it the parent of scopes opened inside."""
        self.start()
        self.bind()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        """End the scope, or fail it if an exception is leaving; never suppress the exception."""
        self.unbind()
        if exc_type is None:
            self.end()
        else:
            self.fail()

    def bind(self) -> None:
        """Make this scope the parent of scopes opened from now on in this context."""
        self._token = _PARENT.set(self)

    def unbind(self) -> None:
        """Restore the parent that was in place before :meth:`bind`."""
        token, self._token = self._token, None
        if token is None:
            return
        try:
            _PARENT.reset(token)
        except ValueError:
            _PARENT.set(self._parent_scope)

    def _set_completed(self, completed: float) -> None:
        self.completed = min(completed, _MAX_NUMBER)
        self._after_change()

    def _after_change(self) -> None:
        if self._state == "new":
            self.start()
        if self._state != "open":
            return
        now = self._out.clock.monotonic()
        # The rank check reads the environment, so it comes after the cheap time gate.
        target = None
        if now >= self._next_check and self._out._settings.progress_enabled():
            target = self._out._target()
        if target is not None:
            settings = self._out._settings
            gap = _REDRAW_INTERVAL if target.bar else min(_REDRAW_INTERVAL, settings.resolve("min_interval"))
            self._next_check = now + gap
            if target.bar or self._is_due(now, settings.resolve("min_interval"), settings.resolve("heartbeat")):
                # Writing the update also writes the held lines that are due, this scope's start among them.
                self._emit(Event.UPDATE, now, target)
                return
        self._out._write_held()

    def _is_due(self, now: float, min_interval: float, heartbeat: float) -> bool:
        since = now - self._last_out
        if since < min_interval:
            return False
        if since >= heartbeat:
            return True
        if self.total is None:
            return False
        return self.completed - self._last_completed >= _UPDATE_FRACTION * self.total

    def _finish(self, event: Event) -> None:
        with contextlib.suppress(Exception), self._out._lock:
            if self._state == "new":
                self.start()
            if self._state != "open":
                return
            self._state = "closed"
            self._emit(event, self._out.clock.monotonic())

    def _start_delay(self) -> float:
        """Return how long a one-off step or a child scope may hold back its start line; 0 to write it now."""
        child = self._parent_scope is not None and not self.root and not self.resumed
        if not (self._ONE_OFF or child) or not self._out._settings.progress_enabled():
            return 0.0
        target = self._out._target()
        if target is None or target.bar:
            return 0.0
        return self._out._settings.resolve("min_interval")

    def _write_pending_start(self) -> None:
        """Write the start held back by a scope that is still running, and the starts it must follow."""
        with contextlib.suppress(Exception), self._out._lock:
            if self._pending:
                self._write_late_start()
                now = self._out.clock.monotonic()
                if not self._ONE_OFF and now >= self._due:
                    # The start says when the child began and this how far it has got since. Its
                    # next update stays due as if the start had been written on time, so the
                    # child still writes every record it would have written without holding.
                    self._emit(Event.UPDATE, now)
                    self._last_out, self._last_completed = self._started_at, self._start_completed

    def _write_late_start(self) -> None:
        self._pending = False
        self._out._release(self)
        if self._ONE_OFF:
            self._emit(Event.START, self._out.clock.monotonic())
            return
        # A child's start is written as it was when it started, so a reader times its first units.
        completed, self.completed = self.completed, self._start_completed
        try:
            self._emit(Event.START, self._started_at)
        finally:
            self.completed = completed

    def _settle_pending(self, event: Event) -> Event:
        """Return the event to write: a pending start is written first, or folded into a step."""
        if self._parent_scope is not None:
            # A reader meets a parent before its children, and the text indents under it.
            self._parent_scope._write_pending_start()
        if not self._pending or event is Event.START:
            return event
        folds = not self._ONE_OFF or (self.completed == 1 and self.total == 1)
        if event is Event.END and folds and self._out.clock.monotonic() < self._due:
            self._pending = False
            return Event.STEP
        self._write_late_start()
        return event

    def _record(self, event: Event, now: float, phases: tuple[tuple[str, float], ...] = ()) -> Record:
        self._seq += 1
        first = event in (Event.START, Event.STEP)
        return Record(
            scope=self.id,
            event=event,
            seq=self._seq if event in (Event.UPDATE, Event.PLAN) else None,
            elapsed=max(now - self._started_at, 0.0),
            completed=None if event is Event.PLAN else self.completed,
            total=self.total,
            # A one-off step is written as the bare one-unit step; a child's step carries its counts and unit.
            unit=self.unit if event is Event.START or (first and not self._ONE_OFF) else None,
            parent=self.parent,
            root=self.root,
            resumed=first and self.resumed,
            phases=phases,
            label=(self.description.strip() or None) if first else None,
            run=self._out.run_id,
        )

    def _emit(
        self,
        event: Event,
        now: float,
        target: _Target | None = None,
        phases: tuple[tuple[str, float], ...] = (),
    ) -> None:
        if not self._out._settings.progress_enabled():
            return
        target = target or self._out._target()
        if target is None:
            return
        if self._pending:
            # This scope's own held start is settled below, against the line being written.
            self._out._release(self)
        self._out._write_held()
        event = self._settle_pending(event)
        record = self._record(event, now, phases)
        self._last_out = now
        self._last_completed = self.completed
        if event is Event.PLAN:
            text = plan_text(phases)
        else:
            self._forecast.observe(record, now)
            forecast = self._forecast.snapshot(now)
            drawing = target.bar and event in (Event.START, Event.UPDATE)
            text = progress_text(
                record, forecast, self.description, self.unit, target.charset, bar=drawing, depth=self.depth
            )
            if drawing:
                self._out._draw_bar(text, target)
                return
        self._out._write_line(self._progress_line(text, record, target), target)

    def _progress_line(self, text: str, record: Record, target: _Target) -> str:
        stamp = format_timestamp(self._out.clock.wall())
        # A child folded into one line keeps the PROGRESS label its start and end would have had.
        name = STEP_LABEL if _is_bare_step(record) else PROGRESS_LABEL
        label = colorize(name, target.color) + " " * (len(PROGRESS_LABEL) - len(name))
        prefix = f"{stamp} {label} "
        trailer = f"  {format_trailer(record)}"
        # A reader rejects lines over _MAX_LINE_BYTES, so the readable text gives way.
        budget = _MAX_LINE_BYTES - len(prefix.encode("utf-8")) - len(trailer.encode("utf-8"))
        return prefix + _fit_bytes(text, budget) + trailer


class Phase(Scope):
    """A named stage of a larger job: a scope of total 1 that parents the scopes opened inside it.

    Create it with :func:`phase`. Pass it to :func:`plan` before the work starts to
    declare its share of the job.

    Parameters
    ----------
    name : str
        Phase name, also the source of its scope id.
    weight : float, optional
        Share of the whole job, used by :func:`plan`.
    unit : str
        Unit shown beside the single count.
    scope : str, optional
        Explicit scope id.
    logger : Logger, optional
        Output to use.
    """

    _ONE_OFF = True

    def __init__(
        self,
        name: str,
        weight: float | None = None,
        *,
        unit: str = "phase",
        scope: str | None = None,
        logger: Logger | None = None,
    ) -> None:
        super().__init__(name, 1, unit, scope=scope, logger=logger)
        self.name = name
        self.weight = _total(weight)

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        """Mark the phase complete unless an exception is leaving, then end or fail it."""
        if exc_type is None and self._state == "open":
            self.completed = 1.0
        super().__exit__(exc_type, exc, traceback)


_PlanEntry = Phase | str | tuple[str, float]


def progress(
    total: float | None = None,
    description: str = "",
    unit: str = "it",
    *,
    scope: str | None = None,
    root: bool = False,
    completed: float = 0,
    resumed: bool = False,
    logger: Logger | None = None,
) -> Scope:
    """Report progress of a counted piece of work, as a context manager.

    ``start`` is written on entry, ``upd`` as the work advances, ``end`` on normal exit
    and ``fail`` when an exception leaves (the exception is re-raised). A scope opened
    inside another that ends before its first line is due writes one ``step`` line instead.

    Parameters
    ----------
    total : float, optional
        Total amount; ``None`` when unknown.
    description : str
        What the work is.
    unit : str
        Unit of the counts.
    scope : str, optional
        Explicit scope id.
    root : bool
        Mark this scope as standing for the whole job.
    completed : float
        Amount already done at the start.
    resumed : bool
        Whether this resumes earlier work.
    logger : Logger, optional
        Output to use; the module-level logger by default.

    Returns
    -------
    Scope
        The handle: ``advance(n=1)``, ``update(completed)`` and ``set_total(total)``.

    Notes
    -----
    The positional order is ``total, description, unit``, unlike :func:`track`, which
    puts ``total`` last, after ``iterable``, ``description`` and ``unit``. The order is part
    of the 1.0 API and does not change; pass ``description`` and ``unit`` by keyword.
    """
    return Scope(description, total, unit, scope=scope, root=root, completed=completed, resumed=resumed, logger=logger)


def track(
    iterable: Iterable[_T],
    description: str = "",
    unit: str = "it",
    total: float | None = None,
    *,
    scope: str | None = None,
    root: bool = False,
    completed: float = 0,
    resumed: bool = False,
    logger: Logger | None = None,
) -> Iterator[_T]:
    """Yield the items of ``iterable`` and report progress as the loop consumes them.

    The scope starts when iteration begins. An item counts as completed when the loop
    asks for the next one. ``total`` defaults to ``len(iterable)`` when that exists.

    How the scope ends:

    * the iterable is exhausted: ``end``;
    * the iterable raises: ``fail``, and the exception propagates;
    * the loop is abandoned (``break``, ``return``, garbage collection): ``end`` with the
      ``completed`` reached, because the loop finished on the caller's terms.

    A generator cannot tell ``break`` from an exception raised in the loop *body*: both
    reach it as ``GeneratorExit``. A crash in the body therefore also writes ``end``. Use
    :func:`progress` when the scope must be marked failed by an exception in the body.

    Parameters
    ----------
    iterable : iterable
        What to loop over.
    description, unit, total, scope, root, completed, resumed, logger
        As for :func:`progress`.

    Returns
    -------
    iterator
        An iterator over the items of ``iterable``.

    Notes
    -----
    The positional order is ``iterable, description, unit, total``, unlike
    :func:`progress`, which starts with ``total``. The order is part of the 1.0 API and
    does not change; pass everything after ``iterable`` by keyword.
    """
    if total is None:
        with contextlib.suppress(Exception):
            total = len(iterable) if isinstance(iterable, Sized) else None
    handle = Scope(
        description, total, unit, scope=scope, root=root, completed=completed, resumed=resumed, logger=logger
    )
    return _iterate(iterable, handle)


def _iterate(iterable: Iterable[_T], handle: Scope) -> Iterator[_T]:
    try:
        handle.start()
        handle.bind()
        for item in iterable:
            yield item
            handle.advance()
    except GeneratorExit:
        handle.end()
        raise
    except BaseException:
        handle.fail()
        raise
    else:
        handle.end()
    finally:
        handle.unbind()


def timed(description: Any = "", *, scope: str | None = None, logger: Logger | None = None) -> Any:
    """Decorate a function so each call is a scope of total 1 that starts and ends around it.

    Works with or without arguments (``@timed`` and ``@timed("Load model")``). With no
    description the function's name is used. An exception writes ``fail`` and propagates.
    A generator or coroutine function is timed only until the call returns the generator
    or coroutine object.

    Parameters
    ----------
    description : str
        Description of the work.
    scope : str, optional
        Explicit scope id.
    logger : Logger, optional
        Output to use.

    Returns
    -------
    callable
        A decorator, or the decorated function when used without arguments.
    """
    if callable(description):
        return _timed(description, getattr(description, "__name__", "call"), scope, logger)

    def decorate(func: Callable[..., _T]) -> Callable[..., _T]:
        return _timed(func, description or func.__name__, scope, logger)

    return decorate


def _timed(func: Callable[..., _T], description: str, scope: str | None, logger: Logger | None) -> Callable[..., _T]:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> _T:
        with Phase(description, unit="call", scope=scope, logger=logger):
            return func(*args, **kwargs)

    return wrapper


def phase(
    name: str,
    weight: float | None = None,
    *,
    scope: str | None = None,
    logger: Logger | None = None,
) -> Phase:
    """Open a named stage of a larger job; scopes opened inside it name it as their parent.

    Parameters
    ----------
    name : str
        Phase name.
    weight : float, optional
        Share of the whole job. Written only by :func:`plan`, which accepts the phase object.
    scope : str, optional
        Explicit scope id.
    logger : Logger, optional
        Output to use.

    Returns
    -------
    Phase
        A context manager: ``with phase("Preprocess", weight=0.05): ...``.
    """
    return Phase(name, weight, scope=scope, logger=logger)


def plan(phases: Sequence[_PlanEntry], *, scope: str | None = None, logger: Logger | None = None) -> None:
    """Write a ``plan`` record declaring the phases still to come and their weights.

    Each entry is a :class:`Phase` (from :func:`phase`, not yet entered), a name, or a
    ``(name, weight)`` pair. A name becomes the scope id the phase of that name will
    get, provided no earlier scope used it. Entries without a weight share what the
    others leave of 1.0 equally (all equal when none has a weight).

    Parameters
    ----------
    phases : sequence
        The phases, in the order they will run.
    scope : str, optional
        Scope id for the plan record itself; ``plan`` by default.
    logger : Logger, optional
        Output to use.
    """
    out = logger or default_logger
    with contextlib.suppress(Exception), out._lock:
        name = scope or "plan"
        # Every plan() under one name shares a scope, so q orders them and a reader
        # never lets a late copy of an older plan replace a newer one.
        declaring = out._plan_scopes.get(name)
        if declaring is None:
            declaring = out._plan_scopes[name] = Scope(scope=name, logger=out)
        declaring.declare(_weighted(phases, out))


def _fit_bytes(text: str, budget: int) -> str:
    """Return ``text`` cut to at most ``budget`` UTF-8 bytes, ending in ``...`` when cut."""
    data = text.encode("utf-8")
    if len(data) <= budget:
        return text
    if budget <= 3:
        return ""
    return data[: budget - 3].decode("utf-8", errors="ignore") + "..."


def _entry(entry: _PlanEntry, out: Logger, reserved: set[str]) -> tuple[str, float | None]:
    if isinstance(entry, Phase):
        return entry.id, entry.weight
    name, weight = (entry, None) if isinstance(entry, str) else (entry[0], _total(entry[1]))
    # Predict the id the phase will get, counting earlier entries of this plan as taken,
    # so two phases with the same name are listed as "train" and "train-2" like their scopes.
    return out._predict_scope(_derive_scope_id(name), reserved), weight


def _weighted(phases: Sequence[_PlanEntry], out: Logger) -> tuple[tuple[str, float], ...]:
    entries: list[tuple[str, float | None]] = []
    reserved: set[str] = set()
    for phase_entry in phases:
        name, weight = _entry(phase_entry, out, reserved)
        reserved.add(name)
        entries.append((name, weight))
    given = sum(weight for _, weight in entries if weight is not None)
    missing = sum(1 for _, weight in entries if weight is None)
    share = max(1.0 - given, 0.0) / missing if missing else 0.0
    if missing == len(entries):
        share = 1.0 / max(len(entries), 1)
    return tuple((name, share if weight is None else weight) for name, weight in entries)
