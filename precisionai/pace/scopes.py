# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Progress scopes: ``progress``, ``track``, ``timed``, ``phase`` and ``plan``.

A scope is one piece of work with a start, a completed amount, an optional total and
an end. Each progress line it writes carries the ``[pace:1 ...]`` trailer built by
:func:`precisionai.pace.protocol.format_trailer`; the text before it is for people.

Output rules (see ``docs/design.md``):

* On a terminal a single in-place bar line is redrawn about ten times a second, with no
  trailer. ``start`` draws it; ``end`` and ``fail`` print a normal line with a trailer.
* Otherwise ``start``, ``end``, ``fail`` and ``plan`` are written at once, and ``upd``
  only after ``PACE_MIN_INTERVAL`` seconds and either 1% more of the total or
  ``PACE_HEARTBEAT`` seconds (with an unknown total, the heartbeat alone).
* Only rank 0 writes progress unless ``PACE_ALL_RANKS=1``.

Scopes nest through a context variable: a scope opened inside ``progress``, ``phase``
or an iterated ``track`` records that scope as its parent. Threads and ``asyncio`` tasks
start with the context they were created in, so a scope opened in a new thread has no
parent unless the context is copied.
"""

from __future__ import annotations

import contextlib
import functools
import math
import re
from collections.abc import Sized
from contextvars import ContextVar, Token
from typing import Any, Callable, Iterable, Iterator, Sequence, Tuple, TypeVar, Union

from precisionai.pace._render import PROGRESS_LABEL, colorize, format_timestamp, plan_text, progress_text
from precisionai.pace.core import Logger, Target
from precisionai.pace.core import logger as default_logger
from precisionai.pace.forecast import ScopeForecast
from precisionai.pace.protocol import SCOPE_PATTERN, Event, Record, format_trailer

REDRAW_INTERVAL = 0.1
MAX_DERIVED_ID_CHARS = 48
MAX_ID_CHARS = 64
FALLBACK_ID = "scope"
UPDATE_FRACTION = 0.01

_UNSAFE_ID_CHARS = re.compile(r"[^A-Za-z0-9._-]+")
_PARENT: ContextVar[str | None] = ContextVar("pace_parent_scope", default=None)

T = TypeVar("T")
PlanEntry = Union["Phase", str, Tuple[str, float]]


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _count(value: Any) -> float:
    number = _number(value)
    return 0.0 if number is None or number < 0 else number


def _total(value: Any) -> float | None:
    number = _number(value)
    return None if number is None or number <= 0 else number


def derive_scope_id(description: str) -> str:
    """Return a scope id from a description: lower case, anything outside the id alphabet as ``-``, at most 48 characters."""
    cleaned = _UNSAFE_ID_CHARS.sub("-", description.lower())[:MAX_DERIVED_ID_CHARS].strip("-")
    return cleaned or FALLBACK_ID


def validate_scope_id(scope: str) -> str:
    """Return ``scope`` if it is a valid scope id, otherwise a sanitised version of it."""
    if SCOPE_PATTERN.fullmatch(scope):
        return scope
    cleaned = _UNSAFE_ID_CHARS.sub("-", scope)[:MAX_ID_CHARS].strip("-")
    return cleaned or FALLBACK_ID


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
        self._out = logger or default_logger
        base = derive_scope_id(description) if scope is None else validate_scope_id(scope)
        self.id = self._out.claim_scope(base)
        self.description = description
        self.unit = unit
        self.root = bool(root)
        self.resumed = bool(resumed)
        self.completed = _count(completed)
        self.total = _total(total)
        self.parent: str | None = None
        self._seq = 0
        self._started_at = 0.0
        self._state = "new"
        self._last_out = 0.0
        self._last_completed = 0.0
        self._next_check = 0.0
        self._active = self._out.settings.progress_enabled()
        self._forecast = ScopeForecast(self.id)
        self._token: Token[str | None] | None = None

    def start(self) -> None:
        """Write the ``start`` record. Does nothing if the scope already started."""
        with contextlib.suppress(Exception), self._out.lock:
            if self._state != "new":
                return
            now = self._out.clock.monotonic()
            self._started_at = now
            self.parent = _PARENT.get()
            self._state = "open"
            self._emit(Event.START, now)

    def advance(self, n: float = 1) -> None:
        """Add ``n`` to the completed amount."""
        with contextlib.suppress(Exception), self._out.lock:
            self._set_completed(self.completed + _count(n))

    def update(self, completed: float) -> None:
        """Set the completed amount; a value that is not a non-negative number is ignored."""
        with contextlib.suppress(Exception), self._out.lock:
            number = _number(completed)
            if number is not None and number >= 0:
                self._set_completed(number)

    def set_total(self, total: float | None) -> None:
        """Change the total; ``None`` or a non-positive number makes it unknown."""
        with contextlib.suppress(Exception), self._out.lock:
            self.total = _total(total)
            self._after_change()

    def declare(self, phases: tuple[tuple[str, float], ...]) -> None:
        """Write a ``plan`` record from this scope and close it; used by :func:`plan`."""
        with contextlib.suppress(Exception), self._out.lock:
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

    def __enter__(self) -> Scope:
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
        self._token = _PARENT.set(self.id)

    def unbind(self) -> None:
        """Restore the parent that was in place before :meth:`bind`."""
        token, self._token = self._token, None
        if token is None:
            return
        try:
            _PARENT.reset(token)
        except ValueError:
            _PARENT.set(self.parent)

    def _set_completed(self, completed: float) -> None:
        self.completed = completed
        self._after_change()

    def _after_change(self) -> None:
        if self._state == "new":
            self.start()
        if self._state != "open" or not self._active:
            return
        now = self._out.clock.monotonic()
        if now < self._next_check:
            return
        target = self._out.target()
        if target is None:
            return
        settings = self._out.settings
        gap = REDRAW_INTERVAL if target.bar else min(REDRAW_INTERVAL, settings.resolve_min_interval())
        self._next_check = now + gap
        if target.bar or self._is_due(now, settings.resolve_min_interval(), settings.resolve_heartbeat()):
            self._emit(Event.UPDATE, now, target)

    def _is_due(self, now: float, min_interval: float, heartbeat: float) -> bool:
        since = now - self._last_out
        if since < min_interval:
            return False
        if since >= heartbeat:
            return True
        if self.total is None:
            return False
        return self.completed - self._last_completed >= UPDATE_FRACTION * self.total

    def _finish(self, event: Event) -> None:
        with contextlib.suppress(Exception), self._out.lock:
            if self._state == "new":
                self.start()
            if self._state != "open":
                return
            self._state = "closed"
            self._emit(event, self._out.clock.monotonic())

    def _record(self, event: Event, now: float, phases: tuple[tuple[str, float], ...] = ()) -> Record:
        self._seq += 1
        first = event is Event.START
        return Record(
            scope=self.id,
            event=event,
            seq=self._seq,
            elapsed=max(now - self._started_at, 0.0),
            completed=None if event is Event.PLAN else self.completed,
            total=self.total,
            unit=self.unit if first else None,
            parent=self.parent,
            root=self.root,
            resumed=first and self.resumed,
            phases=phases,
            label=(self.description.strip() or None) if first else None,
            run=self._out.run_id if first else None,
        )

    def _emit(
        self,
        event: Event,
        now: float,
        target: Target | None = None,
        phases: tuple[tuple[str, float], ...] = (),
    ) -> None:
        if not self._active:
            return
        target = target or self._out.target()
        if target is None:
            return
        record = self._record(event, now, phases)
        self._last_out = now
        self._last_completed = self.completed
        if event is Event.PLAN:
            text = plan_text(phases)
        else:
            self._forecast.observe(record, now)
            forecast = self._forecast.snapshot(now)
            drawing = target.bar and event in (Event.START, Event.UPDATE)
            text = progress_text(record, forecast, self.description, self.unit, target.charset, bar=drawing)
            if drawing:
                self._out.draw_bar(text, target)
                return
        self._out.write_line(self._progress_line(text, record, target), target)

    def _progress_line(self, text: str, record: Record, target: Target) -> str:
        stamp = format_timestamp(self._out.clock.wall())
        label = colorize(PROGRESS_LABEL, target.color)
        return f"{stamp} {label} {text}  {format_trailer(record)}"


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
    and ``fail`` when an exception leaves (the exception is re-raised).

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
    """
    return Scope(description, total, unit, scope=scope, root=root, completed=completed, resumed=resumed, logger=logger)


def track(
    iterable: Iterable[T],
    description: str = "",
    unit: str = "it",
    total: float | None = None,
    *,
    scope: str | None = None,
    root: bool = False,
    completed: float = 0,
    resumed: bool = False,
    logger: Logger | None = None,
) -> Iterator[T]:
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
    """
    if total is None:
        with contextlib.suppress(Exception):
            total = len(iterable) if isinstance(iterable, Sized) else None
    handle = Scope(
        description, total, unit, scope=scope, root=root, completed=completed, resumed=resumed, logger=logger
    )
    return _iterate(iterable, handle)


def _iterate(iterable: Iterable[T], handle: Scope) -> Iterator[T]:
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

    def decorate(func: Callable[..., T]) -> Callable[..., T]:
        return _timed(func, description or func.__name__, scope, logger)

    return decorate


def _timed(func: Callable[..., T], description: str, scope: str | None, logger: Logger | None) -> Callable[..., T]:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
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


def plan(phases: Sequence[PlanEntry], *, scope: str | None = None, logger: Logger | None = None) -> None:
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
    with contextlib.suppress(Exception), out.lock:
        Scope(scope=scope or "plan", logger=out).declare(_weighted(phases, out))


def _entry(entry: PlanEntry, out: Logger, reserved: set[str]) -> tuple[str, float | None]:
    if isinstance(entry, Phase):
        return entry.id, entry.weight
    name, weight = (entry, None) if isinstance(entry, str) else (entry[0], _total(entry[1]))
    # Predict the id the phase will get, counting earlier entries of this plan as taken,
    # so two phases with the same name are listed as "train" and "train-2" like their scopes.
    return out.predict_scope(derive_scope_id(name), reserved), weight


def _weighted(phases: Sequence[PlanEntry], out: Logger) -> tuple[tuple[str, float], ...]:
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
