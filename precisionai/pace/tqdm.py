# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""A stand-in for the ``tqdm`` class that reports to Pace instead of drawing a bar.

Some libraries take a progress-bar class from the caller (vLLM's ``LLM.generate(...,
use_tqdm=...)`` is one) and run a long job inside a single call, so nothing the caller
writes around the call can report progress. Pass :class:`PaceTqdm` there and the
library's own progress becomes a Pace scope.

It is written against the ``tqdm`` calls such libraries make, not as a subclass of
``tqdm``, so it needs no dependency and never draws or writes anything itself.
"""

import contextlib
from collections.abc import Iterable, Iterator, Sized
from typing import Any, Self

from precisionai.pace.core import Logger
from precisionai.pace.core import logger as default_logger
from precisionai.pace.scopes import Scope

__all__ = ["PaceTqdm"]

# A caller divides by the elapsed time (vLLM does, for its tokens-per-second figure), so
# it is never reported as zero.
_MIN_ELAPSED = 1e-6

# tqdm's data attributes that share a name with its constructor argument, with tqdm's
# defaults. They are set as values so that a caller reading one (huggingface_hub reads
# ``pos``) gets what tqdm would give, not the do-nothing method ``__getattr__`` returns.
_TQDM_SETTINGS: dict[str, Any] = {
    "ascii": None,
    "bar_format": None,
    "colour": None,
    "delay": 0,
    "dynamic_ncols": False,
    "gui": False,
    "leave": True,
    "lock_args": None,
    "maxinterval": 10.0,
    "mininterval": 0.1,
    "miniters": 0,
    "ncols": None,
    "nrows": None,
    "smoothing": 0.3,
    "unit_divisor": 1000,
    "unit_scale": False,
}


def _ignore(*args: Any, **kwargs: Any) -> None:
    """Accept any call and do nothing."""


class PaceTqdm:
    """Report the progress a library makes through a ``tqdm`` class as a Pace scope.

    The scope starts when the object is created and ends when it is closed, as
    ``tqdm``'s bar does. It works both ways ``tqdm`` is used: wrapping an iterable
    (``PaceTqdm(items, desc="Render")``) and counting by hand
    (``PaceTqdm(total=100)`` and ``update(n)``). Everything a bar would show other than
    the count (postfix text, column widths, colours) is accepted and ignored, but kept
    under ``tqdm``'s attribute name (``leave``, ``ncols``, ``pos``, ...) so a caller that
    reads one gets a value. A ``tqdm`` method this class does not have is accepted and
    ignored too, so a library that calls an unusual one does not fail because of Pace.

    Parameters
    ----------
    iterable : iterable, optional
        What to loop over; its length is the total when ``total`` is not given.
    desc : str, optional
        What the work is, written as the scope's label.
    total : float, optional
        Total amount; ``None`` when unknown.
    unit : str
        Unit of the counts, ``"it"`` by default.
    initial : float
        Amount already done when the scope starts.
    disable : bool
        When true nothing is reported, and iterating just yields the items.
    postfix : object, optional
        Accepted for compatibility and ignored.
    scope : str, optional
        Explicit scope id; derived from ``desc`` by default.
    root : bool
        Mark the scope as standing for the whole job. Leave it off when the library opens
        more than one bar per call.
    logger : Logger, optional
        Output to use; the module-level logger by default.
    **ignored : Any
        Every other ``tqdm`` argument (``leave``, ``file``, ``dynamic_ncols``, ...).

    Notes
    -----
    Pass a configured class with ``functools.partial``, for example
    ``use_tqdm=functools.partial(PaceTqdm, scope="generate")``. Each bar the library opens
    is a scope of its own, made unique with a numeric suffix when the id repeats.
    """

    def __init__(
        self,
        iterable: Iterable[Any] | None = None,
        desc: str | None = None,
        total: float | None = None,
        *,
        unit: str = "it",
        initial: float = 0,
        disable: bool = False,
        postfix: Any = None,
        scope: str | None = None,
        root: bool = False,
        logger: Logger | None = None,
        **ignored: Any,
    ) -> None:
        if total is None and isinstance(iterable, Sized):
            with contextlib.suppress(Exception):
                total = len(iterable)
        self.iterable = iterable
        self._log = logger or default_logger
        self._started = self._log.clock.monotonic()
        self._total = total
        for name, default in _TQDM_SETTINGS.items():
            value = ignored.get(name)
            setattr(self, name, default if value is None else value)
        self.dynamic_miniters = not ignored.get("miniters")
        self.fp = ignored.get("file")
        self.pos = ignored.get("position") or 0
        self.start_t = self.last_print_t = self._log.clock.wall()
        self.last_print_n = initial
        self.desc = "" if desc is None else str(desc)
        self.unit = str(unit)
        self.initial = initial
        self.n = initial
        self.postfix = postfix
        self.disable = bool(disable)
        self._scope: Scope | None = None
        if not self.disable:
            self._scope = Scope(
                self.desc, total, self.unit, scope=scope, root=root, completed=initial, logger=self._log
            )
            self._scope.start()

    def __iter__(self) -> Iterator[Any]:
        """Yield the items, counting each as done when the loop asks for the next.

        The scope ends when the items run out or the loop is abandoned, and fails when
        the iterable raises, as for :func:`~precisionai.pace.scopes.track`.
        """
        if self.iterable is None:
            return
        try:
            for item in self.iterable:
                yield item
                self.update()
        except GeneratorExit:
            self.close()
            raise
        except BaseException:
            self.fail()
            raise
        else:
            self.close()

    def __len__(self) -> int:
        """Return the length of the iterable, or the total when there is none."""
        count = len(self.iterable) if isinstance(self.iterable, Sized) else self._total
        if count is None:
            raise TypeError("object of type 'PaceTqdm' has no len()")
        return int(count)

    def __bool__(self) -> bool:
        """Return true: an object that reports progress is never 'empty'."""
        return True

    def __enter__(self) -> Self:
        """Return the object, so it can be used in a ``with`` block."""
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        """Close the scope, or fail it if an exception is leaving; never suppress the exception."""
        if exc_type is None:
            self.close()
        else:
            self.fail()

    def __getattr__(self, name: str) -> Any:
        """Return a do-nothing method for any other public ``tqdm`` method.

        Only names the object does not have reach here. ``tqdm``'s data attributes (``n``,
        ``total``, ``desc``, ``pos``, ``leave``, ``start_t``, ...) are set as values, so
        reading one gives a value, as with ``tqdm``.
        """
        if name.startswith("_"):
            raise AttributeError(name)
        return _ignore

    @property
    def total(self) -> float | None:
        """The total amount; set it to change the scope's total."""
        return self._total

    @total.setter
    def total(self, value: float | None) -> None:
        self._total = value
        if self._scope is not None:
            self._scope.set_total(value)

    @property
    def format_dict(self) -> dict[str, Any]:
        """The count, total, elapsed time and rate, under the keys ``tqdm``'s ``format_dict`` uses."""
        elapsed = max(self._log.clock.monotonic() - self._started, _MIN_ELAPSED)
        done = float(self.n) - float(self.initial)
        return {
            "n": self.n,
            "total": self._total,
            "elapsed": elapsed,
            "rate": done / elapsed if done > 0 else None,
            "unit": self.unit,
            "prefix": self.desc,
            "postfix": self.postfix,
            "initial": self.initial,
        }

    def update(self, n: float = 1) -> None:
        """Add ``n`` to the completed amount."""
        self.n += n
        self.refresh()

    def refresh(self) -> None:
        """Report the current value of ``n``, which the caller may have set directly."""
        self.last_print_n = self.n
        self.last_print_t = self._log.clock.wall()
        if self._scope is not None:
            self._scope.update(self.n)

    def reset(self, total: float | None = None) -> None:
        """Start counting again from zero, with a new total when one is given."""
        self.n = 0
        self._started = self._log.clock.monotonic()
        self.start_t = self._log.clock.wall()
        if total is not None:
            self.total = total
        self.refresh()

    def set_description(self, desc: str | None = None, refresh: bool = True) -> None:
        """Change the label shown in the readable part of later lines."""
        self.desc = "" if desc is None else str(desc)
        if self._scope is not None:
            self._scope.description = self.desc

    def set_description_str(self, desc: str | None = None, refresh: bool = True) -> None:
        """Change the label, as :meth:`set_description` does (``tqdm`` differs only in the bar's text)."""
        self.set_description(desc, refresh)

    def set_postfix(self, ordered_dict: Any = None, refresh: bool = True, **kwargs: Any) -> None:
        """Accept postfix text for compatibility; it is not written."""
        self.postfix = ordered_dict or kwargs

    def set_postfix_str(self, s: str = "", refresh: bool = True) -> None:
        """Accept postfix text for compatibility; it is not written."""
        self.postfix = s

    def close(self) -> None:
        """End the scope. Does nothing if it has already ended."""
        if self._scope is not None:
            self._scope.end()

    def fail(self) -> None:
        """Mark the scope failed. Does nothing if it has already ended."""
        if self._scope is not None:
            self._scope.fail()

    @staticmethod
    def write(s: Any, file: Any = None, end: str = "\n", nolock: bool = False) -> None:
        """Write a message through the Pace logger, as ``tqdm.write`` prints one beside its bar."""
        default_logger.info(s)
