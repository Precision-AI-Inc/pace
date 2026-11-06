# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""The logger: levelled log lines, and the output path progress lines share.

Writing never raises into the caller. A record that cannot be formatted or written
is dropped.

Configuration precedence, highest first: a keyword passed to :meth:`Logger.configure`,
the ``PACE_*`` environment variable, the default. Environment variables are read at
write time, so they apply to a running process.

========================  =======================================================
``PACE_LEVEL``            minimum log level, default ``INFO``
``PACE_STREAM``           ``stdout`` (default) or ``stderr``
``PACE_COLOR``            ``auto`` (default), ``always`` or ``never``
``PACE_BAR``              in-place bar: ``auto`` (on a terminal), ``always``, ``never``
``PACE_MIN_INTERVAL``     seconds between progress lines of one scope, default 15
``PACE_HEARTBEAT``        seconds after which a line is written regardless, default 120
``PACE_ALL_RANKS``        ``1`` to write progress from every rank, not only rank 0
========================  =======================================================

Progress lines are not filtered by level: a scheduler depends on them.
"""

import contextlib
import os
import secrets
import shutil
import sys
import threading
import time
import traceback
from collections.abc import Callable, Collection
from dataclasses import dataclass
from typing import Any

from precisionai.pace._env import LEVELS, level_name, parse_level
from precisionai.pace._env import Settings as _Settings
from precisionai.pace._render import Charset, charset_for, encode_safely, fit, format_timestamp, log_line

__all__ = ["LEVELS", "Clock", "Logger", "logger"]

_DEFAULT_TERMINAL_COLUMNS = 80


@dataclass(frozen=True, slots=True)
class Clock:
    """The two clocks the package reads.

    Parameters
    ----------
    monotonic : callable
        Seconds on a monotonic clock. Used for every rate, interval and ``e=`` value.
    wall : callable
        Seconds since the epoch. Used only for the printed timestamp.
    """

    monotonic: Callable[[], float] = time.monotonic
    wall: Callable[[], float] = time.time


@dataclass(frozen=True, slots=True)
class _Target:
    """Where one line is going and what that stream can show.

    Parameters
    ----------
    stream : file-like
        The stream, resolved at write time.
    encoding : str
        The stream's encoding.
    charset : Charset
        Unicode glyphs on a UTF-8 stream, ASCII otherwise.
    tty : bool
        Whether the stream is a terminal.
    color : bool
        Whether to colour the level label.
    bar : bool
        Whether progress is drawn as one in-place line.
    """

    stream: Any
    encoding: str
    charset: Charset
    tty: bool
    color: bool
    bar: bool


def _is_tty(stream: Any) -> bool:
    try:
        return bool(stream.isatty())
    except (AttributeError, ValueError, OSError):
        return False


class Logger:
    """Write levelled log lines and carry the shared output state for progress lines.

    Parameters
    ----------
    clock : Clock, optional
        Monotonic and wall clocks; replace them in tests instead of sleeping.
    """

    def __init__(self, clock: Clock | None = None) -> None:
        self.clock = clock or Clock()
        self._settings = _Settings()
        self._lock = threading.RLock()
        # Written on every start record, so a reader can tell a restarted process
        # from the same start line read twice.
        self.run_id = secrets.token_hex(4)
        # plan() scopes by name, kept so repeated plans share a scope and a q sequence.
        self._plan_scopes: dict[str, Any] = {}
        self._scope_ids: set[str] = set()
        self._next_suffix: dict[str, int] = {}
        self._bar_width = 0
        self._bar_stream: Any = None
        # Lines held back by one-off steps and child scopes, by key: (deadline, how to write it, owning process id).
        self._held: dict[object, tuple[float, Callable[[], None], int]] = {}
        self._writing_held = False

    def configure(
        self,
        *,
        level: Any = None,
        stream: Any = None,
        color: str | None = None,
        bar: str | None = None,
        min_interval: float | None = None,
        heartbeat: float | None = None,
        all_ranks: bool | None = None,
        clock: Clock | None = None,
    ) -> None:
        """Set options. An option left out keeps its current value.

        An explicit value here beats the matching environment variable, which beats
        the default. Call :meth:`reset` to go back to environment and defaults.

        Parameters
        ----------
        level : str or int, optional
            Minimum level written.
        stream : {"stdout", "stderr"} or file-like, optional
            Where to write. A name (any case) is looked up at write time; an object is
            used as is. Any other name warns and means ``stdout``.
        color : {"auto", "always", "never"}, optional
            Colour of the level label.
        bar : {"auto", "always", "never"}, optional
            Whether progress is drawn as one in-place line (``auto``: on a terminal).
        min_interval : float, optional
            Seconds between progress lines of one scope.
        heartbeat : float, optional
            Seconds after which a progress line is written whatever the progress.
        all_ranks : bool, optional
            Write progress from every rank instead of rank 0 only.
        clock : Clock, optional
            Replacement clocks, for tests.

        Raises
        ------
        ValueError
            If ``level``, ``color``, ``bar`` or an interval is not a valid value.

        Warns
        -----
        UserWarning
            If ``stream`` is a name other than ``stdout`` or ``stderr``. It does not raise,
            because 1.0.0 accepted any name as ``stdout``.
        """
        with self._lock:
            self._settings.update(
                level=level,
                stream=stream,
                color=color,
                bar=bar,
                min_interval=min_interval,
                heartbeat=heartbeat,
                all_ranks=all_ranks,
            )
            if clock is not None:
                self.clock = clock

    def reset(self) -> None:
        """Forget every explicit setting, so the environment and defaults apply again."""
        with self._lock:
            self._settings = _Settings()

    def debug(self, message: object) -> None:
        """Write a ``DEBUG`` line."""
        self.log("DEBUG", message)

    def info(self, message: object) -> None:
        """Write an ``INFO`` line."""
        self.log("INFO", message)

    def success(self, message: object) -> None:
        """Write a ``SUCCESS`` line."""
        self.log("SUCCESS", message)

    def warning(self, message: object) -> None:
        """Write a ``WARNING`` line."""
        self.log("WARNING", message)

    def error(self, message: object) -> None:
        """Write an ``ERROR`` line."""
        self.log("ERROR", message)

    def critical(self, message: object) -> None:
        """Write a ``CRITICAL`` line."""
        self.log("CRITICAL", message)

    def exception(self, message: object) -> None:
        """Write an ``ERROR`` line followed by the traceback of the exception being handled."""
        with contextlib.suppress(Exception):
            text = str(message)
            if sys.exc_info()[0] is not None:
                text = f"{text}\n{traceback.format_exc().rstrip()}"
            self.log("ERROR", text)

    def log(self, level: Any, message: object) -> None:
        """Write a line at ``level``, a level name or a number; an unknown level is written as ``INFO``.

        Parameters
        ----------
        level : str or int
            Level name (any case) or a number such as the one standard ``logging`` uses.
        message : object
            Converted with ``str``.
        """
        with contextlib.suppress(Exception):
            number = parse_level(level)
            number = LEVELS["INFO"] if number is None else number
            label = level_name(number)
            with self._lock:
                if number < self._settings.resolve("level"):
                    return
                target = self._target()
                if target is None:
                    return
                self._write_held()
                line = log_line(format_timestamp(self.clock.wall()), label, str(message), target.color)
                self._write_line(line, target)

    def _target(self) -> _Target | None:
        """Resolve the stream now and describe what it can show; ``None`` when there is no usable stream."""
        stream = self._settings.resolve("stream")
        if stream is None or not callable(getattr(stream, "write", None)):
            return None
        encoding = getattr(stream, "encoding", None)
        encoding = encoding if isinstance(encoding, str) and encoding else "ascii"
        tty = _is_tty(stream)
        color_mode = self._settings.resolve("color")
        bar_mode = self._settings.resolve("bar")
        return _Target(
            stream=stream,
            encoding=encoding,
            charset=charset_for(encoding),
            tty=tty,
            color=color_mode == "always" or (color_mode == "auto" and tty and not os.environ.get("NO_COLOR")),
            bar=bar_mode == "always" or (bar_mode == "auto" and tty),
        )

    def _write_line(self, text: str, target: _Target) -> None:
        """Write one line, first erasing any in-place bar so the two never overlap."""
        with self._lock:
            self._erase_bar(target)
            self._put(target, text + "\n")

    def _draw_bar(self, text: str, target: _Target) -> None:
        """Redraw the single in-place line with ``text``."""
        with self._lock:
            columns = shutil.get_terminal_size((_DEFAULT_TERMINAL_COLUMNS, 24)).columns - 1
            shown = fit(text, max(columns, 1), target.charset)
            pad = max(self._bar_width - len(shown), 0) if self._bar_stream is target.stream else 0
            self._put(target, "\r" + shown + " " * pad)
            self._bar_width = len(shown) + pad
            self._bar_stream = target.stream

    def _erase_bar(self, target: _Target) -> None:
        if self._bar_width and self._bar_stream is target.stream:
            self._put(target, "\r" + " " * self._bar_width + "\r")
        self._bar_width = 0
        self._bar_stream = None

    @staticmethod
    def _put(target: _Target, text: str) -> None:
        target.stream.write(encode_safely(text, target.encoding))
        target.stream.flush()

    def _hold(self, key: object, deadline: float, action: Callable[[], None]) -> None:
        """Hold back a line: run ``action`` at the first write once ``deadline`` has passed, unless released.

        Pace starts no threads, so a held-back line waits for the next line this logger writes
        (or the next progress check of any scope). A forked child never writes its parent's.

        Parameters
        ----------
        key : object
            Names the held line, for :meth:`release`.
        deadline : float
            When it is due, on :attr:`clock`'s monotonic clock.
        action : callable
            Writes the line.
        """
        with self._lock:
            self._held[key] = (deadline, action, os.getpid())

    def _release(self, key: object) -> None:
        """Drop the line held under ``key``, if it has not been written."""
        with self._lock:
            self._held.pop(key, None)

    def _write_held(self) -> None:
        """Write the held-back lines that are due; called before each line this logger writes."""
        if not self._held:
            return
        with self._lock:
            # A held line's own write calls back in here; the outer pass writes the rest in the
            # order they were held, which keeps a parent's start ahead of its children's.
            if self._writing_held:
                return
            self._writing_held = True
            try:
                now, pid = self.clock.monotonic(), os.getpid()
                for key, (deadline, action, owner) in list(self._held.items()):
                    if owner != pid:
                        self._held.pop(key, None)
                    elif deadline <= now and self._held.pop(key, None) is not None:
                        with contextlib.suppress(Exception):
                            action()
            finally:
                self._writing_held = False

    def _claim_scope(self, base: str) -> str:
        """Return ``base`` or ``base`` with a numeric suffix, whichever no earlier scope in this process has used.

        Parameters
        ----------
        base : str
            A valid scope id.

        Returns
        -------
        str
            A scope id of at most 64 characters, unique within this logger.
        """
        with self._lock:
            candidate, number = self._free_scope(base, ())
            self._next_suffix[base] = number
            self._scope_ids.add(candidate)
            return candidate

    def _predict_scope(self, base: str, reserved: Collection[str] = ()) -> str:
        """Return the id :meth:`claim_scope` would give ``base`` next, without claiming it.

        Parameters
        ----------
        base : str
            A valid scope id.
        reserved : collection of str
            Ids to treat as already taken, such as earlier entries of the same plan.

        Returns
        -------
        str
            The predicted scope id.
        """
        with self._lock:
            return self._free_scope(base, reserved)[0]

    def _free_scope(self, base: str, reserved: Collection[str]) -> tuple[str, int]:
        candidate = base
        number = self._next_suffix.get(base, 1)
        while candidate in self._scope_ids or candidate in reserved:
            number += 1
            suffix = f"-{number}"
            candidate = base[: 64 - len(suffix)] + suffix
        return candidate, number


logger = Logger()
