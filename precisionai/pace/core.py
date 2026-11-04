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

from precisionai.pace._env import LEVELS, Clock, Settings, level_name, parse_level
from precisionai.pace._render import Charset, charset_for, encode_safely, fit, format_timestamp, log_line

DEFAULT_TERMINAL_COLUMNS = 80


@dataclass(frozen=True, slots=True)
class Target:
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
        self.settings = Settings()
        self.lock = threading.RLock()
        # Written on every start record, so a reader can tell a restarted process
        # from the same start line read twice.
        self.run_id = secrets.token_hex(4)
        # plan() scopes by name, kept so repeated plans share a scope and a q sequence.
        self.plan_scopes: dict[str, Any] = {}
        self._scope_ids: set[str] = set()
        self._next_suffix: dict[str, int] = {}
        self._bar_width = 0
        self._bar_stream: Any = None
        # Deferred actions by key: (deadline in real monotonic seconds, action). One thread runs them.
        self._deferred: dict[object, tuple[float, Callable[[], None]]] = {}
        self._wake = threading.Condition(self.lock)
        self._timer: threading.Thread | None = None

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
            Where to write. A name is looked up at write time; an object is used as is.
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
        """
        with self.lock:
            self.settings.update(
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
        with self.lock:
            self.settings = Settings()

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
            with self.lock:
                if number < self.settings.resolve("level"):
                    return
                target = self.target()
                if target is None:
                    return
                line = log_line(format_timestamp(self.clock.wall()), label, str(message), target.color)
                self.write_line(line, target)

    def target(self) -> Target | None:
        """Resolve the stream now and describe what it can show; ``None`` when there is no usable stream."""
        stream = self.settings.resolve("stream")
        if stream is None or not callable(getattr(stream, "write", None)):
            return None
        encoding = getattr(stream, "encoding", None)
        encoding = encoding if isinstance(encoding, str) and encoding else "ascii"
        tty = _is_tty(stream)
        color_mode = self.settings.resolve("color")
        bar_mode = self.settings.resolve("bar")
        return Target(
            stream=stream,
            encoding=encoding,
            charset=charset_for(encoding),
            tty=tty,
            color=color_mode == "always" or (color_mode == "auto" and tty and not os.environ.get("NO_COLOR")),
            bar=bar_mode == "always" or (bar_mode == "auto" and tty),
        )

    def write_line(self, text: str, target: Target) -> None:
        """Write one line, first erasing any in-place bar so the two never overlap."""
        with self.lock:
            self._erase_bar(target)
            self._put(target, text + "\n")

    def draw_bar(self, text: str, target: Target) -> None:
        """Redraw the single in-place line with ``text``."""
        with self.lock:
            columns = shutil.get_terminal_size((DEFAULT_TERMINAL_COLUMNS, 24)).columns - 1
            shown = fit(text, max(columns, 1), target.charset)
            pad = max(self._bar_width - len(shown), 0) if self._bar_stream is target.stream else 0
            self._put(target, "\r" + shown + " " * pad)
            self._bar_width = len(shown) + pad
            self._bar_stream = target.stream

    def _erase_bar(self, target: Target) -> None:
        if self._bar_width and self._bar_stream is target.stream:
            self._put(target, "\r" + " " * self._bar_width + "\r")
        self._bar_width = 0
        self._bar_stream = None

    @staticmethod
    def _put(target: Target, text: str) -> None:
        target.stream.write(encode_safely(text, target.encoding))
        target.stream.flush()

    def defer(self, key: object, delay: float, action: Callable[[], None]) -> None:
        """Run ``action`` after ``delay`` seconds, from a background thread, unless :meth:`cancel` comes first.

        The delay is real time whatever clock the logger has, so a test's clock never sets it
        off. One daemon thread serves every deferred action, holding :attr:`lock` while it
        runs one; it ends when none is left, and the next call starts another.

        Parameters
        ----------
        key : object
            Names the action, for :meth:`cancel`. Deferring under a key again replaces it.
        delay : float
            Seconds to wait.
        action : callable
            What to run. An exception it raises is dropped.
        """
        with self.lock:
            self._deferred[key] = (time.monotonic() + delay, action)
            if self._timer is None or not self._timer.is_alive():
                self._timer = threading.Thread(target=self._run_timer, name="pace-timer", daemon=True)
                self._timer.start()
            self._wake.notify()

    def cancel(self, key: object) -> None:
        """Drop the action deferred under ``key``, if it has not run yet."""
        with self.lock:
            if self._deferred.pop(key, None) is not None:
                # Wake the timer, so it ends now if nothing else is deferred.
                self._wake.notify()

    def run_due(self, now: float) -> float | None:
        """Run the deferred actions whose time has come.

        Parameters
        ----------
        now : float
            The current time, in real monotonic seconds (``time.monotonic()``).

        Returns
        -------
        float or None
            Seconds until the next deferred action is due, or ``None`` when none is left.
        """
        with self.lock:
            for key, (deadline, action) in list(self._deferred.items()):
                if deadline <= now:
                    del self._deferred[key]
                    with contextlib.suppress(Exception):
                        action()
            deadlines = [deadline for deadline, _ in self._deferred.values()]
            return max(min(deadlines) - now, 0.0) if deadlines else None

    def _run_timer(self) -> None:
        # Ends once nothing is deferred, so an idle logger holds no thread; defer() starts a new
        # one. Both check and clear self._timer under the lock, so neither can miss the other.
        with self.lock:
            while not sys.is_finalizing():
                wait = self.run_due(time.monotonic())
                if wait is None:
                    break
                self._wake.wait(wait)
            self._timer = None

    def claim_scope(self, base: str) -> str:
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
        with self.lock:
            candidate, number = self._free_scope(base, ())
            self._next_suffix[base] = number
            self._scope_ids.add(candidate)
            return candidate

    def predict_scope(self, base: str, reserved: Collection[str] = ()) -> str:
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
        with self.lock:
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
