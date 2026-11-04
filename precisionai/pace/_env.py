# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Configuration, process rank and the clock behind the logger and progress APIs.

Every setting resolves in the same order: a value passed to ``Logger.configure``,
then the ``PACE_*`` environment variable, then the built-in default. Environment
variables are read each time they are needed, so a scheduler (or a test) can change
them without restarting the process.
"""

import contextlib
import math
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

DEFAULT_MIN_INTERVAL = 15.0
DEFAULT_HEARTBEAT = 120.0

LEVELS: dict[str, int] = {
    "DEBUG": 10,
    "INFO": 20,
    "SUCCESS": 25,
    "WARNING": 30,
    "ERROR": 40,
    "CRITICAL": 50,
}
DEFAULT_LEVEL = LEVELS["INFO"]

_RANK_VARIABLES = ("RANK", "LOCAL_RANK", "SLURM_PROCID")
_COLOR_MODES = ("auto", "always", "never")
_TRUE_WORDS = frozenset({"1", "true", "yes", "on"})


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


def parse_level(value: Any) -> int | None:
    """Return the numeric level for a name or number, or ``None`` if it is not one.

    Parameters
    ----------
    value : str or int
        A level name (any case) or a number.

    Returns
    -------
    int or None
        The numeric level.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return LEVELS.get(value.strip().upper())
    return None


def level_name(number: int) -> str:
    """Return the name of the highest level that does not exceed ``number``.

    Parameters
    ----------
    number : int
        A numeric level, such as the one standard ``logging`` uses.

    Returns
    -------
    str
        A level name; ``DEBUG`` for anything below it.
    """
    name = "DEBUG"
    for candidate, value in LEVELS.items():
        if value <= number:
            name = candidate
    return name


def current_rank(env: dict[str, str] | None = None) -> int:
    """Return this process's rank from ``RANK``, ``LOCAL_RANK`` or ``SLURM_PROCID``.

    Parameters
    ----------
    env : mapping, optional
        Environment to read; defaults to ``os.environ``.

    Returns
    -------
    int
        The first variable that holds an integer, or ``0`` when none does.
    """
    source = os.environ if env is None else env
    for name in _RANK_VARIABLES:
        rank = _parse_int(source.get(name))
        if rank is not None:
            return rank
    return 0


def _parse_int(text: str | None) -> int | None:
    try:
        return int((text or "").strip())
    except ValueError:
        return None


def _env_text(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    return value.strip().lower()


def _check_level(value: Any) -> int:
    parsed = parse_level(value)
    if parsed is None:
        raise ValueError(f"unknown level {value!r}; expected one of {', '.join(LEVELS)}")
    return parsed


def _check_mode(name: str, value: Any) -> str:
    if value not in _COLOR_MODES:
        raise ValueError(f"{name} must be 'auto', 'always' or 'never', got {value!r}")
    return value


def _check_seconds(name: str, value: Any) -> float:
    number = value if isinstance(value, (int, float)) and not isinstance(value, bool) else None
    if number is None or not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be a non-negative number of seconds, got {value!r}")
    return value


def _same(value: Any) -> Any:
    return value


def _stream(value: Any) -> Any:
    """Look up a stream name now, so that a redirected ``sys.stdout`` is honoured; an object is used as is."""
    if not isinstance(value, str):
        return value
    return sys.stderr if value == "stderr" else sys.stdout


@dataclass(frozen=True, slots=True)
class _Option:
    """One setting: its environment variable, how a value is checked, and its default.

    ``check`` raises ``ValueError`` for a bad explicit value. The environment holds text, which
    ``convert`` turns into a value before ``check`` sees it; a bad environment value is not an
    error, the default applies. ``finish`` is the last step, for a value looked up when it is read.
    """

    env: str
    check: Callable[[Any], Any]
    default: Any
    convert: Callable[[str], Any] = _same
    finish: Callable[[Any], Any] = _same


_OPTIONS: dict[str, _Option] = {
    "level": _Option("PACE_LEVEL", _check_level, DEFAULT_LEVEL),
    "stream": _Option("PACE_STREAM", _same, "stdout", finish=_stream),
    "color": _Option("PACE_COLOR", partial(_check_mode, "color"), "auto"),
    "bar": _Option("PACE_BAR", partial(_check_mode, "bar"), "auto"),
    "min_interval": _Option("PACE_MIN_INTERVAL", partial(_check_seconds, "min_interval"), DEFAULT_MIN_INTERVAL, float),
    "heartbeat": _Option("PACE_HEARTBEAT", partial(_check_seconds, "heartbeat"), DEFAULT_HEARTBEAT, float),
    "all_ranks": _Option("PACE_ALL_RANKS", bool, False, _TRUE_WORDS.__contains__),
}


@dataclass
class Settings:
    """Explicit settings, each ``None`` until ``configure`` sets it.

    A ``None`` field defers to the environment variable, then to the default. Read a setting with
    :meth:`resolve`; the environment is read each time, so a change applies to a running process.
    """

    level: int | None = None
    stream: Any = None
    color: str | None = None
    bar: str | None = None
    min_interval: float | None = None
    heartbeat: float | None = None
    all_ranks: bool | None = None

    def update(self, **values: Any) -> None:
        """Set each value that is not ``None``, after checking all of them, so a bad one changes nothing.

        Raises
        ------
        ValueError
            If a value is not valid for its setting.
        """
        checked = {name: _OPTIONS[name].check(value) for name, value in values.items() if value is not None}
        for name, value in checked.items():
            setattr(self, name, value)

    def resolve(self, name: str) -> Any:
        """Return the value of setting ``name``: explicit, else its environment variable, else its default.

        A stream is looked up now, so redirection is honoured; a value in the environment that is not
        valid for its setting is ignored.
        """
        option = _OPTIONS[name]
        value = getattr(self, name)
        if value is None:
            value = option.default
            text = _env_text(option.env)
            if text is not None:
                with contextlib.suppress(ValueError):
                    value = option.check(option.convert(text))
        return option.finish(value)

    def progress_enabled(self) -> bool:
        """Return whether this process writes progress lines (rank 0, or every rank if allowed)."""
        return self.resolve("all_ranks") or current_rank() == 0
