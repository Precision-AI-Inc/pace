# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Configuration, process rank and the clock behind the logger and progress APIs.

Every setting resolves in the same order: a value passed to ``Logger.configure``,
then the ``PACE_*`` environment variable, then the built-in default. Environment
variables are read each time they are needed, so a scheduler (or a test) can change
them without restarting the process.
"""

import math
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
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


def _env_seconds(name: str) -> float | None:
    text = _env_text(name)
    if text is None:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) and value >= 0 else None


def _choice(explicit: str | None, variable: str) -> str:
    for candidate in (explicit, _env_text(variable)):
        if candidate in _COLOR_MODES:
            return str(candidate)
    return "auto"


@dataclass
class Settings:
    """Explicit settings, each ``None`` until ``configure`` sets it.

    A ``None`` field defers to the environment variable, then to the default.
    """

    level: int | None = None
    stream: Any = None
    color: str | None = None
    bar: str | None = None
    min_interval: float | None = None
    heartbeat: float | None = None
    all_ranks: bool | None = None

    def resolve_level(self) -> int:
        """Return the minimum level that is written."""
        if self.level is not None:
            return self.level
        text = _env_text("PACE_LEVEL")
        parsed = None if text is None else parse_level(text)
        return DEFAULT_LEVEL if parsed is None else parsed

    def resolve_stream(self) -> Any:
        """Return the stream to write to, looked up now so redirection is honoured."""
        if self.stream is not None and not isinstance(self.stream, str):
            return self.stream
        name = self.stream if isinstance(self.stream, str) else _env_text("PACE_STREAM")
        return sys.stderr if name == "stderr" else sys.stdout

    def resolve_color_mode(self) -> str:
        """Return ``auto``, ``always`` or ``never``."""
        return _choice(self.color, "PACE_COLOR")

    def resolve_bar_mode(self) -> str:
        """Return whether to draw an in-place bar: ``auto`` (on a terminal), ``always`` or ``never``."""
        return _choice(self.bar, "PACE_BAR")

    def resolve_min_interval(self) -> float:
        """Return the minimum seconds between progress lines of one scope."""
        if self.min_interval is not None:
            return self.min_interval
        value = _env_seconds("PACE_MIN_INTERVAL")
        return DEFAULT_MIN_INTERVAL if value is None else value

    def resolve_heartbeat(self) -> float:
        """Return the seconds after which a progress line is written whatever the progress."""
        if self.heartbeat is not None:
            return self.heartbeat
        value = _env_seconds("PACE_HEARTBEAT")
        return DEFAULT_HEARTBEAT if value is None else value

    def progress_enabled(self) -> bool:
        """Return whether this process writes progress lines (rank 0, or every rank if allowed)."""
        if self.all_ranks is not None:
            allowed = self.all_ranks
        else:
            allowed = _env_text("PACE_ALL_RANKS") in _TRUE_WORDS
        return allowed or current_rank() == 0
