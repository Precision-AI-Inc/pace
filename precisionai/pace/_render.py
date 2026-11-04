# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Turn log messages and progress records into the readable part of an output line.

Nothing here writes; it only builds strings. The text is for people and may change
between releases. Only the ``[pace:1 ...]`` trailer that follows it is a contract.
"""

import codecs
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass

from precisionai.pace.forecast import Forecast, ForecastState
from precisionai.pace.protocol import ANSI_PATTERN, Event, Record

PROGRESS_LABEL = "PROGRESS"
LABEL_WIDTH = 8
BAR_WIDTH = 20
MAX_DESCRIPTION_CHARS = 80
SEPARATOR = "  "


@dataclass(frozen=True, slots=True)
class Charset:
    """The glyphs used in the readable part of a line."""

    dot: str
    ellipsis: str
    bar_full: str
    bar_empty: str


UNICODE = Charset(dot="·", ellipsis="…", bar_full="━", bar_empty="─")
ASCII = Charset(dot="|", ellipsis="...", bar_full="#", bar_empty="-")

_COLOR_CODES: dict[str, str] = {
    "DEBUG": "2",
    "SUCCESS": "32",
    "WARNING": "33",
    "ERROR": "31",
    "CRITICAL": "1;31",
    PROGRESS_LABEL: "36",
}
_ESCAPE = "\x1b["
_WHITESPACE = re.compile(r"\s+")
_FORGED_MARKER = re.compile(r"\[(?=pace:)")


def is_utf8(encoding: str) -> bool:
    """Return whether ``encoding`` is UTF-8, so Unicode glyphs are safe to print."""
    try:
        return codecs.lookup(encoding).name == "utf-8"
    except LookupError:
        return False


def charset_for(encoding: str) -> Charset:
    """Return the Unicode glyphs for a UTF-8 stream and the ASCII ones for any other."""
    return UNICODE if is_utf8(encoding) else ASCII


def encode_safely(text: str, encoding: str) -> str:
    """Return ``text`` with every character the encoding cannot represent replaced.

    Parameters
    ----------
    text : str
        Text about to be written.
    encoding : str
        The stream's encoding. An unknown encoding is treated as ASCII.

    Returns
    -------
    str
        Text the stream can write without raising ``UnicodeEncodeError``.
    """
    try:
        return text.encode(encoding, "replace").decode(encoding, "replace")
    except LookupError:
        return text.encode("ascii", "replace").decode("ascii")


def format_timestamp(wall: float) -> str:
    """Return ``wall`` (seconds since the epoch) as ISO-8601 UTC with a ``Z``, to the second."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(wall))


def colorize(label: str, enabled: bool) -> str:
    """Wrap a level label in its colour when ``enabled``; the trailer is never coloured."""
    code = _COLOR_CODES.get(label)
    if not enabled or code is None:
        return label
    return f"{_ESCAPE}{code}m{label}{_ESCAPE}0m"


def log_line(timestamp: str, label: str, message: str, color: bool) -> str:
    """Return a log line: timestamp, level padded to eight characters, message."""
    pad = " " * max(LABEL_WIDTH - len(label), 0)
    return f"{timestamp} {colorize(label, color)}{pad} {neutralize(message)}"


def neutralize(message: str) -> str:
    """Break up any ``[pace:`` in a user message so it cannot pass for a progress trailer.

    A reader removes terminal escape sequences before it looks for a trailer, so they
    are removed here first: ``[pa`` + ESC ``[31m`` + ``ce:`` must not survive as a
    marker. Any ESC left over is dropped too.
    """
    plain = ANSI_PATTERN.sub("", message).replace("\x1b", "")
    return _FORGED_MARKER.sub("[ ", plain)


def clean_text(text: str, limit: int = MAX_DESCRIPTION_CHARS) -> str:
    """Return ``text`` on one line (whitespace and control characters become one space), cut to ``limit``."""
    one_line = _WHITESPACE.sub(" ", "".join(" " if ord(char) < 32 else char for char in text)).strip()
    return neutralize(one_line)[:limit]


def format_duration(seconds: float) -> str:
    """Return a short duration such as ``45s``, ``21m57s``, ``3h05m`` or ``2d4h``."""
    total = max(int(seconds + 0.5), 0)
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    if days:
        return f"{days}d{hours}h"
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def format_count(value: float) -> str:
    """Return a count without a trailing ``.0``."""
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.4g}"


def format_rate(rate: float, unit: str) -> str:
    """Return a rate such as ``2.4 step/s``."""
    shown = f"{rate:.0f}" if rate >= 100 else f"{rate:.3g}"
    return f"{shown} {unit}/s"


def format_bar(fraction: float, charset: Charset) -> str:
    """Return a fixed-width bar, ``fraction`` of it filled."""
    filled = min(max(int(fraction * BAR_WIDTH), 0), BAR_WIDTH)
    return charset.bar_full * filled + charset.bar_empty * (BAR_WIDTH - filled)


def estimate_text(record: Record, forecast: Forecast, charset: Charset) -> str:
    """Describe the outcome of a finished scope, or its time remaining, or why there is no estimate."""
    if record.event is Event.END:
        return f"done in {format_duration(record.elapsed)}"
    if record.event is Event.FAIL:
        return f"failed after {format_duration(record.elapsed)}"
    remaining = forecast.remaining_seconds
    state = forecast.state
    if remaining is not None and state in (ForecastState.LIVE, ForecastState.STALE, ForecastState.BOOTSTRAP):
        note = {ForecastState.STALE: " (stale)", ForecastState.BOOTSTRAP: " (guess)"}.get(state, "")
        return f"~{format_duration(remaining)}{note}"
    if state is ForecastState.LEARNING:
        return f"estimating{charset.ellipsis}"
    return "no estimate"


def progress_text(
    record: Record,
    forecast: Forecast,
    description: str,
    unit: str,
    charset: Charset,
    bar: bool = False,
) -> str:
    """Return the readable part of a progress line, for example ``Train  36.8%  1840/5000 step · 2.4 step/s · ~21m57s``.

    Parameters
    ----------
    record : Record
        The record being written.
    forecast : Forecast
        The scope's estimate after ``record``.
    description : str
        What the scope is doing; the scope id is used when empty.
    unit : str
        Unit of the counts.
    charset : Charset
        Unicode or ASCII glyphs.
    bar : bool
        Whether to include a bar after the description (the in-place terminal line).

    Returns
    -------
    str
        One line of text with no timestamp, level or trailer.
    """
    completed = record.completed or 0.0
    total = record.total
    parts: list[str] = [clean_text(description) or record.scope]
    if total is not None:
        if bar:
            parts.append(format_bar(completed / total, charset))
        parts.append(f"{100 * completed / total:.1f}%")
    count = format_count(completed) if total is None else f"{format_count(completed)}/{format_count(total)}"
    details = [f"{count} {clean_text(unit, 32)}".rstrip()]
    if forecast.rate is not None:
        details.append(format_rate(forecast.rate, clean_text(unit, 32) or "it"))
    details.append(estimate_text(record, forecast, charset))
    parts.append(f" {charset.dot} ".join(details))
    return SEPARATOR.join(parts)


def plan_text(phases: Sequence[tuple[str, float]]) -> str:
    """Return the readable part of a ``plan`` line, for example ``Plan  prep 5%, train 90%``."""
    whole = sum(weight for _, weight in phases) or 1.0
    shares = ", ".join(f"{name} {round(100 * weight / whole)}%" for name, weight in phases)
    return f"Plan{SEPARATOR}{shares}"


def fit(text: str, width: int, charset: Charset | None = None) -> str:
    """Return ``text`` cut so it fits ``width`` columns, ending in an ellipsis when it was cut."""
    if len(text) <= width:
        return text
    mark = (charset or ASCII).ellipsis
    return (text[: max(width - len(mark), 0)] + mark)[:width]
