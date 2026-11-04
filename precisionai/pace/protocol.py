# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Progress records and the ``[pace ...]`` line trailer that carries them.

This module is the contract between whatever writes progress (the logger, the
framework adapters) and whatever reads it back (the estimator, a scheduler). The
format is specified in ``docs/design.md``; a change here is a change to that
specification. The trailer has no version number, so the format may only grow:
a reader ignores keys and words it does not know.
"""

import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import TypeVar
from urllib.parse import unquote_to_bytes

MARKER = "pace"

MAX_LINE_BYTES = 4096
# Largest count, total, elapsed time or weight a trailer may carry.
MAX_NUMBER = 1e15
MAX_SEQ = 10**12
MAX_FIELDS = 32
# Raw (still percent-encoded) length. 768 holds a 64-character label or unit of
# 4-byte characters, each written as 12 escape characters.
MAX_VALUE_CHARS = 768
MAX_LABEL_CHARS = 64
MAX_UNIT_CHARS = 64
MAX_PHASES = 32
# A full plan: 32 phases, each a 64-character scope id, ":" and the widest weight the
# writer produces ("999999999999999.999999", 22 characters), joined by 31 commas.
MAX_PHASES_CHARS = 32 * (64 + 1 + 22) + 31

RUN_PATTERN = re.compile(r"[a-z0-9]{1,16}")

SCOPE_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,64}")

# The terminal escape sequences a reader removes before it looks for a trailer. The logger
# strips the same ones from user messages, so a forged marker cannot hide inside one.
ANSI_PATTERN = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")

# Characters that may appear in a trailer value without percent-encoding.
_SAFE_VALUE_CHARS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-:,")


class Event(str, Enum):
    """What a progress record reports about its scope."""

    START = "start"
    UPDATE = "upd"
    END = "end"
    FAIL = "fail"
    PLAN = "plan"
    STEP = "step"


@dataclass(frozen=True, slots=True)
class Record:
    """One progress record, as carried by one trailer.

    Parameters
    ----------
    scope : str
        Scope id, matching ``SCOPE_PATTERN``. Unique within one process run.
    event : Event
        What happened to the scope. ``STEP`` is a start and an end at once.
    seq : int or None
        Sequence number within the scope, on an update (from 2) and a plan (from 1). A
        start comes first in its run and an end, fail or step last, so they need none.
    elapsed : float
        Seconds since the scope started, from a monotonic clock.
    completed : float or None
        Amount done. Present on ``start``, ``upd``, ``end`` and ``step`` (1 on a step
        written without counts).
    total : float or None
        Total amount, or ``None`` when unknown (written as ``?``).
    unit : str or None
        Unit of ``completed`` and ``total``. Written on ``start``, and on a ``step`` that
        carries its counts (a child scope that ended before its first line was due).
    parent : str or None
        Parent scope id for a nested scope.
    root : bool
        Whether this scope stands for the whole job.
    resumed : bool
        On ``start``, whether the scope resumes earlier work.
    phases : tuple of (str, float)
        On ``plan``, the remaining phases (as scope ids) and their weights, in order.
    label : str or None
        Short human label, on ``start`` and ``step``.
    run : str or None
        Id of the process run that wrote the record, on every record. Tells a restarted
        scope apart from a line read twice.
    """

    scope: str
    event: Event
    seq: int | None = None
    elapsed: float = 0.0
    completed: float | None = None
    total: float | None = None
    unit: str | None = None
    parent: str | None = None
    root: bool = False
    resumed: bool = False
    phases: tuple[tuple[str, float], ...] = ()
    label: str | None = None
    run: str | None = None


def encode_value(value: str) -> str:
    """Percent-encode ``value`` for use in a trailer.

    Parameters
    ----------
    value : str
        Raw value. Any character outside the safe alphabet is written as the
        ``%XX`` escapes of its UTF-8 bytes.

    Returns
    -------
    str
        The encoded value, plain ASCII.

    Raises
    ------
    ValueError
        If ``value`` is empty.
    """
    if not value:
        raise ValueError("trailer values must not be empty")
    out = []
    for char in value:
        if char in _SAFE_VALUE_CHARS:
            out.append(char)
        else:
            out.extend(f"%{byte:02X}" for byte in char.encode("utf-8"))
    return "".join(out)


def format_number(value: float) -> str:
    """Return ``value`` as a trailer number: no exponent, no trailing zeros.

    Parameters
    ----------
    value : float
        A finite, non-negative number no larger than ``MAX_NUMBER``.

    Returns
    -------
    str
        ``"1840"`` for an integral value, otherwise up to six decimal places.

    Raises
    ------
    ValueError
        If ``value`` is negative, infinite, NaN or above ``MAX_NUMBER``.
    """
    if not math.isfinite(value) or value < 0 or value > MAX_NUMBER:
        raise ValueError(f"trailer numbers must be finite, non-negative and at most {MAX_NUMBER:g}, got {value!r}")
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def format_trailer(record: Record) -> str:
    """Return the ``[pace ...]`` trailer for ``record``.

    The scope id comes first, followed by ``<`` and the parent's id for a nested scope,
    then the event as a bare word (none for an update), the fields in the order
    ``c t u e q ph n r``, and the flags ``root`` and ``res``. A field a reader can fill
    in is left out: ``c`` and ``e`` when zero, and an end's ``t`` when it equals ``c``.

    Parameters
    ----------
    record : Record
        The record to write.

    Returns
    -------
    str
        The trailer, plain ASCII, with no surrounding whitespace.

    Raises
    ------
    ValueError
        If the scope, parent or run id is invalid or missing, an update or plan has
        no ``seq`` of at least 1, or a number is negative or not finite. Callers that must never
        raise (the logger) are expected to sanitise their input first.
    """
    _validate_ids(record)
    head = record.scope if record.parent is None else f"{record.scope}<{record.parent}"
    items = [head]
    if record.event is not Event.UPDATE:
        items.append(record.event.value)
    items.extend(_count_fields(record))
    elapsed = format_number(round(record.elapsed, 1))
    if elapsed != "0":
        items.append(f"e={elapsed}")
    items.extend(_tail_fields(record))
    items.append(f"r={record.run}")
    if record.root:
        items.append("root")
    if record.resumed and record.event is Event.START:
        items.append("res")
    return f"[{MARKER} {' '.join(items)}]"


def check_id(pattern: re.Pattern[str], value: str, what: str) -> str:
    """Return ``value`` if it matches ``pattern`` entirely, otherwise raise ``ValueError`` naming ``what``."""
    if not pattern.fullmatch(value):
        raise ValueError(f"invalid {what} {value!r}")
    return value


def _validate_ids(record: Record) -> None:
    check_id(SCOPE_PATTERN, record.scope, "scope id")
    if record.parent is not None:
        check_id(SCOPE_PATTERN, record.parent, "parent scope id")
    check_id(RUN_PATTERN, record.run or "", "run id")
    if (record.seq or 0) < _FIRST_SEQ.get(record.event, 0):
        raise ValueError(f"{record.event.value} needs a seq of at least {_FIRST_SEQ[record.event]}, got {record.seq}")
    if len(record.phases) > MAX_PHASES:
        raise ValueError(f"at most {MAX_PHASES} phases, got {len(record.phases)}")
    for name, _ in record.phases:
        check_id(SCOPE_PATTERN, name, "phase scope id")


def _count_fields(record: Record) -> list[str]:
    event = record.event
    # A fail carries the counts it reached when it has them, so a reader still sees how far it got.
    # A step without a unit is the bare one-off step of one unit; with one, it carries its counts like an end.
    if (
        event is Event.PLAN
        or (event is Event.STEP and record.unit is None)
        or (event is Event.FAIL and record.completed is None)
    ):
        return []
    completed = format_number(record.completed or 0.0)
    total = _format_total(record.total)
    fields = [] if completed == "0" else [f"c={completed}"]
    if event not in (Event.END, Event.STEP) or total != completed:
        fields.append(f"t={total}")
    if record.unit is not None and event in (Event.START, Event.STEP):
        fields.append(f"u={encode_value(_printable(record.unit, MAX_UNIT_CHARS))}")
    return fields


def _format_total(total: float | None) -> str:
    # A total too small to show at six decimal places would read as t=0, which no reader
    # accepts (a total must be positive), so it is written as unknown instead.
    text = "?" if total is None else format_number(total)
    return "?" if text == "0" else text


def _tail_fields(record: Record) -> list[str]:
    # Only the keys this event may carry (docs/design.md): q on update and plan, ph on plan, n on start and step.
    fields = []
    if record.event in (Event.UPDATE, Event.PLAN):
        fields.append(f"q={record.seq}")
    if record.event is Event.PLAN and record.phases:
        fields.append("ph=" + ",".join(f"{name}:{format_number(weight)}" for name, weight in record.phases))
    if record.label is not None and record.event in (Event.START, Event.STEP):
        fields.append(f"n={encode_value(_printable(record.label, MAX_LABEL_CHARS))}")
    return fields


def _printable(text: str, limit: int) -> str:
    cleaned = "".join(" " if _is_control(char) else char for char in text[:limit])
    return cleaned if cleaned.strip() else "-"


def _is_control(char: str) -> bool:
    code = ord(char)
    return code < 0x20 or 0x7F <= code <= 0x9F


_T = TypeVar("_T")

_KEY_PATTERN = re.compile(r"[a-z0-9]+")
_VALUE_PATTERN = re.compile(r"(?:[A-Za-z0-9._:,-]|%[0-9A-Fa-f]{2})+")
_NUMBER_PATTERN = re.compile(r"[0-9]+(?:\.[0-9]+)?")
_INTEGER_PATTERN = re.compile(r"[0-9]+")
_BAD_ESCAPE_PATTERN = re.compile(r"%(?![0-9A-Fa-f]{2})")
# The lowest q an event may carry: a start counts as 1, so updates begin at 2.
_FIRST_SEQ = {Event.UPDATE: 2, Event.PLAN: 1}
# The event is a bare word, and an update has none.
_KINDS = {event.value: event for event in Event if event is not Event.UPDATE}
# The keys that only some events may carry.
_EVENT_KEYS: dict[str, tuple[Event, ...]] = {
    "c": (Event.START, Event.UPDATE, Event.END, Event.FAIL, Event.STEP),
    "t": (Event.START, Event.UPDATE, Event.END, Event.FAIL, Event.STEP),
    "u": (Event.START, Event.STEP),
    "n": (Event.START, Event.STEP),
    "q": (Event.UPDATE, Event.PLAN),
    "ph": (Event.PLAN,),
}


def decode_value(value: str) -> str:
    """Reverse :func:`encode_value`.

    Parameters
    ----------
    value : str
        An encoded trailer value. Each ``%XX`` escape stands for one byte of
        UTF-8; every other character stands for itself.

    Returns
    -------
    str
        The decoded text.

    Raises
    ------
    ValueError
        If a ``%`` is not followed by two hexadecimal digits, or the decoded
        bytes are not valid UTF-8.
    """
    if _BAD_ESCAPE_PATTERN.search(value):
        raise ValueError(f"malformed percent escape in {value!r}")
    try:
        return unquote_to_bytes(value).decode("utf-8")
    except UnicodeError as err:
        raise ValueError(f"value {value!r} is not valid UTF-8") from err


def parse_line(line: str) -> Record | None:
    """Return the record carried by the trailer at the end of ``line``, if any.

    A consumer must treat every log line as untrusted input, so this never
    raises: a line without a trailer, or with one that breaks any rule in
    ``docs/design.md``, yields ``None``. Keys and words it does not know are
    ignored, so a later release that adds some stays readable.

    Parameters
    ----------
    line : str
        One log line, with or without its line ending. ANSI escape sequences
        and trailing whitespace are ignored.

    Returns
    -------
    Record or None
        The record, or ``None`` when the line carries no valid trailer.
    """
    if not isinstance(line, str):
        return None
    line = line.rstrip("\r\n")
    if len(line) > MAX_LINE_BYTES:
        return None
    # U+FFFD stands for one undecodable source byte (see LineSplitter), so count it as one.
    if len(line.encode("utf-8", errors="surrogatepass")) - 2 * line.count("�") > MAX_LINE_BYTES:
        return None
    tokens = _trailer_tokens(line)
    if tokens is None:
        return None
    try:
        return _build_record(tokens)
    except ValueError:
        return None


def _trailer_tokens(line: str) -> list[str] | None:
    text = ANSI_PATTERN.sub("", line).rstrip()
    if not text.endswith("]"):
        return None
    start = text.rfind(f"[{MARKER} ")
    if start < 0 or (start > 0 and text[start - 1] != " "):
        return None
    return text[start + len(MARKER) + 2 : -1].split(" ")


def _build_record(tokens: list[str]) -> Record:
    if not tokens or len(tokens) > MAX_FIELDS:
        raise ValueError("no scope, or too many items")
    scope, nested, parent = tokens[0].partition("<")
    words = [token for token in tokens[1:] if "=" not in token]
    fields = _parse_fields([token for token in tokens[1:] if "=" in token])
    if len(set(words)) != len(words) or not all(_KEY_PATTERN.fullmatch(word) for word in words):
        raise ValueError("malformed or repeated word")
    kinds = [_KINDS[word] for word in words if word in _KINDS]
    if len(kinds) > 1:
        raise ValueError("more than one event")
    event = kinds[0] if kinds else Event.UPDATE
    _check_fields(event, fields, words)
    completed, total = _counts(event, fields)
    seq = _optional(fields, "q", _parse_seq)
    if seq is not None and seq < _FIRST_SEQ[event]:
        raise ValueError(f"q on {event.value} starts at {_FIRST_SEQ[event]}")
    return Record(
        scope=_parse_scope(scope),
        event=event,
        seq=seq,
        elapsed=_parse_number(fields.get("e", "0")),
        completed=completed,
        total=total,
        unit=_optional(fields, "u", _parse_unit),
        parent=_parse_scope(parent) if nested else None,
        root="root" in words,
        resumed="res" in words,
        phases=_optional(fields, "ph", _parse_phases) or (),
        label=_optional(fields, "n", _parse_label),
        run=_parse_run(fields["r"]),
    )


def _check_fields(event: Event, fields: dict[str, str], words: list[str]) -> None:
    """Reject a missing required key, and a key or flag that belongs to another event."""
    required = {Event.START: ("t", "r"), Event.UPDATE: ("t", "q", "r"), Event.PLAN: ("q", "r")}.get(event, ("r",))
    if any(key not in fields for key in required):
        raise ValueError(f"{event.value} requires {', '.join(required)}")
    for key, events in _EVENT_KEYS.items():
        if key in fields and event not in events:
            raise ValueError(f"{key} does not belong on {event.value}")
    if "res" in words and event is not Event.START:
        raise ValueError("res belongs on start")
    if event is Event.FAIL and "c" in fields and "t" not in fields:
        raise ValueError("a fail with a count carries its total")


def _counts(event: Event, fields: dict[str, str]) -> tuple[float | None, float | None]:
    """Return the counts a record stands for, filling in what the writer leaves out."""
    if event is Event.STEP and "c" not in fields and "t" not in fields:
        return 1.0, 1.0
    if event is Event.PLAN or (event is Event.FAIL and "t" not in fields):
        return None, None
    completed = _parse_number(fields.get("c", "0"))
    if "t" in fields:
        return completed, _parse_total(fields["t"])
    # Only an end or a step may leave out its total, which then equals its count.
    if completed <= 0:
        raise ValueError("an end without a total needs a positive count")
    return completed, completed


def _parse_fields(tokens: list[str]) -> dict[str, str]:
    fields: dict[str, str] = {}
    for token in tokens:
        key, sep, value = token.partition("=")
        if not sep or not _KEY_PATTERN.fullmatch(key) or not _valid_value(key, value):
            raise ValueError(f"malformed field {token!r}")
        if key in fields:
            raise ValueError(f"duplicate key {key!r}")
        fields[key] = value
    return fields


def _valid_value(key: str, value: str) -> bool:
    if value == "?":
        return key == "t"
    limit = MAX_PHASES_CHARS if key == "ph" else MAX_VALUE_CHARS
    return len(value) <= limit and _VALUE_PATTERN.fullmatch(value) is not None


def _optional(fields: dict[str, str], key: str, parse: Callable[[str], _T]) -> _T | None:
    return parse(fields[key]) if key in fields else None


def _parse_seq(value: str) -> int:
    if not _INTEGER_PATTERN.fullmatch(value):
        raise ValueError(f"invalid sequence number {value!r}")
    seq = int(value)
    if not 1 <= seq <= MAX_SEQ:
        raise ValueError(f"sequence number out of range: {value!r}")
    return seq


def _parse_number(value: str) -> float:
    if not _NUMBER_PATTERN.fullmatch(value):
        raise ValueError(f"invalid number {value!r}")
    number = float(value)
    if number > MAX_NUMBER:
        raise ValueError(f"number out of range: {value!r}")
    return number


def _parse_total(value: str) -> float | None:
    if value == "?":
        return None
    total = _parse_number(value)
    if total <= 0:
        raise ValueError("total must be positive")
    return total


def _parse_phases(value: str) -> tuple[tuple[str, float], ...]:
    pairs = value.split(",")
    if len(pairs) > MAX_PHASES:
        raise ValueError("too many phases")
    phases = []
    for pair in pairs:
        name, sep, weight = pair.partition(":")
        if not sep:
            raise ValueError(f"malformed phase {pair!r}")
        phases.append((_parse_scope(name), _parse_number(weight)))
    return tuple(phases)


def _parse_text(value: str, limit: int, what: str) -> str:
    text = decode_value(value)
    if len(text) > limit:
        raise ValueError(f"{what} too long")
    if any(_is_control(char) for char in text):
        raise ValueError("control character in decoded value")
    return text


_parse_scope = partial(check_id, SCOPE_PATTERN, what="scope id")
_parse_run = partial(check_id, RUN_PATTERN, what="run id")
_parse_unit = partial(_parse_text, limit=MAX_UNIT_CHARS, what="unit")
_parse_label = partial(_parse_text, limit=MAX_LABEL_CHARS, what="label")


class LineSplitter:
    r"""Cut a byte stream read in arbitrary chunks into text lines.

    Memory stays bounded however hostile the stream: a line longer than
    ``MAX_LINE_BYTES`` cannot hold a valid trailer, so it is dropped rather
    than buffered, and the result does not depend on where the chunks fall.
    ``\n`` and ``\r\n`` both end a line.
    """

    def __init__(self) -> None:
        self._tail = bytearray()
        self._skipping = False

    def feed(self, chunk: bytes) -> list[str]:
        r"""Add ``chunk`` and return the lines it completes.

        Parameters
        ----------
        chunk : bytes
            The next bytes of the stream, split anywhere (even inside a
            multi-byte character or between ``\r`` and ``\n``).

        Returns
        -------
        list of str
            Each completed line, decoded as UTF-8 with undecodable bytes
            replaced, without its line ending. Over-long lines are omitted.
        """
        lines: list[str] = []
        start = 0
        while True:
            end = chunk.find(b"\n", start)
            if end < 0:
                break
            self._complete(chunk, start, end, lines)
            start = end + 1
        self._keep(chunk[start:])
        return lines

    def reset(self) -> None:
        """Drop the partial line, for when the consumer detects a gap in the log."""
        self._tail.clear()
        self._skipping = False

    def flush(self) -> list[str]:
        """Return the partial line left at the end of the stream, if any.

        Returns
        -------
        list of str
            The pending text as one line, or an empty list when there is none.
        """
        lines: list[str] = []
        if not self._skipping:
            self._emit(bytes(self._tail), lines, drop_empty=True)
        self.reset()
        return lines

    def _complete(self, chunk: bytes, start: int, end: int, lines: list[str]) -> None:
        if self._skipping:
            self._skipping = False
            return
        # Decide on the size before copying, so one huge line in a huge chunk is never materialised.
        if self._too_long(end - start):
            self._tail.clear()
            return
        data = bytes(self._tail) + chunk[start:end]
        self._tail.clear()
        self._emit(data, lines)

    def _keep(self, rest: bytes) -> None:
        if self._skipping:
            return
        if self._too_long(len(rest)):
            self._tail.clear()
            self._skipping = True
            return
        self._tail += rest

    def _too_long(self, extra: int) -> bool:
        # One byte of slack: a line of exactly MAX_LINE_BYTES waits here with its "\r".
        return len(self._tail) + extra > MAX_LINE_BYTES + 1

    @staticmethod
    def _emit(data: bytes, lines: list[str], *, drop_empty: bool = False) -> None:
        if data.endswith(b"\r"):
            data = data[:-1]
        if len(data) > MAX_LINE_BYTES or (drop_empty and not data):
            return
        lines.append(data.decode("utf-8", errors="replace"))
