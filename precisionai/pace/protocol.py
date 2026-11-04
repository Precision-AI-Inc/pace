# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Progress records and the ``[pace:1 ...]`` line trailer that carries them.

This module is the contract between whatever writes progress (the logger, the
framework adapters) and whatever reads it back (the estimator, a scheduler). The
format is specified in ``docs/design.md``; a change here is a change to that
specification.
"""

import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TypeVar
from urllib.parse import unquote_to_bytes

MARKER = "pace"
VERSION = 1

MAX_LINE_BYTES = 4096
# Largest count, total, elapsed time or weight a trailer may carry.
MAX_NUMBER = 1e15
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

# Characters that may appear in a trailer value without percent-encoding.
_SAFE_VALUE_CHARS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-:,")


class Event(str, Enum):
    """What a progress record reports about its scope."""

    START = "start"
    UPDATE = "upd"
    END = "end"
    FAIL = "fail"
    PLAN = "plan"


@dataclass(frozen=True, slots=True)
class Record:
    """One progress record, as carried by one trailer.

    Parameters
    ----------
    scope : str
        Scope id, matching ``SCOPE_PATTERN``. Unique within one process run.
    event : Event
        What happened to the scope.
    seq : int
        Sequence number within the scope, starting at 1.
    elapsed : float
        Seconds since the scope started, from a monotonic clock.
    completed : float or None
        Amount done. Present on ``start``, ``upd`` and ``end``.
    total : float or None
        Total amount, or ``None`` when unknown (written as ``?``).
    unit : str or None
        Unit of ``completed`` and ``total``. Written on ``start``.
    parent : str or None
        Parent scope id for a nested scope.
    root : bool
        Whether this scope stands for the whole job.
    resumed : bool
        On ``start``, whether the scope resumes earlier work.
    phases : tuple of (str, float)
        On ``plan``, the remaining phases (as scope ids) and their weights, in order.
    label : str or None
        Short human label, on ``start`` only.
    run : str or None
        Id of the process run that wrote the record, on ``start`` only. Tells a
        restarted scope apart from the same ``start`` line read twice.
    """

    scope: str
    event: Event
    seq: int
    elapsed: float
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
    """Return the ``[pace:1 ...]`` trailer for ``record``.

    Fields are written in a fixed order: ``s p root k c t u e q res ph n``. Optional
    fields are omitted when unset.

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
        If the scope or parent id is invalid, ``seq`` is below 1, or a number is
        negative or not finite. Callers that must never raise (the logger) are
        expected to sanitise their input first.
    """
    _validate_ids(record)
    fields = [f"s={record.scope}"]
    if record.parent is not None:
        fields.append(f"p={record.parent}")
    if record.root:
        fields.append("root=1")
    fields.append(f"k={record.event.value}")
    fields.extend(_count_fields(record))
    fields.append(f"e={format_number(round(record.elapsed, 1))}")
    fields.append(f"q={record.seq}")
    fields.extend(_tail_fields(record))
    return f"[{MARKER}:{VERSION} {' '.join(fields)}]"


def _validate_ids(record: Record) -> None:
    if not SCOPE_PATTERN.fullmatch(record.scope):
        raise ValueError(f"invalid scope id {record.scope!r}")
    if record.parent is not None and not SCOPE_PATTERN.fullmatch(record.parent):
        raise ValueError(f"invalid parent scope id {record.parent!r}")
    if record.seq < 1:
        raise ValueError(f"seq must be at least 1, got {record.seq}")
    if record.run is not None and not RUN_PATTERN.fullmatch(record.run):
        raise ValueError(f"invalid run id {record.run!r}")
    if len(record.phases) > MAX_PHASES:
        raise ValueError(f"at most {MAX_PHASES} phases, got {len(record.phases)}")
    for name, _ in record.phases:
        if not SCOPE_PATTERN.fullmatch(name):
            raise ValueError(f"invalid phase scope id {name!r}")


def _count_fields(record: Record) -> list[str]:
    fields = []
    # A fail carries the counts it reached when it has them, so a reader still sees how far it got.
    if record.event in (Event.START, Event.UPDATE, Event.END) or (
        record.event is Event.FAIL and record.completed is not None
    ):
        fields.append(f"c={format_number(record.completed or 0.0)}")
        fields.append(f"t={_format_total(record.total)}")
    if record.unit is not None and record.event is Event.START:
        fields.append(f"u={encode_value(_printable(record.unit, MAX_UNIT_CHARS))}")
    return fields


def _format_total(total: float | None) -> str:
    # A total too small to show at six decimal places would read as t=0, which no reader
    # accepts (a total must be positive), so it is written as unknown instead.
    text = "?" if total is None else format_number(total)
    return "?" if text == "0" else text


def _tail_fields(record: Record) -> list[str]:
    # Only the keys this event may carry (docs/design.md): res, run and n on start, ph on plan.
    start = record.event is Event.START
    fields = []
    if record.resumed and start:
        fields.append("res=1")
    if record.run is not None and start:
        fields.append(f"run={record.run}")
    if record.phases and record.event is Event.PLAN:
        fields.append("ph=" + ",".join(f"{name}:{format_number(weight)}" for name, weight in record.phases))
    if record.label is not None and start:
        fields.append(f"n={encode_value(_printable(record.label, MAX_LABEL_CHARS))}")
    return fields


def _printable(text: str, limit: int) -> str:
    cleaned = "".join(" " if _is_control(char) else char for char in text[:limit])
    return cleaned if cleaned.strip() else "-"


def _is_control(char: str) -> bool:
    code = ord(char)
    return code < 0x20 or 0x7F <= code <= 0x9F


MAX_SEQ = 10**12

_T = TypeVar("_T")

_ANSI_PATTERN = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")
_KEY_PATTERN = re.compile(r"[a-z0-9]+")
_VALUE_PATTERN = re.compile(r"(?:[A-Za-z0-9._:,-]|%[0-9A-Fa-f]{2})+")
_NUMBER_PATTERN = re.compile(r"[0-9]+(?:\.[0-9]+)?")
_INTEGER_PATTERN = re.compile(r"[0-9]+")
_BAD_ESCAPE_PATTERN = re.compile(r"%(?![0-9A-Fa-f]{2})")
_COUNTED_EVENTS = (Event.START, Event.UPDATE, Event.END)


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
    ``docs/design.md``, yields ``None``. Keys this version does not know are
    ignored, so a later minor version stays readable.

    Parameters
    ----------
    line : str
        One log line, with or without its line ending. ANSI escape sequences
        and trailing whitespace are ignored.

    Returns
    -------
    Record or None
        The record, or ``None`` when the line carries no valid version-1 trailer.
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
        return _build_record(_parse_fields(tokens))
    except ValueError:
        return None


def _trailer_tokens(line: str) -> list[str] | None:
    text = _ANSI_PATTERN.sub("", line).rstrip()
    if not text.endswith("]"):
        return None
    start = text.rfind(f"[{MARKER}:")
    if start < 0 or (start > 0 and text[start - 1] != " "):
        return None
    tokens = text[start + 1 : -1].split(" ")
    if tokens[0] != f"{MARKER}:{VERSION}":
        return None
    return tokens[1:]


def _parse_fields(tokens: list[str]) -> dict[str, str]:
    if len(tokens) > MAX_FIELDS:
        raise ValueError("too many fields")
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


def _build_record(fields: dict[str, str]) -> Record:
    for key in ("s", "k", "q", "e"):
        if key not in fields:
            raise ValueError(f"missing required key {key!r}")
    event = Event(fields["k"])
    if event in _COUNTED_EVENTS and ("c" not in fields or "t" not in fields):
        raise ValueError(f"{event.value} requires c and t")
    _check_event_fields(event, fields)
    return Record(
        scope=_parse_scope(fields["s"]),
        event=event,
        seq=_parse_seq(fields["q"]),
        elapsed=_parse_number(fields["e"]),
        completed=_optional(fields, "c", _parse_number),
        total=_optional(fields, "t", _parse_total),
        unit=_optional(fields, "u", _parse_unit),
        parent=_optional(fields, "p", _parse_scope),
        root=_parse_flag(fields, "root"),
        resumed=_parse_flag(fields, "res"),
        phases=_optional(fields, "ph", _parse_phases) or (),
        label=_optional(fields, "n", _parse_label),
        run=_optional(fields, "run", _parse_run),
    )


def _check_event_fields(event: Event, fields: dict[str, str]) -> None:
    """Reject keys that belong to another event: run, n, res and u on start, ph on plan."""
    if event is not Event.START and any(key in fields for key in ("run", "n", "res", "u")):
        raise ValueError(f"run, n, res and u belong on start, not {event.value}")
    if event is Event.FAIL and ("c" in fields) != ("t" in fields):
        raise ValueError("a fail carries both c and t, or neither")
    if event is not Event.PLAN and "ph" in fields:
        raise ValueError(f"ph belongs on plan, not {event.value}")
    if event is Event.PLAN and any(key in fields for key in ("c", "t")):
        raise ValueError("a plan carries no counts")


def _optional(fields: dict[str, str], key: str, parse: Callable[[str], _T]) -> _T | None:
    return parse(fields[key]) if key in fields else None


def _parse_scope(value: str) -> str:
    if not SCOPE_PATTERN.fullmatch(value):
        raise ValueError(f"invalid scope id {value!r}")
    return value


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


def _parse_flag(fields: dict[str, str], key: str) -> bool:
    if key not in fields:
        return False
    if fields[key] != "1":
        raise ValueError(f"{key} must be 1, got {fields[key]!r}")
    return True


def _parse_unit(value: str) -> str:
    unit = decode_value(value)
    if len(unit) > MAX_UNIT_CHARS:
        raise ValueError("unit too long")
    _reject_control(unit)
    return unit


def _parse_label(value: str) -> str:
    label = decode_value(value)
    if len(label) > MAX_LABEL_CHARS:
        raise ValueError("label too long")
    _reject_control(label)
    return label


def _parse_run(value: str) -> str:
    if not RUN_PATTERN.fullmatch(value):
        raise ValueError(f"invalid run id {value!r}")
    return value


def _reject_control(text: str) -> None:
    if any(_is_control(char) for char in text):
        raise ValueError("control character in decoded value")


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
        if len(self._tail) + (end - start) > MAX_LINE_BYTES + 1:
            self._tail.clear()
            return
        data = bytes(self._tail) + chunk[start:end]
        self._tail.clear()
        self._emit(data, lines)

    def _keep(self, rest: bytes) -> None:
        if self._skipping:
            return
        if len(self._tail) + len(rest) > MAX_LINE_BYTES + 1:
            self._tail.clear()
            self._skipping = True
            return
        self._tail += rest
        # One byte of slack: a line of exactly MAX_LINE_BYTES waits here with its "\r".
        if len(self._tail) > MAX_LINE_BYTES + 1:
            self._tail.clear()
            self._skipping = True

    @staticmethod
    def _emit(data: bytes, lines: list[str], *, drop_empty: bool = False) -> None:
        if data.endswith(b"\r"):
            data = data[:-1]
        if len(data) > MAX_LINE_BYTES or (drop_empty and not data):
            return
        lines.append(data.decode("utf-8", errors="replace"))
