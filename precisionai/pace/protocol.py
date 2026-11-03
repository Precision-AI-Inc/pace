# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Progress records and the ``[pace:1 ...]`` line trailer that carries them.

This module is the contract between whatever writes progress (the logger, the
framework adapters) and whatever reads it back (the estimator, a scheduler). The
format is specified in ``docs/design.md``; a change here is a change to that
specification.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum

MARKER = "pace"
VERSION = 1

MAX_LINE_BYTES = 4096
MAX_FIELDS = 32
MAX_VALUE_CHARS = 256
MAX_LABEL_CHARS = 64

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


@dataclass(frozen=True)
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
        A finite, non-negative number.

    Returns
    -------
    str
        ``"1840"`` for an integral value, otherwise up to six decimal places.

    Raises
    ------
    ValueError
        If ``value`` is negative, infinite or NaN.
    """
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"trailer numbers must be finite and non-negative, got {value!r}")
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
    for name, _ in record.phases:
        if not SCOPE_PATTERN.fullmatch(name):
            raise ValueError(f"invalid phase scope id {name!r}")


def _count_fields(record: Record) -> list[str]:
    fields = []
    if record.event in (Event.START, Event.UPDATE, Event.END):
        fields.append(f"c={format_number(record.completed or 0.0)}")
        fields.append(f"t={'?' if record.total is None else format_number(record.total)}")
    if record.unit is not None:
        fields.append(f"u={encode_value(record.unit)}")
    return fields


def _tail_fields(record: Record) -> list[str]:
    fields = []
    if record.resumed:
        fields.append("res=1")
    if record.phases:
        fields.append("ph=" + ",".join(f"{name}:{format_number(weight)}" for name, weight in record.phases))
    if record.label is not None:
        fields.append(f"n={encode_value(record.label[:MAX_LABEL_CHARS])}")
    return fields
