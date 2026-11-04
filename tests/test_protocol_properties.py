# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Property tests for the trailer writer, the parser, the line splitter and the estimator.

Log lines are untrusted input, so these feed the reading side arbitrary and almost-valid text,
and check that what the writer produces is always read back as written. The runs are derandomized:
the same examples on every machine, so a failure is reproducible rather than flaky.
"""

import math
from typing import Any

import pytest

from precisionai.pace import Event, ForecastState, JobForecast, LineSplitter, Record, format_trailer, parse_line
from precisionai.pace.protocol import MAX_LINE_BYTES, MAX_NUMBER, MAX_PHASES, MAX_SEQ, SCOPE_PATTERN

hypothesis = pytest.importorskip("hypothesis")
st: Any = pytest.importorskip("hypothesis.strategies")

PROFILE = hypothesis.settings(
    max_examples=200,
    deadline=None,
    derandomize=True,
    suppress_health_check=[hypothesis.HealthCheck.too_slow, hypothesis.HealthCheck.data_too_large],
)

scope_ids = st.from_regex(SCOPE_PATTERN, fullmatch=True)
# Multiples of a quarter are exact in binary, so they survive the writer's six decimals unchanged.
quarters = st.integers(0, 4_000_000).map(lambda n: n / 4)
positive_quarters = st.integers(1, 4_000_000).map(lambda n: n / 4)
tenths_safe = st.integers(0, 4_000_000).map(lambda n: n / 2)
printable_text = st.text(st.characters(blacklist_categories=("Cc", "Cs")), min_size=1, max_size=64).filter(
    lambda text: text.strip() != ""
)
run_ids = st.from_regex(r"[a-z0-9]{1,16}", fullmatch=True)


@st.composite
def records(draw: Any) -> Record:
    event = draw(st.sampled_from(list(Event)))
    common: dict[str, Any] = {
        "scope": draw(scope_ids),
        "event": event,
        "seq": draw(st.integers(1, MAX_SEQ)),
        "elapsed": draw(tenths_safe),
        "parent": draw(st.none() | scope_ids),
        "root": draw(st.booleans()),
    }
    if event is Event.PLAN:
        phases = draw(st.lists(st.tuples(scope_ids, quarters), max_size=MAX_PHASES))
        return Record(phases=tuple(phases), **common)
    if event is Event.FAIL and draw(st.booleans()):
        return Record(**common)
    common["completed"] = draw(quarters)
    common["total"] = draw(st.none() | positive_quarters)
    if event is Event.START:
        common["unit"] = draw(st.none() | printable_text)
        common["resumed"] = draw(st.booleans())
        common["label"] = draw(st.none() | printable_text)
        common["run"] = draw(st.none() | run_ids)
    return Record(**common)


@PROFILE
@hypothesis.given(records())
def test_a_written_trailer_is_read_back_as_written(record: Record) -> None:
    trailer = format_trailer(record)
    assert trailer.isascii()
    assert len(trailer.encode("ascii")) <= MAX_LINE_BYTES
    assert parse_line(trailer) == record
    assert parse_line(f"2026-01-01T00:00:00Z PROGRESS readable text  {trailer}\n") == record


# Fragments of the syntax, so that arbitrary text sometimes gets close to a trailer.
FRAGMENTS = [
    "[pace:1 ",
    "[pace:",
    "]",
    " ",
    "s=a ",
    "k=upd ",
    "c=1 ",
    "t=? ",
    "q=2 ",
    "e=1",
    "%",
    "%4",
    "=",
    "\x1b[",
    "\x1b]",
    "\x07",
    "\r",
    "\n",
    "\u00e9",
    "\ud800",
    "\U0001f600",
    "0",
    "-",
    "ph=a:1,b:2",
    "root=1",
    "n=%FF",
    "u=caf%C3%A9",
]
garbage = st.text() | st.lists(st.sampled_from(FRAGMENTS) | st.text(max_size=3), max_size=40).map("".join)


@PROFILE
@hypothesis.given(garbage)
def test_parse_line_never_raises_on_arbitrary_text(text: str) -> None:
    assert parse_line(text) is None or isinstance(parse_line(text), Record)


@PROFILE
@hypothesis.given(records(), st.integers(0, 4000), st.integers(0, 6), st.text(max_size=6))
def test_a_damaged_trailer_is_a_record_or_nothing_and_never_an_error(
    record: Record, at: int, cut: int, noise: str
) -> None:
    trailer = format_trailer(record)
    start = at % (len(trailer) + 1)
    found = parse_line(trailer[:start] + noise + trailer[start + cut :])
    if found is None:
        return
    assert SCOPE_PATTERN.fullmatch(found.scope)
    assert 1 <= found.seq <= MAX_SEQ
    for number in (found.elapsed, found.completed or 0.0, found.total or 0.0):
        assert math.isfinite(number)
        assert 0 <= number <= MAX_NUMBER
    assert found.total is None or found.total > 0
    format_trailer(found)


def reference_lines(data: bytes) -> list[str]:
    """What a stream of ``data`` holds, by the splitter's rules, computed without any chunking."""
    *complete, rest = data.split(b"\n")
    lines = []
    for line, final in [(line, False) for line in complete] + [(rest, True)]:
        raw = line[:-1] if line.endswith(b"\r") else line
        if len(raw) <= MAX_LINE_BYTES and not (final and not raw):
            lines.append(raw.decode("utf-8", errors="replace"))
    return lines


def line_bytes() -> Any:
    boundary = st.integers(MAX_LINE_BYTES - 3, MAX_LINE_BYTES + 3).flatmap(
        lambda size: st.sampled_from([b"a" * size, b"a" * (size - 1) + b"\r", b"a" * (size - 2) + b"\xe2\x82"])
    )
    return st.one_of(st.binary(max_size=40), st.just(b""), st.just(b"\r"), boundary)


def split_at(data: bytes, cuts: list[int]) -> list[bytes]:
    points = sorted({cut % (len(data) + 1) for cut in cuts})
    pieces, last = [], 0
    for point in points:
        pieces.append(data[last:point])
        last = point
    pieces.append(data[last:])
    return pieces


def run_splitter(chunks: list[bytes]) -> list[str]:
    splitter = LineSplitter()
    lines: list[str] = []
    for chunk in chunks:
        lines.extend(splitter.feed(chunk))
    lines.extend(splitter.flush())
    return lines


@PROFILE
@hypothesis.given(st.lists(line_bytes(), max_size=6), st.booleans(), st.lists(st.integers(0, 40_000), max_size=12))
def test_line_splitter_result_does_not_depend_on_where_the_chunks_fall(
    pieces: list[bytes], ends_with_newline: bool, cuts: list[int]
) -> None:
    data = b"\n".join(pieces) + (b"\n" if ends_with_newline else b"")
    assert run_splitter(split_at(data, cuts)) == run_splitter([data]) == reference_lines(data)


@st.composite
def estimator_streams(draw: Any) -> list[tuple[float, Record]]:
    names = ["a", "b", "c"]
    parents = {"a": None, "b": draw(st.sampled_from([None, "a"])), "c": draw(st.sampled_from([None, "a", "b"]))}
    is_root = draw(st.booleans())
    state: dict[str, dict[str, Any]] = {}
    now = 0.0
    out: list[tuple[float, Record]] = []
    for _ in range(draw(st.integers(1, 40))):
        name = draw(st.sampled_from(names))
        now += draw(st.sampled_from([0.5, 1.0, 2.0, 30.0, 90.0, 300.0]))
        if draw(st.integers(0, 15)) == 0:
            out.append((now, Record("plan", Event.PLAN, len(out) + 1, 0.0, phases=(("a", 0.5), ("b", 0.5)))))
            continue
        scope = state.get(name)
        if scope is None:
            total = draw(st.sampled_from([None, 10.0, 100.0, 1000.0]))
            scope = state[name] = {"seq": 0, "done": 0.0, "closed": False, "started": now, "total": total}
        if scope["closed"]:
            continue
        scope["seq"] += 1
        if scope["seq"] == 1:
            event = Event.START
        else:
            event = draw(st.sampled_from([Event.UPDATE] * 6 + [Event.END, Event.FAIL]))
            scope["done"] += draw(st.sampled_from([0.0, 1.0, 5.0]))
            scope["closed"] = event is not Event.UPDATE
        record = Record(
            name,
            event,
            scope["seq"],
            now - scope["started"],
            completed=scope["done"],
            total=scope["total"],
            parent=parents[name],
            root=name == "a" and is_root,
        )
        out.append((now, record))
    return out


@PROFILE
@hypothesis.given(estimator_streams(), st.sampled_from([0.0, 1.0, 1000.0]))
def test_the_estimate_is_always_finite_and_never_negative(stream: list[tuple[float, Record]], later: float) -> None:
    job = JobForecast()
    for received, record in stream:
        job.observe(record, received)
        for now in (received, received + later):
            estimate = job.snapshot(now)
            assert isinstance(estimate.state, ForecastState)
            assert estimate.remaining_seconds is None or 0 <= estimate.remaining_seconds < math.inf
            for forecast in estimate.scopes.values():
                assert forecast.remaining_seconds is None or 0 <= forecast.remaining_seconds < math.inf


@PROFILE
@hypothesis.given(estimator_streams())
def test_reading_every_record_twice_changes_no_estimate(stream: list[tuple[float, Record]]) -> None:
    once, twice = JobForecast(), JobForecast()
    for received, record in stream:
        once.observe(record, received)
        twice.observe(record, received)
        twice.observe(record, received)
        assert once.snapshot(received) == twice.snapshot(received)
