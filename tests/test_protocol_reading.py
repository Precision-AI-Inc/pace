# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.pace.protocol (reading side)."""

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

from precisionai.pace.protocol import (
    MAX_LABEL_CHARS,
    MAX_LINE_BYTES,
    Event,
    LineSplitter,
    Record,
    decode_value,
    encode_value,
    format_trailer,
    parse_line,
)

FIXTURES = Path(__file__).parent / "fixtures" / "protocol" / "cases.jsonl"
VALID = "text [pace a c=1 t=2 e=1 q=2 r=1b8b7650]"
RUN = "1b8b7650"


def _load_cases() -> list[dict[str, Any]]:
    with FIXTURES.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _record_from_fixture(fields: dict[str, Any]) -> Record:
    values = dict(fields)
    values["event"] = Event(values["event"])
    values["phases"] = tuple((name, weight) for name, weight in values.get("phases", []))
    return Record(**values)


CASES = _load_cases()


# --- decode_value ---


@pytest.mark.parametrize("value", ["plain", "a b]é", "日本語 label", "100%", "x" * 50, "\x00\n\x1b[31m"])
def test_decode_value_reverses_encode_value(value: str) -> None:
    assert decode_value(encode_value(value)) == value


def test_decode_value_accepts_lower_case_hex() -> None:
    assert decode_value("caf%c3%a9") == "café"


@pytest.mark.parametrize("value", ["%", "%4", "%zz", "a%4", "%%41", "%4g"])
def test_decode_value_rejects_malformed_escape(value: str) -> None:
    with pytest.raises(ValueError, match="malformed percent escape"):
        decode_value(value)


@pytest.mark.parametrize("value", ["%FF", "caf%C3", "%C3%28", "%ED%A0%80"])
def test_decode_value_rejects_invalid_utf8(value: str) -> None:
    with pytest.raises(ValueError, match="not valid UTF-8"):
        decode_value(value)


# --- fixtures ---


def test_fixture_file_covers_both_outcomes() -> None:
    assert sum(case["record"] is not None for case in CASES) >= 20
    assert sum(case["record"] is None for case in CASES) >= 50
    assert len({case["name"] for case in CASES}) == len(CASES)


@pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
def test_parse_line_matches_fixture(case: dict[str, Any]) -> None:
    expected: Record | None = None if case["record"] is None else _record_from_fixture(case["record"])
    assert parse_line(case["line"]) == expected


def test_fixture_records_cover_every_event_kind() -> None:
    events = {case["record"]["event"] for case in CASES if case["record"] is not None}
    assert events == {event.value for event in Event}


# --- parse_line, beyond the fixtures ---


def test_parse_line_reads_a_trailer() -> None:
    record = parse_line("[pace val<train start t=? u=batch n=Val r=1b8b7650 root res]")
    assert record == Record(
        scope="val",
        event=Event.START,
        completed=0.0,
        total=None,
        unit="batch",
        parent="train",
        root=True,
        resumed=True,
        label="Val",
        run=RUN,
    )


def test_an_old_or_versioned_marker_is_ordinary_text() -> None:
    assert parse_line("[pace:1 s=a k=upd c=1 t=2 e=1 q=2]") is None
    assert parse_line("[pace:2 a c=1 t=2 e=1 q=2 r=ab]") is None
    assert parse_line("[pace  a c=1 t=2 e=1 q=2 r=ab]") is None


@pytest.mark.parametrize("value", [None, 5, b"[pace a c=1 t=2 e=1 q=2 r=ab]", ["x"]])
def test_parse_line_ignores_non_strings(value: Any) -> None:
    assert parse_line(value) is None


def test_parse_line_ignores_lone_surrogates() -> None:
    assert parse_line("\ud800 " + VALID) is not None
    assert parse_line("\ud800" * 3000) is None


def test_parse_line_size_limit_is_in_bytes() -> None:
    padding = MAX_LINE_BYTES - len(VALID) - 1
    assert parse_line("x" * padding + " " + VALID) is not None
    assert parse_line("x" * (padding + 1) + " " + VALID) is None
    assert parse_line("é" * (padding // 2 + 1) + " " + VALID) is None


def test_parse_line_does_not_accept_non_ascii_digits() -> None:
    arabic_one, arabic_two = chr(0x661), chr(0x662)
    assert parse_line(f"[pace a c={arabic_one} t=2 e=1 q=2 r=ab]") is None
    assert parse_line(f"[pace a c=1 t=2 e=1 q={arabic_two} r=ab]") is None


# --- round trips ---


def _expected_after_write(record: Record) -> Record:
    label = None if record.label is None else record.label[:MAX_LABEL_CHARS]
    return dataclasses.replace(record, elapsed=round(record.elapsed, 1), label=label)


def _handwritten_records() -> list[Record]:
    return [
        Record("train", Event.START, None, 0.0, 0.0, 5000.0, unit="step", root=True, label="Fine-tune 7B", run=RUN),
        Record("train", Event.START, None, 0.0, 3100.0, 5000.0, unit="step", resumed=True, run=RUN),
        Record("train", Event.UPDATE, 57, 763.12, 1840.0, 5000.0, root=True, run=RUN),
        Record("read", Event.UPDATE, 2, 3.04, 12.5, None, run=RUN),
        Record("val", Event.UPDATE, 3, 0.0, 0.000001, 0.5, parent="train", run=RUN),
        Record("train", Event.END, None, 2075.46, 5000.0, 5000.0, run=RUN),
        Record("train", Event.FAIL, None, 12.5, run=RUN),
        Record("load", Event.STEP, None, 12.04, 1.0, 1.0, parent="prep", label="Load", run=RUN),
        Record("job", Event.PLAN, 1, 0.0, phases=(("prep", 0.05), ("train", 0.9), ("eval", 0.05)), run=RUN),
        Record("job", Event.START, None, 0.0, 0.0, None, unit="café", label="日本語 " * 5, run=RUN),
        Record("job", Event.START, None, 0.0, 0.0, 1.0, label="x" * 100, run=RUN),
        Record("a.b-c_D", Event.UPDATE, 10**12, 1e14, 1e15, 1e15, run=RUN),
    ]


@pytest.mark.parametrize("record", _handwritten_records(), ids=lambda record: f"{record.scope}-{record.event.value}")
def test_trailer_round_trips(record: Record) -> None:
    assert parse_line("text " + format_trailer(record)) == _expected_after_write(record)


def test_parse_line_handles_pathological_input() -> None:
    assert parse_line("\x1b[" * 2000) is None
    assert parse_line("\x1b]" * 2000) is None
    assert parse_line("[pace " * 500 + "]") is None
    assert parse_line("[pace a" + " root" * 2000 + "]") is None
    assert parse_line(" " * 4000 + "]") is None


# --- LineSplitter ---


def test_splitter_returns_complete_lines_and_keeps_the_tail() -> None:
    splitter = LineSplitter()
    assert splitter.feed(b"one\ntwo\nthr") == ["one", "two"]
    assert splitter.feed(b"ee\nfour") == ["three"]
    assert splitter.feed(b"\n") == ["four"]


def test_splitter_returns_empty_lines() -> None:
    assert LineSplitter().feed(b"\n\r\nx\n") == ["", "", "x"]


def test_splitter_handles_crlf_and_lone_cr() -> None:
    splitter = LineSplitter()
    assert splitter.feed(b"one\r\ntwo\rstill two\r\n") == ["one", "two\rstill two"]


def test_splitter_crlf_split_across_chunks() -> None:
    splitter = LineSplitter()
    assert splitter.feed(b"one\r") == []
    assert splitter.feed(b"\ntwo\r\n") == ["one", "two"]


def test_splitter_multibyte_character_split_across_chunks() -> None:
    data = "café 日本語 😀\n".encode()
    for cut in range(1, len(data)):
        splitter = LineSplitter()
        assert splitter.feed(data[:cut]) == []
        assert splitter.feed(data[cut:]) == ["café 日本語 😀"]


def test_splitter_replaces_invalid_utf8() -> None:
    assert LineSplitter().feed(b"a\xffb\n") == ["a�b"]


def test_splitter_accepts_empty_chunks() -> None:
    splitter = LineSplitter()
    assert splitter.feed(b"") == []
    assert splitter.feed(b"x") == []
    assert splitter.feed(b"") == []
    assert splitter.feed(b"\n") == ["x"]


def test_splitter_discards_oversize_line_up_to_next_newline() -> None:
    splitter = LineSplitter()
    assert splitter.feed(b"ok\n" + b"x" * (MAX_LINE_BYTES + 2)) == ["ok"]
    assert splitter.feed(b"y" * 10_000) == []
    assert splitter.feed(b"zz\nnext\nrest") == ["next"]
    assert splitter.flush() == ["rest"]


def test_splitter_drops_oversize_line_that_arrives_in_one_chunk() -> None:
    splitter = LineSplitter()
    assert splitter.feed(b"x" * (MAX_LINE_BYTES + 1) + b"\nafter\n") == ["after"]
    assert splitter.feed(b"x" * MAX_LINE_BYTES + b"\r\n") == ["x" * MAX_LINE_BYTES]


def test_splitter_tail_memory_is_bounded() -> None:
    splitter = LineSplitter()
    for _ in range(100):
        splitter.feed(b"x" * 1000)
        assert len(splitter._tail) <= MAX_LINE_BYTES + 1


def test_splitter_line_of_exactly_the_limit_survives_split_crlf() -> None:
    splitter = LineSplitter()
    assert splitter.feed(b"x" * MAX_LINE_BYTES + b"\r") == []
    assert splitter.feed(b"\n") == ["x" * MAX_LINE_BYTES]


def test_splitter_reset_drops_partial_line() -> None:
    splitter = LineSplitter()
    splitter.feed(b"half a li")
    splitter.reset()
    assert splitter.feed(b"fresh\n") == ["fresh"]
    assert splitter.flush() == []


def test_splitter_reset_ends_oversize_discard() -> None:
    splitter = LineSplitter()
    splitter.feed(b"x" * (MAX_LINE_BYTES + 5))
    splitter.reset()
    assert splitter.feed(b"fresh\n") == ["fresh"]


def test_splitter_flush_returns_pending_tail_once() -> None:
    splitter = LineSplitter()
    splitter.feed(b"done\nlast words")
    assert splitter.flush() == ["last words"]
    assert splitter.flush() == []


def test_splitter_flush_with_nothing_or_only_a_carriage_return() -> None:
    splitter = LineSplitter()
    assert splitter.flush() == []
    splitter.feed(b"\r")
    assert splitter.flush() == []


def test_splitter_flush_after_oversize_tail_returns_nothing() -> None:
    splitter = LineSplitter()
    splitter.feed(b"x" * (MAX_LINE_BYTES + 5))
    assert splitter.flush() == []
    assert splitter.feed(b"ok\n") == ["ok"]


def test_splitter_feeds_parse_line() -> None:
    log = (
        b"2026-11-03T17:23:59Z INFO     Loading\r\n"
        b"Train [pace train c=1 t=2 e=1 q=2 r=ab]\r\n"
        b"Train [pace train c=2 t=2 e=2 q=3 r=ab]"
    )
    splitter = LineSplitter()
    lines = []
    for index in range(0, len(log), 5):
        lines += splitter.feed(log[index : index + 5])
    lines += splitter.flush()
    records = [parse_line(line) for line in lines]
    assert [record.seq for record in records if record is not None] == [2, 3]


def test_parse_line_counts_replacement_characters_as_one_byte() -> None:
    trailer = "[pace a c=1 t=2 e=1 q=2 r=ab]"
    # 3000 undecodable source bytes become 3000 U+FFFD characters (9000 UTF-8 bytes).
    line = "�" * 3000 + " " + trailer
    assert parse_line(line) is not None


def test_line_splitter_drops_huge_line_in_one_chunk_without_buffering() -> None:
    splitter = LineSplitter()
    chunk = b"a" * 10_000_000 + b"\nok\n" + b"b" * 10_000_000
    assert splitter.feed(chunk) == ["ok"]
    assert splitter.feed(b"tail\nnext\n") == ["next"]


def test_a_line_of_exactly_the_limit_parses_with_its_line_ending() -> None:
    trailer = "[pace a c=1 t=2 e=1 q=2 r=ab]"
    line = "x" * (MAX_LINE_BYTES - len(trailer) - 1) + " " + trailer
    assert len(line.encode("utf-8")) == MAX_LINE_BYTES
    assert parse_line(line) is not None
    assert parse_line(line + "\n") is not None
    assert parse_line(line + "\r\n") is not None
