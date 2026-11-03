# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.pace.protocol (reading side)."""

from __future__ import annotations

import dataclasses
import json
import random
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
VALID = "text [pace:1 s=a k=upd c=1 t=2 e=1 q=2]"


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


def test_parse_line_returns_record_fields() -> None:
    record = parse_line("[pace:1 s=val p=train root=1 k=start c=0 t=? u=batch e=0 q=1 res=1 n=Val]")
    assert record == Record(
        scope="val",
        event=Event.START,
        seq=1,
        elapsed=0.0,
        completed=0.0,
        total=None,
        unit="batch",
        parent="train",
        root=True,
        resumed=True,
        label="Val",
    )


@pytest.mark.parametrize("value", [None, 5, b"[pace:1 s=a k=upd c=1 t=2 e=1 q=2]", ["x"]])
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
    assert parse_line(f"[pace:1 s=a k=upd c={arabic_one} t=2 e=1 q=2]") is None
    assert parse_line(f"[pace:1 s=a k=upd c=1 t=2 e=1 q={arabic_two}]") is None


# --- round trips ---


def _expected_after_write(record: Record) -> Record:
    label = None if record.label is None else record.label[:MAX_LABEL_CHARS]
    return dataclasses.replace(record, elapsed=round(record.elapsed, 1), label=label)


def _handwritten_records() -> list[Record]:
    return [
        Record("train", Event.START, 1, 0.0, 0.0, 5000.0, unit="step", root=True, label="Fine-tune 7B"),
        Record("train", Event.START, 1, 0.0, 3100.0, 5000.0, unit="step", resumed=True),
        Record("train", Event.UPDATE, 57, 763.12, 1840.0, 5000.0, root=True),
        Record("read", Event.UPDATE, 2, 3.04, 12.5, None),
        Record("val", Event.UPDATE, 3, 0.0, 0.000001, 0.5, parent="train"),
        Record("train", Event.END, 90, 2075.46, 5000.0, 5000.0),
        Record("train", Event.FAIL, 9, 12.5),
        Record("job", Event.PLAN, 1, 0.0, phases=(("prep", 0.05), ("train", 0.9), ("eval", 0.05))),
        Record("job", Event.START, 1, 0.0, 0.0, None, unit="café", label="日本語 " * 5),
        Record("job", Event.START, 1, 0.0, 0.0, 1.0, label="x" * 100),
        Record("a.b-c_D", Event.UPDATE, 10**12, 1e14, 1e15, 1e15),
    ]


def _random_record(rng: random.Random) -> Record:
    event = rng.choice(list(Event))
    counted = event in (Event.START, Event.UPDATE, Event.END)
    names = ["a", "b.c", "D-e", "f_g", "x" * 64]
    return Record(
        scope=rng.choice(names),
        event=event,
        seq=rng.randint(1, 10**6),
        elapsed=round(rng.uniform(0, 1e6), 3),
        completed=round(rng.uniform(0, 1e6), 6) if counted else None,
        total=rng.choice([None, round(rng.uniform(0.000001, 1e6), 6)]) if counted else None,
        unit=rng.choice([None, "step", "sample/s", "é"]) if event is Event.START else None,
        parent=rng.choice([None, "p", "q-1"]),
        root=rng.random() < 0.3,
        resumed=event is Event.START and rng.random() < 0.3,
        phases=tuple((f"p{i}", round(rng.uniform(0, 1), 3)) for i in range(rng.randint(0, 6)))
        if event is Event.PLAN
        else (),
        label=rng.choice([None, "Train", "résumé ✓", "a b" * 30]) if event is Event.START else None,
    )


@pytest.mark.parametrize("record", _handwritten_records(), ids=lambda record: f"{record.scope}-{record.event.value}")
def test_trailer_round_trips(record: Record) -> None:
    assert parse_line("text " + format_trailer(record)) == _expected_after_write(record)


def test_trailer_round_trips_random_records() -> None:
    rng = random.Random(20261103)
    for _ in range(500):
        record = _random_record(rng)
        assert parse_line("text " + format_trailer(record)) == _expected_after_write(record)
        assert parse_line(format_trailer(record) + "\r\n") == _expected_after_write(record)


# --- never raises ---


def _garbage_line(rng: random.Random) -> str:
    pieces = [
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
    ]
    pieces += [
        "\x07",
        "\r",
        "\n",
        "é",
        "\ud800",
        "\U0001f600",
        "0",
        "-",
        "ph=a:1,b:2",
        "root=1",
        "n=%FF",
        "u=caf%C3%A9",
    ]
    if rng.random() < 0.5:
        return "".join(rng.choice(pieces) for _ in range(rng.randint(0, 60)))
    return "".join(chr(rng.randint(0, 0x10FFFF)) for _ in range(rng.randint(0, 80)))


def _mutate(rng: random.Random, line: str) -> str:
    chars = list(line)
    for _ in range(rng.randint(1, 4)):
        position = rng.randint(0, len(chars))
        action = rng.choice(["delete", "insert", "replace", "swap"])
        junk = rng.choice(list("[]= %.,:-?e\x1b\x00é9aZ"))
        if action == "delete" and chars:
            del chars[min(position, len(chars) - 1)]
        elif action == "replace" and chars:
            chars[min(position, len(chars) - 1)] = junk
        elif action == "swap" and len(chars) > 1:
            index = min(position, len(chars) - 2)
            chars[index], chars[index + 1] = chars[index + 1], chars[index]
        else:
            chars.insert(position, junk)
    return "".join(chars)


def test_parse_line_never_raises_on_garbage() -> None:
    rng = random.Random(1)
    for _ in range(5000):
        result = parse_line(_garbage_line(rng))
        assert result is None or isinstance(result, Record)


def test_parse_line_never_raises_on_mutated_valid_lines() -> None:
    rng = random.Random(2)
    lines = [case["line"] for case in CASES if case["record"] is not None]
    for _ in range(5000):
        result = parse_line(_mutate(rng, rng.choice(lines)))
        assert result is None or isinstance(result, Record)
        if result is not None:
            assert format_trailer(result).startswith("[pace:1 ")


def test_parse_line_handles_pathological_input() -> None:
    assert parse_line("\x1b[" * 2000) is None
    assert parse_line("\x1b]" * 2000) is None
    assert parse_line("[pace:1 " * 500 + "]") is None
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


def test_splitter_output_does_not_depend_on_chunk_boundaries() -> None:
    rng = random.Random(3)
    text = "".join(
        rng.choice(
            [
                "short line\n",
                "crlf line\r\n",
                "é日本 😀\n",
                "\n",
                "x" * 5000 + "\n",
                "x" * MAX_LINE_BYTES + "\r\n",
                "tail",
            ]
        )
        for _ in range(40)
    )
    data = text.encode()
    whole = LineSplitter()
    expected = whole.feed(data) + whole.flush()
    for _ in range(50):
        splitter = LineSplitter()
        lines: list[str] = []
        position = 0
        while position < len(data):
            size = rng.choice([1, 2, 3, 7, 100, 4096, 5000])
            lines += splitter.feed(data[position : position + size])
            position += size
        assert lines + splitter.flush() == expected


def test_splitter_feeds_parse_line() -> None:
    log = (
        b"2026-11-03T17:23:59Z INFO     Loading\r\n"
        b"Train [pace:1 s=train k=upd c=1 t=2 e=1 q=2]\r\n"
        b"Train [pace:1 s=train k=upd c=2 t=2 e=2 q=3]"
    )
    splitter = LineSplitter()
    lines = []
    for index in range(0, len(log), 5):
        lines += splitter.feed(log[index : index + 5])
    lines += splitter.flush()
    records = [parse_line(line) for line in lines]
    assert [record.seq for record in records if record is not None] == [2, 3]
