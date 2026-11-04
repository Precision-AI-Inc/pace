# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.pace.protocol (writing side)."""

import pytest

from precisionai.pace.protocol import Event, Record, encode_value, format_number, format_trailer, parse_line

RUN = "1b8b7650"


def test_format_trailer_update_matches_design_example() -> None:
    record = Record(
        scope="train", event=Event.UPDATE, seq=57, elapsed=763.12, completed=1840, total=5000, root=True, run=RUN
    )
    assert format_trailer(record) == "[pace:2 train c=1840 t=5000 e=763.1 q=57 r=1b8b7650 root]"


def test_format_trailer_start_with_unit_label_and_resume() -> None:
    record = Record(
        scope="train",
        event=Event.START,
        completed=3100,
        total=5000,
        unit="step",
        parent="job",
        resumed=True,
        label="Fine-tune 7B",
        run=RUN,
    )
    assert format_trailer(record) == "[pace:2 train<job start c=3100 t=5000 u=step n=Fine-tune%207B r=1b8b7650 res]"


def test_format_trailer_start_leaves_out_a_zero_count_and_elapsed_time() -> None:
    record = Record(scope="generate", event=Event.START, completed=0, total=128, unit="request", run=RUN)
    assert format_trailer(record) == "[pace:2 generate start t=128 u=request r=1b8b7650]"


def test_format_trailer_unknown_total() -> None:
    record = Record(scope="read", event=Event.UPDATE, seq=2, elapsed=3.0, completed=12, run=RUN)
    assert "t=?" in format_trailer(record)


@pytest.mark.parametrize(
    ("completed", "total", "counts"),
    [(128, 128, "c=128"), (40, 128, "c=40 t=128"), (40, None, "c=40 t=?"), (0, None, "t=?")],
)
def test_format_trailer_end_leaves_out_a_total_equal_to_the_count(
    completed: float, total: float | None, counts: str
) -> None:
    record = Record(scope="g", event=Event.END, elapsed=197.4, completed=completed, total=total, run=RUN)
    assert format_trailer(record) == f"[pace:2 g end {counts} e=197.4 r=1b8b7650]"
    assert parse_line(format_trailer(record)) == record


def test_format_trailer_step_carries_parent_elapsed_label_and_run() -> None:
    record = Record(
        scope="start-ollama",
        event=Event.STEP,
        elapsed=12.04,
        completed=1,
        total=1,
        unit="call",
        parent="generate",
        label="Start Ollama",
        run=RUN,
    )
    assert format_trailer(record) == "[pace:2 start-ollama<generate step e=12 n=Start%20Ollama r=1b8b7650]"


def test_format_trailer_plan_has_no_counts() -> None:
    record = Record(scope="job", event=Event.PLAN, seq=1, phases=(("prep", 0.05), ("train", 0.95)), run=RUN)
    assert format_trailer(record) == "[pace:2 job plan q=1 ph=prep:0.05,train:0.95 r=1b8b7650]"


def test_format_trailer_fail_carries_counts_it_reached() -> None:
    record = Record(scope="train", event=Event.FAIL, elapsed=12.5, completed=10, total=20, run=RUN)
    assert format_trailer(record) == "[pace:2 train fail c=10 t=20 e=12.5 r=1b8b7650]"


def test_format_trailer_fail_without_counts_has_none() -> None:
    record = Record(scope="train", event=Event.FAIL, elapsed=12.5, run=RUN)
    assert format_trailer(record) == "[pace:2 train fail e=12.5 r=1b8b7650]"


def test_format_trailer_truncates_label() -> None:
    record = Record(scope="a", event=Event.START, completed=0, total=1, label="x" * 100, run=RUN)
    assert f"n={'x' * 64} " in format_trailer(record)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"scope": "bad scope"}, "invalid scope id"),
        ({"scope": "a<b"}, "invalid scope id"),
        ({"parent": "bad/parent"}, "invalid parent scope id"),
        ({"run": None}, "invalid run id"),
        ({"seq": None}, "upd needs a seq of at least 2"),
        ({"seq": 1}, "upd needs a seq of at least 2"),
        ({"event": Event.PLAN, "seq": 0}, "plan needs a seq of at least 1"),
    ],
)
def test_format_trailer_rejects_invalid_ids(kwargs: dict, message: str) -> None:
    base = {"scope": "s", "event": Event.UPDATE, "seq": 2, "elapsed": 1.0, "completed": 1, "total": 2, "run": RUN}
    base.update(kwargs)
    with pytest.raises(ValueError, match=message):
        format_trailer(Record(**base))


def test_format_trailer_rejects_invalid_phase_name() -> None:
    record = Record(scope="job", event=Event.PLAN, seq=1, phases=(("a:b", 1.0),), run=RUN)
    with pytest.raises(ValueError, match="invalid phase scope id"):
        format_trailer(record)


def test_encode_value_escapes_unsafe_characters_as_utf8() -> None:
    assert encode_value("a b]é") == "a%20b%5D%C3%A9"


def test_encode_value_rejects_empty() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        encode_value("")


@pytest.mark.parametrize(("value", "expected"), [(5000, "5000"), (2.5, "2.5"), (0.1234567, "0.123457"), (0, "0")])
def test_format_number(value: float, expected: str) -> None:
    assert format_number(value) == expected


@pytest.mark.parametrize("value", [-1.0, float("inf"), float("nan")])
def test_format_number_rejects_invalid(value: float) -> None:
    with pytest.raises(ValueError, match="finite, non-negative"):
        format_number(value)


def test_format_trailer_writes_the_run_id_on_every_record() -> None:
    for event in (Event.START, Event.UPDATE, Event.END):
        record = Record(scope="a", event=event, seq=2, elapsed=2.0, completed=1, total=1, run=RUN)
        assert format_trailer(record).endswith(" r=1b8b7650]")


def test_format_trailer_rejects_invalid_run_id() -> None:
    record = Record(scope="a", event=Event.START, completed=0, total=1, run="Not-Valid")
    with pytest.raises(ValueError, match="invalid run id"):
        format_trailer(record)


def test_format_trailer_rejects_too_many_phases() -> None:
    phases = tuple((f"p{i}", 1.0) for i in range(33))
    record = Record(scope="job", event=Event.PLAN, seq=1, phases=phases, run=RUN)
    with pytest.raises(ValueError, match="at most 32 phases"):
        format_trailer(record)


def test_format_trailer_truncates_unit_and_replaces_control_characters() -> None:
    record = Record(scope="a", event=Event.START, completed=0, total=1, unit="u" * 80, label="two\nlines", run=RUN)
    trailer = format_trailer(record)
    assert "u=" + "u" * 64 + " " in trailer
    assert "n=two%20lines " in trailer


def test_format_trailer_label_of_only_control_characters_becomes_dash() -> None:
    record = Record(scope="a", event=Event.START, completed=0, total=1, label="\x1b\x07", run=RUN)
    assert "n=- " in format_trailer(record)


def test_format_number_rejects_values_the_parser_would_reject() -> None:
    with pytest.raises(ValueError, match="at most"):
        format_number(1e16)


@pytest.mark.parametrize(
    "record",
    [
        Record(scope="a", event=Event.FAIL, seq=2, elapsed=1.0, completed=1, total=2, unit="step", run=RUN),
        Record(scope="a", event=Event.UPDATE, seq=2, elapsed=1.0, completed=1, total=2, run=RUN),
        Record(scope="a", event=Event.END, elapsed=1.0, completed=2, total=2, resumed=True, label="x", run=RUN),
        Record(scope="a", event=Event.PLAN, seq=1, phases=(("b", 1.0),), label="x", unit="it", run=RUN),
        Record(scope="a", event=Event.UPDATE, seq=2, elapsed=1.0, completed=1, total=2, phases=(("b", 1.0),), run=RUN),
        Record(scope="a", event=Event.STEP, elapsed=1.0, completed=1, total=1, unit="call", resumed=True, run=RUN),
    ],
)
def test_format_trailer_leaves_out_keys_the_event_may_not_carry(record: Record) -> None:
    assert parse_line(format_trailer(record)) is not None


def test_a_total_too_small_to_write_is_written_as_unknown() -> None:
    record = Record(scope="a", event=Event.UPDATE, seq=2, elapsed=1.0, completed=0, total=1e-7, run=RUN)
    assert "t=?" in format_trailer(record)
    assert parse_line(format_trailer(record)) is not None


def test_the_largest_plan_the_writer_can_produce_parses() -> None:
    phases = tuple((f"{index:02d}" + "p" * 62, 999999999999999.999999) for index in range(32))
    record = Record(scope="plan", event=Event.PLAN, seq=1, phases=phases, run=RUN)
    parsed = parse_line(format_trailer(record))
    assert parsed is not None
    assert len(parsed.phases) == 32
