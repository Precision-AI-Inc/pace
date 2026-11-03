# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.pace.protocol (writing side)."""

import pytest

from precisionai.pace.protocol import Event, Record, encode_value, format_number, format_trailer


def test_format_trailer_update_matches_design_example() -> None:
    record = Record(scope="train", event=Event.UPDATE, seq=57, elapsed=763.12, completed=1840, total=5000, root=True)
    assert format_trailer(record) == "[pace:1 s=train root=1 k=upd c=1840 t=5000 e=763.1 q=57]"


def test_format_trailer_start_with_unit_label_and_resume() -> None:
    record = Record(
        scope="train",
        event=Event.START,
        seq=1,
        elapsed=0.0,
        completed=3100,
        total=5000,
        unit="step",
        parent="job",
        resumed=True,
        label="Fine-tune 7B",
    )
    assert (
        format_trailer(record) == "[pace:1 s=train p=job k=start c=3100 t=5000 u=step e=0 q=1 res=1 n=Fine-tune%207B]"
    )


def test_format_trailer_unknown_total() -> None:
    record = Record(scope="read", event=Event.UPDATE, seq=2, elapsed=3.0, completed=12)
    assert "t=?" in format_trailer(record)


def test_format_trailer_plan_has_no_counts() -> None:
    record = Record(scope="job", event=Event.PLAN, seq=1, elapsed=0.0, phases=(("prep", 0.05), ("train", 0.95)))
    assert format_trailer(record) == "[pace:1 s=job k=plan e=0 q=1 ph=prep:0.05,train:0.95]"


def test_format_trailer_fail_carries_counts_it_reached() -> None:
    record = Record(scope="train", event=Event.FAIL, seq=9, elapsed=12.5, completed=10, total=20)
    assert format_trailer(record) == "[pace:1 s=train k=fail c=10 t=20 e=12.5 q=9]"


def test_format_trailer_fail_without_counts_has_none() -> None:
    record = Record(scope="train", event=Event.FAIL, seq=9, elapsed=12.5)
    assert format_trailer(record) == "[pace:1 s=train k=fail e=12.5 q=9]"


def test_format_trailer_truncates_label() -> None:
    record = Record(scope="a", event=Event.START, seq=1, elapsed=0.0, completed=0, total=1, label="x" * 100)
    assert format_trailer(record).endswith(f"n={'x' * 64}]")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"scope": "bad scope"}, "invalid scope id"),
        ({"parent": "bad/parent"}, "invalid parent scope id"),
        ({"seq": 0}, "seq must be at least 1"),
    ],
)
def test_format_trailer_rejects_invalid_ids(kwargs: dict, message: str) -> None:
    base = {"scope": "s", "event": Event.UPDATE, "seq": 1, "elapsed": 1.0, "completed": 1, "total": 2}
    base.update(kwargs)
    with pytest.raises(ValueError, match=message):
        format_trailer(Record(**base))


def test_format_trailer_rejects_invalid_phase_name() -> None:
    record = Record(scope="job", event=Event.PLAN, seq=1, elapsed=0.0, phases=(("a:b", 1.0),))
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


def test_format_trailer_writes_run_id_on_start() -> None:
    record = Record(scope="a", event=Event.START, seq=1, elapsed=0.0, completed=0, total=1, run="a1b2c3d4")
    assert format_trailer(record) == "[pace:1 s=a k=start c=0 t=1 e=0 q=1 run=a1b2c3d4]"


def test_format_trailer_rejects_invalid_run_id() -> None:
    record = Record(scope="a", event=Event.START, seq=1, elapsed=0.0, completed=0, total=1, run="Not-Valid")
    with pytest.raises(ValueError, match="invalid run id"):
        format_trailer(record)


def test_format_trailer_rejects_too_many_phases() -> None:
    phases = tuple((f"p{i}", 1.0) for i in range(33))
    record = Record(scope="job", event=Event.PLAN, seq=1, elapsed=0.0, phases=phases)
    with pytest.raises(ValueError, match="at most 32 phases"):
        format_trailer(record)


def test_format_trailer_truncates_unit_and_replaces_control_characters() -> None:
    record = Record(
        scope="a", event=Event.START, seq=1, elapsed=0.0, completed=0, total=1, unit="u" * 80, label="two\nlines"
    )
    trailer = format_trailer(record)
    assert "u=" + "u" * 64 + " " in trailer
    assert "n=two%20lines]" in trailer


def test_format_trailer_label_of_only_control_characters_becomes_dash() -> None:
    record = Record(scope="a", event=Event.START, seq=1, elapsed=0.0, completed=0, total=1, label="\x1b\x07")
    assert format_trailer(record).endswith("n=-]")


def test_format_number_rejects_values_the_parser_would_reject() -> None:
    with pytest.raises(ValueError, match="at most"):
        format_number(1e16)
