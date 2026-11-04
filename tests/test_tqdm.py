# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for the tqdm-compatible class, including the calls vLLM makes with one."""

from __future__ import annotations

import functools
from typing import Any, Callable, Iterator

import pytest

from precisionai.pace import Event, Record, parse_line, phase
from precisionai.pace.core import Logger
from precisionai.pace.tqdm import PaceTqdm
from tests.conftest import FakeClock, FakeStream


def records(stream: FakeStream) -> list[Record]:
    return [record for record in map(parse_line, stream.lines()) if record is not None]


@pytest.fixture
def quiet(log: Logger) -> Logger:
    log.configure(min_interval=0)
    return log


def run_vllm_engine(use_tqdm: Callable[..., Any], clock: FakeClock, requests: int = 4) -> None:
    """Make the calls vLLM's ``LLM._run_engine`` makes on the class it is given."""
    pbar = use_tqdm(
        total=requests,
        desc="Processed prompts",
        dynamic_ncols=True,
        postfix=f"est. speed input: {0:.2f} toks/s, output: {0:.2f} toks/s",
    )
    total_tokens = 0
    for _ in range(requests):
        clock.advance(2.0)
        total_tokens += 10
        speed = total_tokens / pbar.format_dict["elapsed"]
        pbar.postfix = f"est. speed output: {speed:.2f} toks/s"
        pbar.update(1)
        if pbar.n == requests:
            pbar.refresh()
    pbar.close()


def test_counting_by_hand_as_vllm_does_reports_a_scope(quiet: Logger, stream: FakeStream, clock: FakeClock) -> None:
    run_vllm_engine(functools.partial(PaceTqdm, logger=quiet), clock)
    found = records(stream)
    assert found[0].event is Event.START
    assert found[0].scope == "processed-prompts"
    assert found[0].label == "Processed prompts"
    assert found[0].total == 4
    assert found[-1].event is Event.END
    assert found[-1].completed == 4
    assert found[-1].elapsed == 8.0
    assert [r.completed for r in found if r.event is Event.UPDATE] == [1, 2, 3, 4]


def test_the_elapsed_time_is_never_zero(quiet: Logger) -> None:
    pbar = PaceTqdm(total=3, logger=quiet)
    assert pbar.format_dict["elapsed"] > 0
    pbar.close()


def test_format_dict_reports_the_rate_since_the_start(quiet: Logger, clock: FakeClock) -> None:
    pbar = PaceTqdm(total=10, initial=2, unit="req", desc="Run", logger=quiet)
    assert pbar.format_dict["rate"] is None
    clock.advance(4.0)
    pbar.update(2)
    info = pbar.format_dict
    assert info["n"] == 4
    assert info["total"] == 10
    assert info["elapsed"] == 4.0
    assert info["rate"] == 0.5
    assert info["unit"] == "req"
    assert info["prefix"] == "Run"


def test_wrapping_an_iterable_counts_each_item(quiet: Logger, stream: FakeStream) -> None:
    seen = list(PaceTqdm(["a", "b", "c"], desc="Rendering prompts", logger=quiet))
    assert seen == ["a", "b", "c"]
    found = records(stream)
    assert found[0].scope == "rendering-prompts"
    assert found[0].total == 3
    assert found[-1].event is Event.END
    assert found[-1].completed == 3


def test_an_iterable_without_a_length_has_an_unknown_total(quiet: Logger, stream: FakeStream) -> None:
    def numbers() -> Iterator[int]:
        yield from range(2)

    assert list(PaceTqdm(numbers(), logger=quiet)) == [0, 1]
    assert records(stream)[0].total is None


def test_an_explicit_total_beats_the_length(quiet: Logger, stream: FakeStream) -> None:
    list(PaceTqdm([1, 2], total=10, logger=quiet))
    assert records(stream)[0].total == 10


def test_an_abandoned_loop_ends_the_scope_where_it_stopped(quiet: Logger, stream: FakeStream) -> None:
    for item in PaceTqdm([1, 2, 3], logger=quiet):
        if item == 2:
            break
    found = records(stream)
    assert found[-1].event is Event.END
    assert found[-1].completed == 1


def test_an_iterable_that_raises_fails_the_scope(quiet: Logger, stream: FakeStream) -> None:
    def broken() -> Iterator[int]:
        yield 1
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        list(PaceTqdm(broken(), logger=quiet))
    assert records(stream)[-1].event is Event.FAIL


def test_a_bar_with_no_iterable_yields_nothing(quiet: Logger) -> None:
    assert list(PaceTqdm(total=3, logger=quiet)) == []


def test_the_context_manager_ends_or_fails_the_scope(quiet: Logger, stream: FakeStream) -> None:
    with PaceTqdm(total=2, desc="Fine", logger=quiet) as bar:
        bar.update(2)
    with pytest.raises(ValueError, match="bad"), PaceTqdm(total=2, desc="Broken", logger=quiet):
        raise ValueError("bad")
    found = records(stream)
    assert [r.event for r in found if r.scope == "fine"][-1] is Event.END
    assert [r.event for r in found if r.scope == "broken"][-1] is Event.FAIL


def test_disable_reports_nothing_but_still_works(quiet: Logger, stream: FakeStream) -> None:
    bar = PaceTqdm([1, 2], total=2, disable=True, logger=quiet)
    bar.update(1)
    bar.total = 5
    bar.close()
    bar.fail()
    assert bar.n == 1
    assert list(bar) == [1, 2]
    assert stream.getvalue() == ""


def test_setting_n_and_refreshing_reports_it(quiet: Logger, stream: FakeStream) -> None:
    bar = PaceTqdm(total=10, logger=quiet)
    bar.n = 7
    bar.refresh()
    bar.close()
    assert records(stream)[-1].completed == 7


def test_the_total_can_be_set_after_the_fact(quiet: Logger, stream: FakeStream) -> None:
    bar = PaceTqdm(logger=quiet)
    bar.total = 8
    bar.update(2)
    bar.close()
    assert bar.total == 8
    assert records(stream)[-1].total == 8


def test_reset_starts_again_from_zero_with_a_new_total(quiet: Logger, stream: FakeStream, clock: FakeClock) -> None:
    bar = PaceTqdm(total=4, logger=quiet)
    clock.advance(1.0)
    bar.update(4)
    bar.reset(total=6)
    assert bar.n == 0
    assert bar.total == 6
    assert bar.format_dict["elapsed"] == pytest.approx(1e-6)
    bar.update(1)
    bar.close()
    last = records(stream)[-1]
    assert (last.completed, last.total) == (1, 6)


def test_description_and_postfix_setters_do_not_write_anything_extra(quiet: Logger, stream: FakeStream) -> None:
    bar = PaceTqdm(total=3, desc="Old", logger=quiet)
    before = len(stream.lines())
    bar.set_description("New")
    bar.set_description()
    bar.set_postfix({"loss": 1.0})
    bar.set_postfix(loss=2.0)
    bar.set_postfix()
    bar.set_postfix_str("text")
    assert bar.desc == ""
    assert bar.postfix == "text"
    assert len(stream.lines()) == before


def test_a_new_description_appears_in_later_lines(quiet: Logger, stream: FakeStream) -> None:
    bar = PaceTqdm(total=3, desc="Old", logger=quiet)
    bar.set_description("New")
    bar.update(1)
    assert "New" in stream.lines()[-1]


def test_an_unknown_tqdm_method_is_accepted_and_ignored(quiet: Logger) -> None:
    bar = PaceTqdm(total=3, logger=quiet)
    assert bar.clear() is None
    assert bar.unpause(1, key="x") is None
    assert not hasattr(bar, "_private")


def test_write_goes_through_the_pace_logger(capsys: pytest.CaptureFixture[str]) -> None:
    PaceTqdm.write("beside the bar")
    assert "beside the bar" in capsys.readouterr().out


def test_len_and_truthiness(quiet: Logger) -> None:
    assert len(PaceTqdm([1, 2, 3], logger=quiet)) == 3
    assert len(PaceTqdm(total=5, logger=quiet)) == 5
    unsized = PaceTqdm(logger=quiet)
    with pytest.raises(TypeError, match="no len"):
        len(unsized)
    assert unsized


def test_scope_root_and_unit_options_reach_the_scope(quiet: Logger, stream: FakeStream) -> None:
    PaceTqdm(total=2, unit="req", scope="gen", root=True, logger=quiet).close()
    first = records(stream)[0]
    assert (first.scope, first.unit, first.root) == ("gen", "req", True)


def test_a_scope_opened_inside_a_phase_names_it_as_its_parent(quiet: Logger, stream: FakeStream) -> None:
    with phase("Generate", logger=quiet):
        PaceTqdm(total=2, desc="Processed prompts", logger=quiet).close()
    inner = [r for r in records(stream) if r.scope == "processed-prompts"]
    assert inner[0].parent == "generate"


def test_the_default_logger_is_used_when_none_is_given(capsys: pytest.CaptureFixture[str]) -> None:
    PaceTqdm(total=1, desc="Default").close()
    assert "[pace:1 s=default" in capsys.readouterr().out
