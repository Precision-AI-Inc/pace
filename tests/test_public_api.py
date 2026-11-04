# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""End-to-end: what the logger writes, the parser reads and the estimator understands."""

import importlib

import pytest

import precisionai.pace as pace
from precisionai.pace import ForecastState, JobForecast, Logger, parse_line, track

from .conftest import FakeClock, FakeStream

PACKAGE_NAMES = [
    "Event",
    "Forecast",
    "ForecastConfig",
    "ForecastState",
    "JobEstimate",
    "JobForecast",
    "LineSplitter",
    "Logger",
    "PaceHandler",
    "Phase",
    "Record",
    "Scope",
    "ScopeForecast",
    "__version__",
    "capture_logging",
    "format_trailer",
    "get_version",
    "logger",
    "parse_line",
    "phase",
    "plan",
    "progress",
    "timed",
    "track",
]

# Every name a documented submodule offers, which `docs/modules.rst` and the docs name. They stay importable
# from these paths however the code behind them is organised.
SUBMODULE_NAMES = {
    "core": "DEFAULT_TERMINAL_COLUMNS LEVELS Logger Settings Target logger",
    "scopes": (
        "FALLBACK_ID MAX_DERIVED_ID_CHARS MAX_ID_CHARS Phase PlanEntry REDRAW_INTERVAL Scope UPDATE_FRACTION "
        "derive_scope_id phase plan progress timed track validate_scope_id"
    ),
    "capture": "PaceHandler capture_logging",
    "protocol": (
        "Event LineSplitter MARKER MAX_FIELDS MAX_LABEL_CHARS MAX_LINE_BYTES MAX_NUMBER MAX_PHASES MAX_PHASES_CHARS "
        "MAX_SEQ MAX_UNIT_CHARS MAX_VALUE_CHARS RUN_PATTERN Record SCOPE_PATTERN decode_value encode_value "
        "format_number format_trailer parse_line"
    ),
    "forecast": "DEFAULT_CONFIG Forecast ForecastConfig ForecastState JobEstimate JobForecast ScopeForecast",
    "lightning": "PaceCallback",
    "transformers": "PaceCallback",
    "tqdm": "MIN_ELAPSED PaceTqdm",
    "version": "DISTRIBUTION UNKNOWN_VERSION get_version",
}


def test_public_names_are_exported() -> None:
    assert sorted(pace.__all__) == pace.__all__
    assert pace.__all__ == sorted(PACKAGE_NAMES)
    for name in pace.__all__:
        assert hasattr(pace, name), name


@pytest.mark.parametrize("module", sorted(SUBMODULE_NAMES))
def test_documented_submodules_offer_their_names(module: str) -> None:
    imported = importlib.import_module(f"precisionai.pace.{module}")
    assert [name for name in SUBMODULE_NAMES[module].split() if not hasattr(imported, name)] == []


@pytest.mark.parametrize("module", sorted(SUBMODULE_NAMES))
def test_documented_submodules_are_attributes_of_the_package(module: str) -> None:
    imported = importlib.import_module(f"precisionai.pace.{module}")
    assert getattr(pace, module) is imported


def test_written_progress_parses_back_and_drives_a_live_forecast(
    log: Logger, clock: FakeClock, stream: FakeStream
) -> None:
    def batches():
        for index in range(100):
            clock.advance(2.0)
            yield index

    for _ in track(batches(), description="Train", unit="batch", total=100, root=True, logger=log):
        pass

    records = [parse_line(line) for line in stream.lines()]
    assert all(record is not None for record in records)
    job = JobForecast()
    for record in records[:-1]:
        assert record is not None
        job.observe(record, record.elapsed)
    middle = job.snapshot(records[-2].elapsed)  # type: ignore[union-attr]
    assert middle.state is ForecastState.LIVE
    assert middle.source == "root"

    final = records[-1]
    assert final is not None
    job.observe(final, final.elapsed)
    assert job.snapshot(final.elapsed).state is ForecastState.COMPLETE
    assert records[0] is not None
    assert records[0].run == log.run_id
