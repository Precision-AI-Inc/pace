# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""End-to-end: what the logger writes, the parser reads and the estimator understands."""

from __future__ import annotations

import precisionai.pace as pace
from precisionai.pace import ForecastState, JobForecast, Logger, parse_line, track

from .conftest import FakeClock, FakeStream


def test_public_names_are_exported() -> None:
    assert sorted(pace.__all__) == pace.__all__
    for name in pace.__all__:
        assert hasattr(pace, name), name


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
