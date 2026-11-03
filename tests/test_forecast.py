# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.pace.forecast.

Every test injects its clock: records carry ``elapsed`` and are delivered with an
explicit ``received_at``; nothing sleeps.
"""

from __future__ import annotations

import dataclasses
import random

import pytest

from precisionai.pace.forecast import (
    DEFAULT_CONFIG,
    ForecastConfig,
    ForecastState,
    JobForecast,
    ScopeForecast,
)
from precisionai.pace.protocol import Event, Record


class Run:
    """Drive one scope's records into a forecast; ``received_at`` is ``offset + elapsed``."""

    def __init__(self, sink, scope="train", total=1000.0, *, offset=1000.0, twice=False, **fields):
        self.sink = sink
        self.scope = scope
        self.total: float | None = total
        self.offset = offset
        self.twice = twice
        self.fields = fields
        self.seq = 0
        self.t = 0.0
        self.done = 0.0

    @property
    def now(self):
        return self.offset + self.t

    def _emit(self, event, completed, **extra):
        self.seq += 1
        record = Record(
            scope=self.scope,
            event=event,
            seq=self.seq,
            elapsed=self.t,
            completed=completed,
            total=None if event is Event.FAIL else self.total,
            **self.fields,
            **extra,
        )
        for _ in range(2 if self.twice else 1):
            self.sink.observe(record, self.now)
        return record

    def start(self, completed=0.0, **extra):
        self.done = completed
        return self._emit(Event.START, completed, unit="step", **extra)

    def tick(self, seconds, rate):
        self.t += seconds
        self.done += seconds * rate
        return self._emit(Event.UPDATE, self.done)

    def ticks(self, count, seconds, rate):
        for _ in range(count):
            self.tick(seconds, rate)

    def mark(self, elapsed, completed):
        self.t = elapsed
        self.done = completed
        return self._emit(Event.UPDATE, completed)

    def end(self):
        return self._emit(Event.END, self.total if self.total is not None else self.done)

    def fail(self):
        return self._emit(Event.FAIL, None)

    def restart(self, **extra):
        self.offset = self.now + 30.0
        self.seq = 0
        self.t = 0.0
        return self.start(**extra)


def _live_run(**kwargs):
    """A scope that has run 6 x 15 s at 2 units/s and is Live."""
    fc = ScopeForecast("train")
    run = Run(fc, **kwargs)
    run.start()
    run.ticks(6, 15.0, 2.0)
    return fc, run


def _rec(event: Event, seq: int, elapsed: float, completed: float | None = None, **kw: object) -> Record:
    return Record(scope="train", event=event, seq=seq, elapsed=elapsed, completed=completed, **kw)  # type: ignore[arg-type]


def test_default_config_holds_the_initial_thresholds() -> None:
    assert DEFAULT_CONFIG.min_observed_seconds == 60
    assert DEFAULT_CONFIG.min_intervals == 3
    assert DEFAULT_CONFIG.min_fraction == 0.01
    assert DEFAULT_CONFIG.heartbeat_seconds == 120
    assert DEFAULT_CONFIG.stale_factor == 3.0


def test_unknown_before_any_record() -> None:
    assert ScopeForecast("train").snapshot(0.0).state is ForecastState.UNKNOWN


def test_unknown_without_total() -> None:
    fc = ScopeForecast("train")
    fc.observe(_rec(Event.START, 1, 0.0, 0, unit="step"), 0.0)
    assert fc.snapshot(0.0).state is ForecastState.UNKNOWN


def test_total_that_disappears_makes_the_scope_unknown_again() -> None:
    fc, run = _live_run()
    run.total = None
    run.tick(15.0, 2.0)
    assert fc.snapshot(run.now).state is ForecastState.UNKNOWN


def test_learning_before_any_work_done() -> None:
    fc = ScopeForecast("train")
    fc.observe(_rec(Event.START, 1, 0.0, 0, total=100, unit="step"), 0.0)
    assert fc.snapshot(0.0).state is ForecastState.LEARNING


def test_ignores_other_scopes() -> None:
    fc = ScopeForecast("train")
    fc.observe(Record(scope="eval", event=Event.END, seq=1, elapsed=1.0, completed=1, total=1), 1.0)
    assert fc.snapshot(1.0).state is ForecastState.UNKNOWN


def test_live_needs_enough_observed_seconds() -> None:
    fc = ScopeForecast("train")
    run = Run(fc, total=10000.0)
    run.start()
    run.ticks(5, 10.0, 10.0)  # 50 s, 5 intervals, 5% done
    assert fc.snapshot(run.now).state is ForecastState.LEARNING
    run.tick(10.0, 10.0)  # 60 s
    assert fc.snapshot(run.now).state is ForecastState.LIVE


def test_live_needs_enough_intervals() -> None:
    fc = ScopeForecast("train")
    run = Run(fc)
    run.start()
    run.tick(100.0, 1.0)  # 100 s but one interval, 10% done
    run.tick(20.0, 1.0)
    assert fc.snapshot(run.now).state is ForecastState.LEARNING
    run.tick(20.0, 1.0)  # third interval
    assert fc.snapshot(run.now).state is ForecastState.LIVE


def test_live_needs_enough_of_the_total_done() -> None:
    fc = ScopeForecast("train")
    run = Run(fc, total=100000.0)
    run.start()
    run.ticks(40, 20.0, 1.0)  # 800 s, 0.8% done
    assert fc.snapshot(run.now).state is ForecastState.LEARNING
    run.ticks(6, 20.0, 1.0)  # 920 s, 0.92%
    assert fc.snapshot(run.now).state is ForecastState.LEARNING
    run.ticks(5, 20.0, 1.0)  # 1020 s, 1.02%
    assert fc.snapshot(run.now).state is ForecastState.LIVE


def test_estimates_remaining_from_rate() -> None:
    fc = ScopeForecast("train")
    fc.observe(_rec(Event.START, 1, 0.0, 0, total=100, unit="step"), 0.0)
    for seq, elapsed, done in ((2, 20.0, 5), (3, 40.0, 10), (4, 60.0, 15), (5, 80.0, 20)):
        fc.observe(_rec(Event.UPDATE, seq, elapsed, done, total=100), elapsed)
    snap = fc.snapshot(80.0)
    assert snap.state is ForecastState.LIVE
    assert snap.rate == pytest.approx(0.25)
    assert snap.remaining_seconds == pytest.approx(320.0)
    assert snap.unit == "step"


def test_remaining_is_never_negative() -> None:
    fc, run = _live_run(total=100.0)
    run.tick(15.0, 20.0)  # overshoots the total
    snap = fc.snapshot(run.now)
    assert snap.remaining_seconds == 0.0


def test_resume_measures_rate_from_resumed_point() -> None:
    fc = ScopeForecast("train")
    run = Run(fc, total=100.0)
    run.start(50.0, resumed=True)
    run.ticks(4, 20.0, 0.1)
    snap = fc.snapshot(run.now)
    assert snap.state is ForecastState.LIVE
    assert snap.rate == pytest.approx(0.1)
    assert snap.remaining_seconds == pytest.approx(420.0)


def test_resume_fraction_counts_only_work_done_in_the_segment() -> None:
    fc = ScopeForecast("train")
    run = Run(fc, total=100.0)
    run.start(50.0, resumed=True)
    run.ticks(4, 20.0, 0.001)  # 0.08 done of 100: below 1% although completed is 50%
    assert fc.snapshot(run.now).state is ForecastState.LEARNING


def test_end_and_fail_are_terminal() -> None:
    done = ScopeForecast("train")
    done.observe(_rec(Event.END, 5, 40.0, 100, total=100), 40.0)
    assert done.snapshot(41.0).state is ForecastState.COMPLETE
    assert done.snapshot(41.0).remaining_seconds == 0.0
    failed = ScopeForecast("train")
    failed.observe(_rec(Event.FAIL, 5, 40.0), 40.0)
    assert failed.snapshot(41.0).state is ForecastState.FAILED
    assert failed.snapshot(41.0).remaining_seconds is None


def test_records_after_the_end_are_ignored_until_a_new_start() -> None:
    fc, run = _live_run()
    run.end()
    run.tick(15.0, 2.0)
    assert fc.snapshot(run.now).state is ForecastState.COMPLETE
    run.restart()
    assert fc.snapshot(run.now).state is ForecastState.LEARNING


def test_a_late_update_does_not_resurrect_a_failed_scope() -> None:
    fc, run = _live_run()
    run.fail()
    run.tick(15.0, 2.0)
    assert fc.snapshot(run.now).state is ForecastState.FAILED


def test_an_outlier_pause_does_not_swing_the_estimate() -> None:
    fc, run = _live_run()
    before = fc.snapshot(run.now)
    assert before.rate == pytest.approx(2.0)
    run.tick(300.0, 0.1)  # an evaluation pause: 30 units in 300 s
    run.tick(15.0, 2.0)
    after = fc.snapshot(run.now)
    assert after.state is ForecastState.LIVE
    assert after.rate == pytest.approx(2.0, rel=0.01)


def test_a_sudden_burst_is_discounted_too() -> None:
    fc, run = _live_run()
    run.tick(15.0, 40.0)
    assert fc.snapshot(run.now).rate == pytest.approx(2.0, rel=0.01)


def test_a_sustained_slowdown_wins_within_a_few_intervals() -> None:
    fc, run = _live_run()
    run.ticks(2, 15.0, 0.4)  # five times slower, but only twice so far
    assert fc.snapshot(run.now).rate == pytest.approx(2.0)
    run.tick(15.0, 0.4)  # third slow interval in a row
    snap = fc.snapshot(run.now)
    assert snap.rate == pytest.approx(0.4)
    assert snap.remaining_seconds == pytest.approx(((run.total or 0.0) - run.done) / 0.4)


def test_a_blip_between_slow_intervals_resets_the_run() -> None:
    fc, run = _live_run()
    run.ticks(2, 15.0, 0.4)
    run.tick(15.0, 2.0)  # normal again: the slow ones were a blip
    run.ticks(2, 15.0, 0.4)
    assert fc.snapshot(run.now).rate == pytest.approx(2.0, rel=0.01)


def test_a_mild_slowdown_is_followed_by_the_moving_average() -> None:
    fc, run = _live_run()
    run.tick(15.0, 1.0)  # half speed is not an outlier: it is averaged in
    assert fc.snapshot(run.now).rate == pytest.approx(2.0 - 0.3)


def test_warmup_rates_are_averaged_plainly() -> None:
    fc = ScopeForecast("train")
    run = Run(fc)
    run.start()
    run.tick(20.0, 0.5)  # slow first interval (compilation)
    run.tick(20.0, 2.0)
    run.tick(20.0, 2.0)
    assert fc.snapshot(run.now).rate == pytest.approx((0.5 + 2.0 + 2.0) / 3)


def test_a_stalled_heartbeat_is_not_an_interval() -> None:
    fc, run = _live_run()
    rate = fc.snapshot(run.now).rate
    run.mark(120.0, run.done)  # heartbeat with no progress
    assert fc.snapshot(run.now).rate == rate
    run.mark(150.0, run.done + 60.0)  # next progress: interval spans the stall back to t=90
    assert fc.snapshot(run.now).rate == pytest.approx(2.0 + 0.3 * (1.0 - 2.0))


def test_intervals_without_elapsed_time_are_skipped() -> None:
    fc, run = _live_run()
    rate = fc.snapshot(run.now).rate
    run.mark(run.t, run.done + 50.0)  # same elapsed
    run.mark(run.t - 5.0, run.done + 50.0)  # elapsed going backwards
    assert fc.snapshot(run.now).rate == rate


def test_completed_going_backwards_restarts_the_baseline() -> None:
    fc, run = _live_run()
    run.mark(run.t + 15.0, run.done - 100.0)
    run.tick(15.0, 2.0)
    assert fc.snapshot(run.now).rate == pytest.approx(2.0)


def test_a_record_without_a_start_is_the_baseline() -> None:
    fc = ScopeForecast("train")
    run = Run(fc)
    run.mark(500.0, 200.0)  # joined the log mid-run
    assert fc.snapshot(run.now).state is ForecastState.LEARNING
    run.ticks(4, 20.0, 2.0)
    snap = fc.snapshot(run.now)
    assert snap.state is ForecastState.LIVE
    assert snap.rate == pytest.approx(2.0)


def test_stale_after_silence_and_back_on_the_next_record() -> None:
    fc, run = _live_run()
    live = fc.snapshot(run.now)
    assert fc.snapshot(run.now + 240.0).state is ForecastState.LIVE
    stale = fc.snapshot(run.now + 241.0)
    assert stale.state is ForecastState.STALE
    assert stale.remaining_seconds == live.remaining_seconds
    assert stale.rate == live.rate
    run.offset += 400.0
    run.tick(15.0, 2.0)
    assert fc.snapshot(run.now).state is ForecastState.LIVE


def test_stale_threshold_scales_with_the_median_record_gap() -> None:
    fc = ScopeForecast("train")
    run = Run(fc)
    run.start()
    run.ticks(6, 100.0, 1.0)  # a record every 100 s: limit is 3 x 100 s
    assert fc.snapshot(run.now + 299.0).state is ForecastState.LIVE
    assert fc.snapshot(run.now + 301.0).state is ForecastState.STALE


def test_learning_never_goes_stale() -> None:
    fc = ScopeForecast("train")
    run = Run(fc)
    run.start()
    assert fc.snapshot(run.now + 10000.0).state is ForecastState.LEARNING


def test_restart_begins_a_new_segment_without_mixing_rates() -> None:
    fc, run = _live_run()
    assert fc.reached_live
    run.restart()
    run.tick(15.0, 10.0)
    assert fc.snapshot(run.now).state is ForecastState.LEARNING
    run.ticks(5, 15.0, 10.0)
    snap = fc.snapshot(run.now)
    assert snap.state is ForecastState.LIVE
    assert snap.rate == pytest.approx(10.0)
    assert fc.elapsed == pytest.approx(90.0)
    assert fc.reached_live


def test_a_repeated_first_record_is_a_duplicate_not_a_restart() -> None:
    fc = ScopeForecast("train")
    record = _rec(Event.START, 1, 0.0, 0, total=100)
    fc.observe(record, 0.0)
    fc.observe(record, 0.0)
    for seq in range(2, 6):
        fc.observe(_rec(Event.UPDATE, seq, 20.0 * (seq - 1), 10.0 * (seq - 1), total=100), 20.0 * (seq - 1))
    assert fc.snapshot(80.0).rate == pytest.approx(0.5)


def test_duplicate_records_change_nothing() -> None:
    single, single_run = _live_run()
    doubled = ScopeForecast("train")
    doubled_run = Run(doubled, twice=True)
    doubled_run.start()
    doubled_run.ticks(6, 15.0, 2.0)
    assert doubled.snapshot(doubled_run.now) == single.snapshot(single_run.now)


def test_a_late_old_record_is_ignored() -> None:
    fc, run = _live_run()
    expected = fc.snapshot(run.now)
    fc.observe(_rec(Event.UPDATE, 3, 45.0, 90.0, total=1000.0), run.now)
    assert fc.snapshot(run.now) == expected


def test_a_changed_total_replaces_the_total_and_keeps_the_rate() -> None:
    fc, run = _live_run()
    assert fc.snapshot(run.now).remaining_seconds == pytest.approx((1000.0 - 180.0) / 2.0)
    run.total = 2000.0
    run.tick(15.0, 2.0)
    snap = fc.snapshot(run.now)
    assert snap.state is ForecastState.LIVE
    assert snap.total == 2000.0
    assert snap.rate == pytest.approx(2.0)
    assert snap.remaining_seconds == pytest.approx((2000.0 - 210.0) / 2.0)


def test_snapshot_never_raises_on_odd_numbers() -> None:
    fc = ScopeForecast("train")
    run = Run(fc, total=float("inf"))
    run.start()
    run.ticks(6, 15.0, 2.0)
    run.mark(float("nan"), float("nan"))
    assert fc.snapshot(float("nan")).state in set(ForecastState)
    assert fc.snapshot(run.now).remaining_seconds is None


def test_scope_metadata_comes_from_the_records() -> None:
    fc = ScopeForecast("child")
    run = Run(fc, scope="child", parent="job")
    run.start()
    assert fc.parent == "job"
    assert not fc.root


def test_custom_config_changes_the_thresholds() -> None:
    config = ForecastConfig(min_observed_seconds=10, min_intervals=1, min_fraction=0.0)
    fc = ScopeForecast("train", config=config)
    run = Run(fc)
    run.start()
    run.tick(10.0, 1.0)
    assert fc.snapshot(run.now).state is ForecastState.LIVE
    assert dataclasses.replace(config, min_intervals=2).min_intervals == 2


def test_simulated_training_run_with_an_evaluation_pause() -> None:
    rng = random.Random(7)
    total_steps = 5000
    fc = ScopeForecast("train")
    base = 5000.0
    fc.observe(Record(scope="train", event=Event.START, seq=1, elapsed=0.0, completed=0, total=total_steps), base)
    seq, now = 1, 0.0
    last_emit, last_done = 0.0, 0
    checkpoints = {}  # fraction done -> (estimate, time it was made)
    for step in range(1, total_steps + 1):
        now += max(rng.gauss(1 / 2.4, 0.1), 0.05)
        if step == 2500:
            now += 100.0  # evaluation
        due = now - last_emit >= 15.0 and step - last_done >= total_steps // 100
        if due or now - last_emit >= 120.0 or step == total_steps:
            seq += 1
            record = Record(scope="train", event=Event.UPDATE, seq=seq, elapsed=now, completed=step, total=total_steps)
            fc.observe(record, base + now)
            last_emit, last_done = now, step
            for fraction in (0.25, 0.6, 0.75):
                if step >= fraction * total_steps and fraction not in checkpoints:
                    checkpoints[fraction] = (fc.snapshot(base + now).remaining_seconds, now)
    assert set(checkpoints) == {0.25, 0.6, 0.75}
    for estimate, at in checkpoints.values():
        assert estimate == pytest.approx(now - at, rel=0.10)


def _plan(*phases: tuple[str, float]) -> Record:
    return Record(scope="plan", event=Event.PLAN, seq=1, elapsed=0.0, phases=tuple(phases))


def _live_root(job, scope="job", total=1000.0, **kwargs):
    run = Run(job, scope, total=total, root=True, **kwargs)
    run.start()
    run.ticks(6, 15.0, 2.0)
    return run


def _live_scope(job, scope, total=1000.0, **fields):
    run = Run(job, scope, total=total, **fields)
    run.start()
    run.ticks(6, 15.0, 2.0)
    return run


def _prep_then_train(job):
    """Prep done after 100 s, train 400 s in at 10 units/s with 16000 of 20000 left."""
    prep = Run(job, "prep", total=100.0)
    prep.start()
    prep.ticks(5, 20.0, 1.0)
    prep.end()
    train = Run(job, "train", total=20000.0, offset=1100.0)
    train.start()
    train.ticks(20, 20.0, 10.0)
    return prep, train


def test_job_without_records_is_unknown() -> None:
    est = JobForecast().snapshot(0.0)
    assert est.state is ForecastState.UNKNOWN
    assert est.remaining_seconds is None
    assert est.source is None
    assert est.scopes == {}


def test_job_estimate_is_frozen() -> None:
    est = JobForecast().snapshot(0.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        est.state = ForecastState.LIVE  # type: ignore[misc]


def test_root_scope_is_the_job_estimate() -> None:
    job = JobForecast()
    root = _live_root(job)
    _live_scope(job, "eval", total=5000.0, parent="job")
    est = job.snapshot(root.now)
    assert est.state is ForecastState.LIVE
    assert est.source == "root"
    assert not est.partial
    assert est.remaining_seconds == pytest.approx(410.0)
    assert set(est.scopes) == {"job", "eval"}
    assert est.scopes["job"].remaining_seconds == pytest.approx(410.0)


def test_a_root_that_is_still_learning_is_not_replaced_by_other_scopes() -> None:
    job = JobForecast()
    root = Run(job, "job", root=True)
    root.start()
    other = _live_scope(job, "other")
    est = job.snapshot(other.now)
    assert est.state is ForecastState.LEARNING
    assert est.remaining_seconds is None
    assert est.source is None


def test_concurrent_top_level_scopes_take_the_longest() -> None:
    job = JobForecast()
    _live_scope(job, "a", total=1000.0)
    b = _live_scope(job, "b", total=3000.0)
    est = job.snapshot(b.now)
    assert est.state is ForecastState.LIVE
    assert est.source == "concurrent"
    assert est.partial
    assert est.remaining_seconds == pytest.approx(1410.0)


def test_nested_scopes_are_never_added_on_top_of_their_parent() -> None:
    job = JobForecast()
    train = _live_scope(job, "train", total=1000.0)
    _live_scope(job, "step", total=9000.0, parent="train")
    est = job.snapshot(train.now)
    assert est.remaining_seconds == pytest.approx(410.0)
    assert est.source == "concurrent"


def test_sequential_scopes_without_a_plan_are_partial() -> None:
    job = JobForecast()
    prep = Run(job, "prep", total=10.0)
    prep.start()
    prep.end()
    train = _live_scope(job, "train", total=1000.0, offset=1100.0)
    est = job.snapshot(train.now)
    assert est.remaining_seconds == pytest.approx(410.0)
    assert est.partial
    assert est.state is ForecastState.LIVE


def test_plan_adds_up_finished_active_and_unstarted_phases() -> None:
    job = JobForecast()
    job.observe(_plan(("prep", 0.1), ("train", 0.8), ("eval", 0.1)), 1000.0)
    _, train = _prep_then_train(job)
    est = job.snapshot(train.now)
    seconds_per_weight = (100.0 + 400.0 + 1600.0) / 0.9
    assert est.source == "plan"
    assert est.state is ForecastState.LIVE
    assert not est.partial
    assert est.remaining_seconds == pytest.approx(1600.0 + 0.1 * seconds_per_weight)


def test_plan_finished_phases_contribute_zero() -> None:
    job = JobForecast()
    job.observe(_plan(("prep", 0.1), ("train", 0.9)), 1000.0)
    _, train = _prep_then_train(job)
    assert job.snapshot(train.now).remaining_seconds == pytest.approx(1600.0)


def test_plan_with_an_unmeasured_active_phase_falls_back() -> None:
    job = JobForecast()
    job.observe(_plan(("prep", 0.1), ("train", 0.9)), 1000.0)
    prep = Run(job, "prep", total=100.0)
    prep.start()
    prep.end()
    train = Run(job, "train", offset=1100.0)
    train.start()
    train.tick(30.0, 10.0)
    est = job.snapshot(train.now)
    assert est.state is ForecastState.LEARNING
    assert est.remaining_seconds is None


def test_plan_with_no_observed_phase_falls_back_to_concurrent() -> None:
    job = JobForecast()
    job.observe(_plan(("a", 1.0), ("b", 1.0)), 1000.0)
    other = _live_scope(job, "x")
    est = job.snapshot(other.now)
    assert est.source == "concurrent"
    assert est.partial
    assert est.remaining_seconds == pytest.approx(410.0)


def test_stale_plan_keeps_its_estimate() -> None:
    job = JobForecast()
    job.observe(_plan(("prep", 0.1), ("train", 0.8), ("eval", 0.1)), 1000.0)
    _, train = _prep_then_train(job)
    live = job.snapshot(train.now)
    stale = job.snapshot(train.now + 1000.0)
    assert stale.state is ForecastState.STALE
    assert stale.source == "plan"
    assert stale.remaining_seconds == live.remaining_seconds


def test_job_with_a_plan_is_not_complete_until_every_phase_ended() -> None:
    job = JobForecast()
    job.observe(_plan(("prep", 0.1), ("train", 0.8), ("eval", 0.1)), 1000.0)
    _, train = _prep_then_train(job)
    train.end()
    assert job.snapshot(train.now).state is not ForecastState.COMPLETE
    evaluation = Run(job, "eval", total=10.0, offset=train.now)
    evaluation.start()
    evaluation.end()
    est = job.snapshot(evaluation.now)
    assert est.state is ForecastState.COMPLETE
    assert est.remaining_seconds == 0.0


def test_bootstrap_counts_down_from_the_first_record_until_live() -> None:
    job = JobForecast(expected_seconds=1000.0)
    est = job.snapshot(50.0)
    assert est.state is ForecastState.BOOTSTRAP
    assert est.remaining_seconds == 1000.0
    assert est.source == "user"
    root = Run(job, "job", root=True, offset=500.0)
    root.start()
    assert job.snapshot(600.0).remaining_seconds == pytest.approx(900.0)
    assert job.snapshot(600.0).state is ForecastState.BOOTSTRAP
    assert job.snapshot(5000.0).remaining_seconds == 0.0


def test_the_user_estimate_is_discarded_for_good_once_live() -> None:
    job = JobForecast(expected_seconds=100000.0)
    root = Run(job, "job", root=True, offset=500.0)
    root.start()
    root.ticks(6, 15.0, 2.0)
    live = job.snapshot(root.now)
    assert live.state is ForecastState.LIVE
    assert live.source == "root"
    assert live.remaining_seconds == pytest.approx(410.0)
    stale = job.snapshot(root.now + 1000.0)
    assert stale.state is ForecastState.STALE
    assert stale.source == "root"
    assert stale.remaining_seconds == pytest.approx(410.0)
    root.restart()
    again = job.snapshot(root.now)
    assert again.state is ForecastState.LEARNING
    assert again.remaining_seconds is None
    assert again.source is None


def test_root_complete_completes_the_job() -> None:
    job = JobForecast(expected_seconds=500.0)
    root = _live_root(job)
    _live_scope(job, "eval", parent="job")
    root.end()
    est = job.snapshot(root.now)
    assert est.state is ForecastState.COMPLETE
    assert est.remaining_seconds == 0.0
    assert est.scopes["eval"].state is not ForecastState.COMPLETE


def test_root_failure_fails_the_job() -> None:
    job = JobForecast()
    root = _live_root(job)
    root.fail()
    est = job.snapshot(root.now)
    assert est.state is ForecastState.FAILED
    assert est.remaining_seconds is None


def test_a_failed_nested_scope_does_not_fail_the_job() -> None:
    job = JobForecast()
    root = _live_root(job)
    child = Run(job, "eval", parent="job")
    child.start()
    child.fail()
    assert job.snapshot(root.now).state is ForecastState.LIVE


def test_without_a_root_every_top_level_scope_must_end() -> None:
    job = JobForecast()
    a = _live_scope(job, "a")
    b = _live_scope(job, "b")
    a.end()
    assert job.snapshot(b.now).state is ForecastState.LIVE
    b.end()
    assert job.snapshot(b.now).state is ForecastState.COMPLETE


def test_without_a_root_any_failed_top_level_scope_fails_the_job() -> None:
    job = JobForecast()
    _live_scope(job, "a")
    b = _live_scope(job, "b")
    b.fail()
    assert job.snapshot(b.now).state is ForecastState.FAILED


def test_job_ignores_duplicate_records_and_plans() -> None:
    once = JobForecast()
    root = _live_root(once)
    twice = JobForecast()
    twice_root = _live_root(twice, twice=True)
    plan = _plan(("job", 1.0))
    twice.observe(plan, 1000.0)
    twice.observe(plan, 1000.0)
    assert twice.snapshot(twice_root.now) == once.snapshot(root.now)


def test_a_plan_without_phases_is_ignored() -> None:
    job = JobForecast()
    job.observe(_plan(), 0.0)
    assert job.snapshot(0.0).state is ForecastState.UNKNOWN


def test_job_config_reaches_every_scope() -> None:
    job = JobForecast(config=ForecastConfig(min_observed_seconds=10, min_intervals=1, min_fraction=0.0))
    run = Run(job, "job", root=True)
    run.start()
    run.tick(10.0, 1.0)
    assert job.snapshot(run.now).state is ForecastState.LIVE
