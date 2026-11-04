# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.pace.forecast.

Every test injects its clock: records carry ``elapsed`` and are delivered with an
explicit ``received_at``; nothing sleeps.
"""

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

    def __init__(self, sink, scope="train", total=1000.0, *, offset=1000.0, twice=False, run_id=None, **fields):
        """With ``run_id``, the records are pace:2 ones: the run id on each, a sequence number only on an update."""
        self.sink = sink
        self.run_id = run_id
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
            seq=None if self.run_id and event is not Event.UPDATE else self.seq,
            elapsed=self.t,
            completed=completed,
            total=None if event is Event.FAIL else self.total,
            **({"run": self.run_id} if self.run_id else {}),
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


def test_without_a_root_or_plan_the_job_is_never_called_complete() -> None:
    job = JobForecast()
    a = _live_scope(job, "a")
    b = _live_scope(job, "b")
    a.end()
    assert job.snapshot(b.now).state is ForecastState.LIVE
    b.end()
    # More unplanned phases may follow, so only a root or a finished plan proves the end.
    est = job.snapshot(b.now)
    assert est.state is ForecastState.UNKNOWN
    assert est.remaining_seconds is None


def test_root_decides_alone_when_another_top_level_scope_fails() -> None:
    job = JobForecast()
    root = _live_root(job)
    other = Run(job, "download")
    other.start()
    other.fail()
    assert job.snapshot(root.now).state is ForecastState.LIVE


def test_an_older_plan_read_late_does_not_replace_a_newer_one() -> None:
    job = JobForecast()
    newer = Record(scope="plan", event=Event.PLAN, seq=2, elapsed=0.0, phases=(("prep", 0.1), ("train", 0.9)))
    older = Record(scope="plan", event=Event.PLAN, seq=1, elapsed=0.0, phases=(("prep", 0.5), ("train", 0.5)))
    job.observe(newer, 1000.0)
    job.observe(older, 1000.0)
    _, train = _prep_then_train(job)
    assert job.snapshot(train.now).remaining_seconds == pytest.approx(1600.0)


def test_a_stale_concurrent_scope_makes_the_job_stale() -> None:
    job = JobForecast()
    _live_scope(job, "short", total=300.0)
    longer = _live_scope(job, "long", total=5000.0, offset=1400.0)
    est = job.snapshot(longer.now)
    assert est.source == "concurrent"
    assert est.state is ForecastState.STALE
    assert est.remaining_seconds == pytest.approx((5000.0 - 180.0) / 2.0)


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


def test_live_estimate_counts_down_between_records() -> None:
    fc, run = _live_run()
    at_record = fc.snapshot(run.now).remaining_seconds
    assert at_record is not None
    later = fc.snapshot(run.now + 10.0)
    assert later.state is ForecastState.LIVE
    assert later.remaining_seconds == pytest.approx(at_record - 10.0)
    assert fc.snapshot(run.now + 10_000.0).remaining_seconds == at_record


def test_a_start_from_the_same_run_read_late_is_a_duplicate() -> None:
    fc = ScopeForecast("train")
    run = Run(fc)
    first = run.start(run="aaaa1111")
    run.ticks(6, 15.0, 2.0)
    live = fc.snapshot(run.now)
    fc.observe(first, run.now)
    assert fc.snapshot(run.now) == live


def test_a_start_from_a_new_run_is_a_restart() -> None:
    fc = ScopeForecast("train")
    run = Run(fc)
    run.start(run="aaaa1111")
    run.ticks(6, 15.0, 2.0)
    run.restart(run="bbbb2222")
    assert fc.snapshot(run.now).state is ForecastState.LEARNING


def test_a_phase_borrows_the_estimate_of_the_loop_inside_it() -> None:
    job = JobForecast()
    job.observe(_plan(("prep", 0.1), ("train", 0.9)), 1000.0)
    prep = Run(job, "prep", total=1.0)
    prep.start()
    prep.mark(100.0, 1.0)
    prep.end()
    phase = Run(job, "train", total=1.0, offset=1100.0)
    phase.start()
    loop = _live_scope(job, "batches", total=1000.0, offset=1100.0, parent="train")
    est = job.snapshot(loop.now)
    assert est.source == "plan"
    assert est.state is ForecastState.LIVE
    assert est.remaining_seconds == pytest.approx(410.0)


def test_a_top_level_phase_without_a_plan_borrows_its_child_estimate() -> None:
    job = JobForecast()
    phase = Run(job, "train", total=1.0)
    phase.start()
    loop = _live_scope(job, "batches", total=1000.0, parent="train")
    est = job.snapshot(loop.now)
    assert est.source == "concurrent"
    assert est.remaining_seconds == pytest.approx(410.0)


def test_child_estimates_reach_the_top_through_nested_phases() -> None:
    job = JobForecast()
    outer = Run(job, "outer", total=1.0)
    outer.start()
    inner = Run(job, "inner", total=1.0, parent="outer")
    inner.start()
    loop = _live_scope(job, "batches", total=1000.0, parent="inner")
    est = job.snapshot(loop.now)
    assert est.state is ForecastState.LIVE
    assert est.remaining_seconds == pytest.approx(410.0)


def test_a_restart_forgets_the_previous_run_hierarchy() -> None:
    fc = ScopeForecast("train")
    run = Run(fc, parent="job", root=True)
    run.start(run="aaaa1111")
    assert fc.parent == "job"
    assert fc.root
    fresh = Run(fc)
    fresh.start(run="bbbb2222")
    assert fc.parent is None
    assert not fc.root


def _epoch(job: JobForecast, index: int, offset: float, parent: str) -> Run:
    run = Run(job, f"train-{index}", total=1000.0, offset=offset, parent=parent)
    run.start()
    run.ticks(6, 15.0, 2.0)
    return run


def test_a_counted_parent_adds_the_children_still_to_come() -> None:
    job = JobForecast()
    epochs = Run(job, "epochs", total=3.0, root=True)
    epochs.start()
    first = _epoch(job, 0, 1000.0, "epochs")
    est = job.snapshot(first.now)
    # 410 s left in this epoch, then two more epochs of 90 + 410 = 500 s each.
    assert est.source == "root"
    assert not est.partial
    assert est.remaining_seconds == pytest.approx(410.0 + 2 * 500.0)


def test_between_children_a_counted_parent_uses_the_finished_ones() -> None:
    job = JobForecast()
    epochs = Run(job, "epochs", total=3.0, root=True)
    epochs.start()
    first = _epoch(job, 0, 1000.0, "epochs")
    first.t = 500.0
    first.end()
    epochs.t = 500.0
    epochs.mark(500.0, 1.0)
    est = job.snapshot(first.now)
    assert est.state is ForecastState.LIVE
    assert est.remaining_seconds == pytest.approx(2 * 500.0)


def test_a_phase_with_several_children_is_only_a_lower_bound() -> None:
    job = JobForecast()
    job.observe(_plan(("train", 1.0)), 1000.0)
    phase = Run(job, "train", total=1.0)
    phase.start()
    loop = _epoch(job, 0, 1000.0, "train")
    est = job.snapshot(loop.now)
    assert est.source == "plan"
    assert est.partial
    assert est.remaining_seconds == pytest.approx(410.0)


def test_elapsed_counts_on_from_the_latest_record_until_the_scope_ends() -> None:
    fc = ScopeForecast("train")
    run = Run(fc, total=1.0)
    run.start()
    assert fc.elapsed_at(run.now + 40.0) == pytest.approx(40.0)
    run.t = 50.0
    run.end()
    assert fc.elapsed_at(run.now + 40.0) == pytest.approx(50.0)


def test_seconds_per_unit_counts_only_this_segments_units() -> None:
    fc = ScopeForecast("epochs")
    run = Run(fc, "epochs", total=10.0)
    run.start(completed=7.0, resumed=True)
    assert fc.seconds_per_unit() is None
    run.mark(100.0, 9.0)
    assert fc.seconds_per_unit() == pytest.approx(50.0)


def test_a_plan_times_unstarted_phases_from_a_running_phase_with_a_loop_inside() -> None:
    # Train is a phase around one loop; evaluate (weight 0.1) has not started. The
    # phase's own records say e=0, so its elapsed time has to come from the clock.
    job = JobForecast()
    job.observe(_plan(("train", 0.9), ("evaluate", 0.1)), 1000.0)
    phase = Run(job, "train", total=1.0)
    phase.start()
    loop = Run(job, "batches", total=1000.0, parent="train")
    loop.start()
    loop.ticks(30, 15.0, 2.0)  # 450 s in, 900 done, 50 s left
    est = job.snapshot(loop.now)
    assert est.source == "plan"
    assert est.remaining_seconds == pytest.approx(50.0 + 0.1 * 500.0 / 0.9)


def test_a_counted_parent_times_units_that_hold_several_children() -> None:
    # Each epoch is a 500 s training loop and a 100 s validation loop.
    job = JobForecast()
    epochs = Run(job, "epochs", total=3.0, root=True)
    epochs.start()
    train = _epoch(job, 0, 1000.0, "epochs")
    train.t = 500.0
    train.end()
    val = Run(job, "val-0", total=100.0, offset=1500.0, parent="epochs")
    val.start()
    val.t = 100.0
    val.end()
    epochs.mark(600.0, 1.0)
    second = _epoch(job, 1, 1600.0, "epochs")
    est = job.snapshot(second.now)
    # 90 s into the second epoch: 510 s of it left, then one more epoch of 600 s.
    assert est.source == "root"
    assert not est.partial
    assert est.remaining_seconds == pytest.approx(510.0 + 600.0)


def test_a_counted_parent_ignores_a_short_finished_child_once_it_has_a_pace() -> None:
    # A one-off step child (saving, say) that took no time must not make the units
    # still to come look free.
    job = JobForecast()
    root = Run(job, "job", total=40.0, root=True)
    root.start()
    root.mark(100.0, 10.0)
    save = Run(job, "save", total=1.0, offset=1100.0, parent="job")
    save.start()
    save.end()
    est = job.snapshot(1100.0)
    assert est.remaining_seconds == pytest.approx(30 * 10.0)


def test_a_stale_borrowed_estimate_stays_put() -> None:
    # The loop inside a phase stops reporting: the plan's estimate goes Stale and keeps
    # the number it had, however long the reader keeps asking.
    job = JobForecast()
    job.observe(_plan(("train", 0.9), ("evaluate", 0.1)), 1000.0)
    Run(job, "train", total=1.0).start()
    loop = Run(job, "batches", total=1000.0, parent="train")
    loop.start()
    loop.ticks(30, 15.0, 2.0)
    first = job.snapshot(loop.now + 1000.0)
    later = job.snapshot(loop.now + 5000.0)
    assert first.state is ForecastState.STALE
    assert later.remaining_seconds == pytest.approx(first.remaining_seconds)
    assert later.remaining_seconds == pytest.approx(50.0 + 0.1 * 500.0 / 0.9)


def test_a_counted_parent_with_a_stale_child_stays_put() -> None:
    job = JobForecast()
    epochs = Run(job, "epochs", total=3.0, root=True)
    epochs.start()
    first = _epoch(job, 0, 1000.0, "epochs")
    stale = job.snapshot(first.now + 1000.0)
    assert stale.state is ForecastState.STALE
    assert stale.remaining_seconds == pytest.approx(410.0 + 2 * 500.0)
    assert job.snapshot(first.now + 5000.0).remaining_seconds == pytest.approx(stale.remaining_seconds)


def test_between_children_a_silent_job_goes_stale_instead_of_counting_to_zero() -> None:
    job = JobForecast()
    epochs = Run(job, "epochs", total=3.0, root=True)
    epochs.start()
    first = _epoch(job, 0, 1000.0, "epochs")
    first.t = 500.0
    first.end()
    epochs.mark(500.0, 1.0)
    # Within the parent's Stale limit (3 x its 500 s gap) the gap counts down; after it
    # the estimate holds.
    assert job.snapshot(1600.0).state is ForecastState.LIVE
    assert job.snapshot(1600.0).remaining_seconds == pytest.approx(900.0)
    stale = job.snapshot(1500.0 + 1501.0)
    assert stale.state is ForecastState.STALE
    assert stale.remaining_seconds == pytest.approx(1000.0)
    assert job.snapshot(1500.0 + 10_000.0).remaining_seconds == pytest.approx(1000.0)


def test_a_child_without_an_estimate_still_keeps_the_job_from_going_stale() -> None:
    # Validation has no total, so it never has an estimate, but its lines show the job
    # is alive between two training loops.
    job = JobForecast()
    epochs = Run(job, "epochs", total=3.0, root=True)
    epochs.start()
    train = _epoch(job, 0, 1000.0, "epochs")
    train.t = 300.0
    train.end()
    val = Run(job, "val-0", offset=1300.0, parent="epochs")
    val.total = None
    val.start()
    val.ticks(3, 100.0, 1.0)
    assert job.snapshot(val.now + 60.0).state is ForecastState.LIVE


def test_between_children_the_parents_own_stale_limit_applies() -> None:
    # Epochs report every 600 s, so a 300 s gap with no scope (a checkpoint save) is normal.
    job = JobForecast()
    epochs = Run(job, "epochs", total=3.0, root=True)
    epochs.start()
    first = _epoch(job, 0, 1000.0, "epochs")
    first.t = 300.0
    first.end()
    epochs.mark(600.0, 1.0)
    second = _epoch(job, 1, 1600.0, "epochs")
    second.t = 300.0
    second.end()
    assert job.snapshot(1900.0 + 241.0).state is ForecastState.LIVE
    assert job.snapshot(1900.0 + 1801.0).state is ForecastState.STALE


def test_a_large_parent_without_a_pace_has_no_estimate_from_its_children() -> None:
    # 200 epochs: the parent writes an advance only every 1% (2 epochs) or heartbeat, so
    # its silence since the start does not mean the first epoch is still running.
    job = JobForecast()
    epochs = Run(job, "epochs", total=200.0, root=True)
    epochs.start()
    second = _epoch(job, 1, 1500.0, "epochs")
    assert job.snapshot(second.now).remaining_seconds is None


def test_a_step_root_is_not_timed_by_a_child_before_its_first_update() -> None:
    # A 60 s model load inside a root over 5000 steps is not the length of a step.
    job = JobForecast()
    root = Run(job, "train", total=5000.0, root=True)
    root.start()
    load = Run(job, "load", total=1.0, parent="train")
    load.start()
    load.t = 60.0
    load.end()
    assert job.snapshot(1065.0).remaining_seconds is None
    # Once the root reports progress, its own pace is used.
    root.mark(100.0, 80.0)
    assert job.snapshot(1100.0).remaining_seconds == pytest.approx(4920 * 1.25)


def test_seconds_per_unit_restarts_when_the_count_goes_backwards() -> None:
    fc = ScopeForecast("epochs")
    run = Run(fc, "epochs", total=10.0)
    run.start()
    run.mark(300.0, 3.0)
    run.mark(301.0, 1.0)
    run.mark(401.0, 2.0)
    assert fc.seconds_per_unit() == pytest.approx(100.0)


# --- pace:2 records: a run id on every record, ordered by elapsed time ---

RUN_A = "aaaa1111"
RUN_B = "bbbb2222"


class Collect:
    """A sink that keeps the records, to replay them as a reader would."""

    def __init__(self):
        self.records = []

    def observe(self, record, received_at):
        self.records.append((record, received_at))


def test_pace2_records_read_twice_change_nothing() -> None:
    once, twice = ScopeForecast("train"), ScopeForecast("train")
    for fc, double in ((once, False), (twice, True)):
        run = Run(fc, run_id=RUN_A, twice=double)
        run.start()
        run.ticks(6, 15.0, 2.0)
    assert twice.snapshot(run.now) == once.snapshot(run.now)
    assert twice.snapshot(run.now).state is ForecastState.LIVE


def test_two_streams_read_one_after_the_other_give_the_estimate_of_one() -> None:
    # A reader that drains stdout before stderr sees every start again after the updates.
    sink = Collect()
    run = Run(sink, run_id=RUN_A, root=True)
    run.start()
    run.ticks(8, 15.0, 2.0)
    one, two = JobForecast(), JobForecast()
    for record, received in sink.records:
        one.observe(record, received)
        two.observe(record, received)
    taken = [two.observe(record, run.now) for record, _ in sink.records]
    assert taken == [False] * len(sink.records)
    assert two.snapshot(run.now) == one.snapshot(run.now)
    assert one.snapshot(run.now).state is ForecastState.LIVE


def test_a_late_start_of_the_run_already_seen_is_not_a_restart() -> None:
    # The reader joined mid-run: it read updates first, then the start from the other stream.
    fc = ScopeForecast("train")
    run = Run(fc, run_id=RUN_A)
    late_start = Record(scope="train", event=Event.START, completed=0.0, total=1000.0, unit="step", run=RUN_A)
    run.ticks(6, 15.0, 2.0)
    live = fc.snapshot(run.now)
    assert live.state is ForecastState.LIVE
    assert fc.observe(late_start, run.now) is False
    assert fc.snapshot(run.now) == live


def test_a_new_run_after_joining_mid_run_is_a_restart() -> None:
    fc = ScopeForecast("train")
    run = Run(fc, run_id=RUN_A)
    run.ticks(6, 15.0, 2.0)
    old_update = run.tick(15.0, 2.0)
    run.run_id = RUN_B
    run.restart(completed=40.0, resumed=True)
    restarted = fc.snapshot(run.now)
    assert restarted.state is ForecastState.LEARNING
    assert restarted.completed == 40.0
    assert fc.elapsed == 0.0
    # A copy of the earlier run's update, read late from the other stream, changes nothing.
    assert fc.observe(old_update, run.now) is False
    assert fc.snapshot(run.now) == restarted


def test_a_new_run_id_on_an_update_also_begins_a_new_segment() -> None:
    # The reader may miss the restarted run's start altogether.
    fc, run = _live_run(run_id=RUN_A)
    run.run_id = RUN_B
    run.offset, run.t = run.now + 30.0, 10.0
    run.mark(10.0, 5.0)
    assert fc.snapshot(run.now).state is ForecastState.LEARNING
    assert fc.snapshot(run.now).completed == 5.0


def test_within_a_run_a_start_comes_first_and_an_end_last() -> None:
    fc = ScopeForecast("train")
    for event, seq in ((Event.START, None), (Event.UPDATE, 2), (Event.UPDATE, 3), (Event.END, None)):
        record = Record(scope="train", event=event, seq=seq, completed=1.0, total=1.0, run=RUN_A)
        assert fc.observe(record, 0.0)
    assert fc.snapshot(0.0).state is ForecastState.COMPLETE
    late_update = Record(scope="train", event=Event.UPDATE, seq=4, completed=1.0, total=1.0, run=RUN_A)
    assert fc.observe(late_update, 0.0) is False


def test_a_step_is_a_start_and_an_end_at_once() -> None:
    job = JobForecast()
    step = Record(scope="load", event=Event.STEP, elapsed=12.0, completed=1.0, total=1.0, parent="job", run=RUN_A)
    assert job.observe(step, 1000.0)
    assert job.observe(step, 1000.0) is False
    forecast = job.snapshot(1000.0).scopes["load"]
    assert forecast.state is ForecastState.COMPLETE
    assert (forecast.completed, forecast.total) == (1.0, 1.0)


def test_a_plan_times_unstarted_phases_from_a_step() -> None:
    job = JobForecast()
    job.observe(Record(scope="plan", event=Event.PLAN, seq=1, phases=(("prep", 0.5), ("train", 0.5)), run=RUN_A), 0.0)
    job.observe(Record(scope="prep", event=Event.STEP, elapsed=100.0, completed=1.0, total=1.0, run=RUN_A), 100.0)
    estimate = job.snapshot(100.0)
    assert estimate.source == "plan"
    assert estimate.remaining_seconds == pytest.approx(100.0)


def test_a_plan_from_a_restarted_run_replaces_the_one_before_it() -> None:
    job = JobForecast()
    older = Record(scope="plan", event=Event.PLAN, seq=3, phases=(("a", 1.0),), run=RUN_A)
    newer = Record(scope="plan", event=Event.PLAN, seq=1, phases=(("b", 1.0),), run=RUN_B)
    assert job.observe(older, 0.0)
    assert job.observe(newer, 1.0)
    assert job.observe(older, 2.0) is False
    assert job.observe(newer, 2.0) is False
    job.observe(Record(scope="b", event=Event.STEP, elapsed=5.0, completed=1.0, total=1.0, run=RUN_B), 3.0)
    assert job.snapshot(3.0).state is ForecastState.COMPLETE


def test_observe_says_whether_a_record_was_taken() -> None:
    fc = ScopeForecast("train")
    assert (
        fc.observe(Record(scope="other", event=Event.UPDATE, seq=2, completed=1.0, total=2.0, run=RUN_A), 0.0) is False
    )
    assert fc.observe(Record(scope="train", event=Event.PLAN, seq=1, phases=(("a", 1.0),), run=RUN_A), 0.0) is False
    assert JobForecast().observe(Record(scope="plan", event=Event.PLAN, seq=1, run=RUN_A), 0.0) is False
