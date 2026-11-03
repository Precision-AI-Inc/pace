# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Lightning callback, driven by real tiny CPU runs."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import pytest

from precisionai.pace import Event, Record, parse_line
from precisionai.pace import lightning as pace_lightning
from precisionai.pace.capture import capture_logging
from precisionai.pace.core import Logger
from precisionai.pace.lightning import PaceCallback
from tests.conftest import FakeClock, FakeStream

torch = pytest.importorskip("torch")
pl = pytest.importorskip("lightning.pytorch")

BATCHES = 4
EPOCHS = 2


class TinyModel(pl.LightningModule):
    def __init__(self, fail_at: int | None = None) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(4, 1)
        self.fail_at = fail_at

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        if self.fail_at is not None and self.global_step >= self.fail_at:
            raise RuntimeError("boom")
        x, y = batch
        return torch.nn.functional.mse_loss(self.layer(x), y)

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        x, y = batch
        self.log("val_loss", torch.nn.functional.mse_loss(self.layer(x), y))

    def test_step(self, batch: Any, batch_idx: int) -> None:
        self.validation_step(batch, batch_idx)

    def predict_step(self, batch: Any, batch_idx: int) -> Any:
        return self.layer(batch[0])

    def configure_optimizers(self) -> Any:
        return torch.optim.SGD(self.parameters(), lr=0.01)


def loader(batches: int = BATCHES) -> Any:
    data = torch.utils.data.TensorDataset(torch.randn(batches * 2, 4), torch.randn(batches * 2, 1))
    return torch.utils.data.DataLoader(data, batch_size=2)


def trainer(log: Logger, tmp_path: Path, **options: Any) -> Any:
    options.setdefault("max_epochs", EPOCHS)
    options.setdefault("callbacks", [PaceCallback(logger=log)])
    options.setdefault("enable_checkpointing", False)
    return pl.Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=str(tmp_path),
        **options,
    )


def records(stream: FakeStream) -> list[Record]:
    return [record for record in map(parse_line, stream.lines()) if record is not None]


def of(found: list[Record], scope: str) -> list[Record]:
    return [record for record in found if record.scope == scope]


@pytest.fixture
def quiet(log: Logger) -> Logger:
    log.configure(min_interval=0)
    return log


def test_fit_writes_one_root_scope_with_the_estimated_steps(quiet: Logger, stream: FakeStream, tmp_path: Path) -> None:
    trainer(quiet, tmp_path).fit(TinyModel(), loader())
    found = records(stream)
    train = of(found, "train")
    assert train[0].event is Event.START
    assert train[0].root
    assert train[0].unit == "step"
    assert train[0].completed == 0
    assert train[0].resumed is False
    assert train[-1].event is Event.END
    assert train[-1].completed == BATCHES * EPOCHS
    counts = [r.completed or 0 for r in train]
    assert counts == sorted(counts)
    assert {r.total for r in train} == {BATCHES * EPOCHS}
    assert {r.scope for r in found if r.root} == {"train"}


def test_validation_is_a_child_of_the_fit_scope(quiet: Logger, stream: FakeStream, tmp_path: Path) -> None:
    trainer(quiet, tmp_path).fit(TinyModel(), loader(), loader(3))
    found = records(stream)
    first = of(found, "validation")
    assert first[0].event is Event.START
    assert first[0].parent == "train"
    assert first[0].unit == "batch"
    assert first[0].total == 3
    assert first[-1].event is Event.END
    assert first[-1].completed == 3
    assert not first[0].root
    assert {r.scope for r in found if r.parent == "train"} == {"validation", "validation-2"}


def test_the_sanity_check_is_not_reported(quiet: Logger, stream: FakeStream, tmp_path: Path) -> None:
    runner = trainer(quiet, tmp_path, max_epochs=1, num_sanity_val_steps=2, limit_val_batches=1)
    runner.fit(TinyModel(), loader(), loader())
    starts = [r for r in records(stream) if r.event is Event.START]
    assert [r.scope for r in starts] == ["train", "validation"]


def test_an_unknown_total_is_written_as_unknown(quiet: Logger, stream: FakeStream, tmp_path: Path) -> None:
    runner = trainer(quiet, tmp_path, max_epochs=-1, max_steps=-1, max_time={"seconds": 1})
    runner.fit(TinyModel(), loader())
    train = of(records(stream), "train")
    assert train[0].total is None
    assert train[-1].event is Event.END
    assert (train[-1].completed or 0) >= BATCHES


def test_resume_from_a_checkpoint_starts_at_the_restored_step(
    quiet: Logger, stream: FakeStream, tmp_path: Path
) -> None:
    first = trainer(quiet, tmp_path, max_epochs=1, callbacks=[])
    first.fit(TinyModel(), loader())
    checkpoint = str(tmp_path / "resume.ckpt")
    first.save_checkpoint(checkpoint)
    assert first.global_step == BATCHES

    trainer(quiet, tmp_path).fit(TinyModel(), loader(), ckpt_path=checkpoint)
    train = of(records(stream), "train")
    assert train[0].resumed is True
    assert train[0].completed == BATCHES
    assert train[0].total == BATCHES * EPOCHS
    assert train[-1].completed == BATCHES * EPOCHS


def test_an_exception_writes_fail(quiet: Logger, stream: FakeStream, tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="boom"):
        trainer(quiet, tmp_path).fit(TinyModel(fail_at=2), loader())
    train = of(records(stream), "train")
    assert train[-1].event is Event.FAIL
    assert train[-1].completed == 2
    assert not [r for r in train if r.event is Event.END]


def test_a_failure_inside_validation_fails_both_scopes(quiet: Logger, stream: FakeStream, tmp_path: Path) -> None:
    class BrokenValidation(TinyModel):
        def validation_step(self, batch: Any, batch_idx: int) -> None:
            raise RuntimeError("bad batch")

    with pytest.raises(RuntimeError, match="bad batch"):
        trainer(quiet, tmp_path, num_sanity_val_steps=0).fit(BrokenValidation(), loader(), loader())
    found = records(stream)
    assert of(found, "validation")[-1].event is Event.FAIL
    assert of(found, "train")[-1].event is Event.FAIL


def test_a_non_zero_rank_writes_nothing(
    quiet: Logger, stream: FakeStream, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PACE_ALL_RANKS", "1")
    monkeypatch.setattr(pl.Trainer, "is_global_zero", property(lambda _: False))
    trainer(quiet, tmp_path).fit(TinyModel(), loader(), loader())
    assert stream.getvalue() == ""


def test_a_standalone_pass_is_top_level_but_not_the_root(quiet: Logger, stream: FakeStream, tmp_path: Path) -> None:
    runner = trainer(quiet, tmp_path)
    runner.validate(TinyModel(), loader(3))
    runner.test(TinyModel(), loader(2))
    found = records(stream)
    validation, test = of(found, "validation"), of(found, "test")
    assert not validation[0].root
    assert validation[0].parent is None
    assert validation[-1].event is Event.END
    assert validation[-1].completed == 3
    assert not test[0].root
    assert test[-1].completed == 2
    assert test[-1].total == 2


def test_predict_is_reported_by_batch(quiet: Logger, stream: FakeStream, tmp_path: Path) -> None:
    trainer(quiet, tmp_path).predict(TinyModel(), loader(3))
    predict = of(records(stream), "predict")
    assert predict[0].total == 3
    assert predict[-1].event is Event.END
    assert predict[-1].completed == 3


def test_fit_then_test_keeps_a_single_root(quiet: Logger, stream: FakeStream, tmp_path: Path) -> None:
    runner = trainer(quiet, tmp_path)
    runner.fit(TinyModel(), loader())
    runner.test(TinyModel(), loader())
    assert {r.scope for r in records(stream) if r.root} == {"train"}


def test_a_description_and_scope_id_can_be_chosen(quiet: Logger, stream: FakeStream, tmp_path: Path) -> None:
    callback = PaceCallback("Finetune", logger=quiet, scope="ft", root=False)
    trainer(quiet, tmp_path, callbacks=[callback]).fit(TinyModel(), loader())
    train = of(records(stream), "ft")
    assert train[0].label == "Finetune"
    assert not train[0].root


def test_without_lightning_installed_the_callback_says_how_to_install_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pace_lightning, "_LIGHTNING_AVAILABLE", False)
    with pytest.raises(ImportError, match=r"precisionai-pace\[lightning\]"):
        PaceCallback()


@pytest.mark.parametrize(
    ("counts", "expected"),
    [([3, 2], 5.0), (4, 4.0), ([float("inf")], None), ([0], None), (None, None), ([None], None)],
)
def test_the_batch_total_is_unknown_when_lightning_cannot_say(counts: Any, expected: float | None) -> None:
    assert pace_lightning._batches(counts) == expected


@pytest.fixture
def lightning_loggers_as_in_a_fresh_process() -> Iterator[None]:
    """Give Lightning's loggers the console handler and no propagation they have when no root handler existed."""
    names = ("lightning", "lightning.pytorch", "lightning.fabric")
    loggers = [logging.getLogger(name) for name in names]
    saved = [(list(each.handlers), each.propagate) for each in loggers]
    root = logging.getLogger()
    root_saved = list(root.handlers), root.level
    for each in loggers:
        each.handlers[:] = [logging.StreamHandler()]
        each.propagate = False
    yield
    for each, (handlers, propagate) in zip(loggers, saved):
        each.handlers[:] = handlers
        each.propagate = propagate
    root.handlers[:] = root_saved[0]
    root.setLevel(root_saved[1])


def test_capture_logging_takes_over_lightnings_own_log_lines(
    quiet: Logger,
    stream: FakeStream,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    lightning_loggers_as_in_a_fresh_process: None,
) -> None:
    capture_logging("INFO", target=quiet, take_over="lightning")
    trainer(quiet, tmp_path, max_epochs=1).fit(TinyModel(), loader())
    lines = [line for line in stream.lines() if "[pace:1" not in line]
    assert any("Trainer.fit" in line or "GPU available" in line for line in lines)
    assert len(lines) == len(set(lines))
    captured = capsys.readouterr()
    assert captured.err == ""
    assert "GPU available" not in captured.out


def test_a_validation_pass_is_timed_from_before_its_first_batch(
    quiet: Logger, stream: FakeStream, clock: FakeClock
) -> None:
    # Lightning opens the scope in on_validation_start, before batch 1, and counts a batch in
    # on_validation_batch_end, so batch 1's time is inside the pass. The Transformers callback
    # has to start its pass with one batch counted because its hook comes after the batch.
    callback = PaceCallback(logger=quiet)
    trainer_stub = SimpleNamespace(is_global_zero=True, sanity_checking=False, num_val_batches=[4])
    callback.on_validation_start(trainer_stub, None)
    for index in range(4):
        clock.advance(2.0)
        callback.on_validation_batch_end(trainer_stub, None, None, None, index)
    callback.on_validation_end(trainer_stub, None)
    found = of(records(stream), "validation")
    assert [(r.event, r.completed, r.elapsed) for r in found] == [
        (Event.START, 0, 0),
        (Event.UPDATE, 1, 2),
        (Event.UPDATE, 2, 4),
        (Event.UPDATE, 3, 6),
        (Event.UPDATE, 4, 8),
        (Event.END, 4, 8),
    ]
    assert stream.lines()[-1].count("0.5 batch/s") == 1
