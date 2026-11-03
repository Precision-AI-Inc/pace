"""Tests for the Transformers callback, driven by real tiny CPU runs of a model built from a config."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from precisionai.pace import Event, Record, parse_line
from precisionai.pace import transformers as pace_transformers
from precisionai.pace.core import Logger
from precisionai.pace.transformers import PaceCallback
from tests.conftest import FakeStream

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("accelerate")

SAMPLES = 8
BATCH_SIZE = 2
STEPS_PER_EPOCH = SAMPLES // BATCH_SIZE
EPOCHS = 2


class Rows(torch.utils.data.Dataset):
    def __init__(self, rows: int = SAMPLES) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return self.rows

    def __getitem__(self, index: int) -> dict[str, Any]:
        generator = torch.Generator().manual_seed(index)
        return {"input_ids": torch.randint(1, 50, (8,), generator=generator), "labels": index % 2}


def model() -> Any:
    config = transformers.BertConfig(
        vocab_size=50,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=16,
        num_labels=2,
    )
    return transformers.BertForSequenceClassification(config)


def trainer(log: Logger, tmp_path: Path, **options: Any) -> Any:
    callbacks = options.pop("callbacks", None)
    options.setdefault("num_train_epochs", EPOCHS)
    arguments = transformers.TrainingArguments(
        output_dir=str(tmp_path),
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        use_cpu=True,
        report_to=[],
        disable_tqdm=True,
        save_strategy=options.pop("save_strategy", "no"),
        logging_strategy="no",
        **options,
    )
    return transformers.Trainer(
        model=model(),
        args=arguments,
        train_dataset=Rows(),
        eval_dataset=Rows(6),
        callbacks=[PaceCallback(logger=log)] if callbacks is None else callbacks,
    )


def records(stream: FakeStream) -> list[Record]:
    return [record for record in map(parse_line, stream.lines()) if record is not None]


def of(found: list[Record], scope: str) -> list[Record]:
    return [record for record in found if record.scope == scope]


@pytest.fixture
def quiet(log: Logger) -> Logger:
    log.configure(min_interval=0)
    return log


def test_training_writes_one_root_scope_with_the_step_budget(quiet: Logger, stream: FakeStream, tmp_path: Path) -> None:
    trainer(quiet, tmp_path).train()
    found = records(stream)
    train = of(found, "train")
    assert train[0].event is Event.START
    assert train[0].root
    assert train[0].unit == "step"
    assert train[0].completed == 0
    assert train[0].resumed is False
    assert train[-1].event is Event.END
    assert train[-1].completed == STEPS_PER_EPOCH * EPOCHS
    counts = [r.completed or 0 for r in train]
    assert counts == sorted(counts)
    assert {r.total for r in train} == {STEPS_PER_EPOCH * EPOCHS}
    assert {r.scope for r in found if r.root} == {"train"}


def test_evaluation_is_a_child_of_the_training_scope(quiet: Logger, stream: FakeStream, tmp_path: Path) -> None:
    trainer(quiet, tmp_path, eval_strategy="epoch").train()
    found = records(stream)
    first = of(found, "evaluation")
    assert first[0].event is Event.START
    assert first[0].parent == "train"
    assert first[0].unit == "batch"
    assert first[0].total == 3
    assert first[-1].event is Event.END
    assert first[-1].completed == 3
    assert not first[0].root
    assert {r.scope for r in found if r.parent == "train"} == {"evaluation", "evaluation-2"}


def test_an_evaluation_after_training_is_not_a_second_root(quiet: Logger, stream: FakeStream, tmp_path: Path) -> None:
    runner = trainer(quiet, tmp_path)
    runner.train()
    runner.evaluate()
    found = records(stream)
    evaluation = of(found, "evaluation")
    assert evaluation[-1].event is Event.END
    assert evaluation[-1].completed == 3
    assert {r.scope for r in found if r.root} == {"train"}


def test_a_standalone_evaluation_is_the_root(quiet: Logger, stream: FakeStream, tmp_path: Path) -> None:
    trainer(quiet, tmp_path).evaluate()
    evaluation = of(records(stream), "evaluation")
    assert evaluation[0].root
    assert evaluation[0].parent is None
    assert evaluation[-1].event is Event.END


def test_predict_ends_its_scope(quiet: Logger, stream: FakeStream, tmp_path: Path) -> None:
    trainer(quiet, tmp_path).predict(Rows(4))
    evaluation = of(records(stream), "evaluation")
    assert evaluation[-1].event is Event.END
    assert evaluation[-1].completed == 2
    assert evaluation[-1].total == 2


def test_an_evaluation_with_no_batches_is_still_written(quiet: Logger, stream: FakeStream) -> None:
    callback = PaceCallback(logger=quiet)
    state = transformers.TrainerState()
    state.is_world_process_zero = True
    callback.on_evaluate(None, state, None, metrics={})
    evaluation = of(records(stream), "evaluation")
    assert [r.event for r in evaluation] == [Event.START, Event.END]
    assert evaluation[0].total is None


def test_a_step_budget_that_is_unknown_is_written_as_unknown(quiet: Logger, stream: FakeStream) -> None:
    state = transformers.TrainerState(max_steps=-1)
    state.is_world_process_zero = True
    PaceCallback(logger=quiet).on_train_begin(None, state, None)
    assert of(records(stream), "train")[0].total is None


def test_resume_from_a_checkpoint_starts_at_the_restored_step(
    quiet: Logger, stream: FakeStream, tmp_path: Path
) -> None:
    first = trainer(quiet, tmp_path, num_train_epochs=1, save_strategy="epoch", callbacks=[])
    first.train()
    checkpoint = str(tmp_path / f"checkpoint-{STEPS_PER_EPOCH}")

    trainer(quiet, tmp_path).train(resume_from_checkpoint=checkpoint)
    train = of(records(stream), "train")
    assert train[0].resumed is True
    assert train[0].completed == STEPS_PER_EPOCH
    assert train[0].total == STEPS_PER_EPOCH * EPOCHS
    assert train[-1].event is Event.END
    assert train[-1].completed == STEPS_PER_EPOCH * EPOCHS


def test_a_non_zero_process_writes_nothing(
    quiet: Logger, stream: FakeStream, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PACE_ALL_RANKS", "1")
    monkeypatch.setattr(transformers.Trainer, "is_world_process_zero", lambda _: False)
    runner = trainer(quiet, tmp_path, eval_strategy="epoch")
    runner.train()
    runner.predict(Rows(4))
    assert stream.getvalue() == ""


def test_a_crash_leaves_the_scope_open_and_fail_writes_it(quiet: Logger, stream: FakeStream, tmp_path: Path) -> None:
    callback = PaceCallback(logger=quiet)

    class Crash(transformers.TrainerCallback):
        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            if state.global_step == 2:
                raise RuntimeError("boom")

    runner = trainer(quiet, tmp_path, callbacks=[callback, Crash()])
    with pytest.raises(RuntimeError, match="boom"):
        runner.train()
    assert of(records(stream), "train")[-1].event is Event.UPDATE
    callback.fail()
    train = of(records(stream), "train")
    assert train[-1].event is Event.FAIL
    assert train[-1].completed == 2


def test_a_description_and_scope_id_can_be_chosen(quiet: Logger, stream: FakeStream, tmp_path: Path) -> None:
    callback = PaceCallback("Finetune", logger=quiet, scope="ft", root=False)
    trainer(quiet, tmp_path, callbacks=[callback]).train()
    train = of(records(stream), "ft")
    assert train[0].label == "Finetune"
    assert not train[0].root


def test_without_transformers_installed_the_callback_says_how_to_install_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pace_transformers, "_TRANSFORMERS_AVAILABLE", False)
    with pytest.raises(ImportError, match=r"precisionai-pace\[transformers\]"):
        PaceCallback()
