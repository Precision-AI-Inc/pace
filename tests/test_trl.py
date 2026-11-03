# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""The Transformers callback on a TRL trainer, which subclasses ``transformers.Trainer``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from precisionai.pace import Event, parse_line
from precisionai.pace.core import Logger
from precisionai.pace.transformers import PaceCallback
from tests.conftest import FakeStream

pytest.importorskip("torch")
pytest.importorskip("accelerate")
transformers = pytest.importorskip("transformers")
trl = pytest.importorskip("trl")
datasets = pytest.importorskip("datasets")
tokenizers = pytest.importorskip("tokenizers")

WORDS = ["<pad>", "<eos>", "<unk>", "hello", "world", "a", "b", "c"]
ROWS = 8
BATCH_SIZE = 2
EPOCHS = 2


def tokenizer() -> Any:
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel({w: i for i, w in enumerate(WORDS)}, unk_token="<unk>"))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>"
    )


def tiny_model() -> Any:
    config = transformers.GPT2Config(
        vocab_size=len(WORDS),
        n_embd=16,
        n_layer=1,
        n_head=2,
        n_positions=32,
        bos_token_id=1,
        eos_token_id=1,
        pad_token_id=0,
    )
    return transformers.GPT2LMHeadModel(config)


def test_an_sft_trainer_reports_through_the_same_callback(log: Logger, stream: FakeStream, tmp_path: Path) -> None:
    log.configure(min_interval=0)
    arguments = trl.SFTConfig(
        output_dir=str(tmp_path),
        per_device_train_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        use_cpu=True,
        report_to=[],
        disable_tqdm=True,
        save_strategy="no",
        logging_strategy="no",
        max_length=16,
        dataset_text_field="text",
    )
    runner = trl.SFTTrainer(
        model=tiny_model(),
        args=arguments,
        train_dataset=datasets.Dataset.from_dict({"text": ["hello world a b c"] * ROWS}),
        processing_class=tokenizer(),
        callbacks=[PaceCallback(logger=log)],
    )
    runner.train()
    found = [r for r in map(parse_line, stream.lines()) if r is not None]
    assert [r.scope for r in found if r.event is Event.START] == ["train"]
    assert found[0].root
    assert found[0].total == ROWS // BATCH_SIZE * EPOCHS
    assert found[-1].event is Event.END
    assert found[-1].completed == ROWS // BATCH_SIZE * EPOCHS
