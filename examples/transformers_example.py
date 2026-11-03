# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Report a Hugging Face ``Trainer`` run through Pace.

Trains a tiny BERT built from a config, so nothing is downloaded, on random token ids on
the CPU in a few seconds. Needs ``pip install precisionai-pace[transformers] accelerate``.
A TRL trainer takes the same callback. Every progress line ends with the ``[pace:1 ...]``
trailer a scheduler reads.
"""

from __future__ import annotations

import tempfile
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import BertConfig, BertForSequenceClassification, Trainer, TrainingArguments

from precisionai.pace import logger
from precisionai.pace.transformers import PaceCallback


class RandomTokens(Dataset):
    def __init__(self, rows: int) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return self.rows

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {"input_ids": torch.randint(1, 100, (16,)), "labels": index % 2}


if __name__ == "__main__":
    # A run of a few seconds would otherwise write only its start and end lines, because
    # Pace spaces progress lines at least PACE_MIN_INTERVAL (15 s) apart. Real jobs keep the default.
    logger.configure(min_interval=0)
    config = BertConfig(
        vocab_size=100,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=32,
    )
    config.num_labels = 2
    arguments = TrainingArguments(
        output_dir=tempfile.mkdtemp(),
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="no",
        report_to=[],
        disable_tqdm=True,
        use_cpu=True,
    )
    trainer = Trainer(
        model=BertForSequenceClassification(config),
        args=arguments,
        train_dataset=RandomTokens(64),
        eval_dataset=RandomTokens(32),
        callbacks=[PaceCallback()],
    )
    trainer.train()
