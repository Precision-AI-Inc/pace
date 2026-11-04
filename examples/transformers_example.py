# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Report a Hugging Face ``Trainer`` run through Pace with ``PaceCallback``.

Situation: a ``Trainer`` fits a tiny BERT, built from a config so nothing is downloaded, on
random token ids on the CPU, for 3 epochs with an evaluation pass after each. The callback
reports the whole run, so the training code needs no changes: one ``root`` scope counting
``state.global_step`` against ``state.max_steps``, and a child scope for each evaluation.
A TRL trainer takes the same callback.

Shows: ``PaceCallback()`` from ``precisionai.pace.transformers``, and ``logger.configure``
to make a run of a few seconds write more than its first and last progress line.

Needs: ``pip install precisionai-pace[transformers]``. The other examples need
no extras.

Run: ``python examples/transformers_example.py``

Expect: a ``train`` scope marked ``root`` counting steps, ``evaluation`` scopes whose
trailers carry ``<train``, and the ``end`` of the root. Every progress line ends with the
``[pace:2 ...]`` trailer a scheduler reads.
"""

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
