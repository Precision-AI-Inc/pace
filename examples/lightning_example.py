# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Report a PyTorch Lightning run through Pace.

Runs a tiny model on random data on the CPU in a few seconds. Needs
``pip install precisionai-pace[lightning]``. Every progress line ends with the
``[pace:1 ...]`` trailer a scheduler reads.
"""

from __future__ import annotations

from typing import Any

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

from precisionai.pace import logger
from precisionai.pace.lightning import PaceCallback


class TinyRegressor(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(8, 1)

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        x, y = batch
        return torch.nn.functional.mse_loss(self.layer(x), y)

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        x, y = batch
        self.log("val_loss", torch.nn.functional.mse_loss(self.layer(x), y))

    def configure_optimizers(self) -> Any:
        return torch.optim.SGD(self.parameters(), lr=0.01)


def loader(samples: int) -> DataLoader:
    return DataLoader(TensorDataset(torch.randn(samples, 8), torch.randn(samples, 1)), batch_size=16)


if __name__ == "__main__":
    # A run of a few seconds would otherwise write only its start and end lines, because
    # Pace spaces progress lines at least PACE_MIN_INTERVAL (15 s) apart. Real jobs keep the default.
    logger.configure(min_interval=0)
    trainer = pl.Trainer(
        accelerator="cpu",
        max_epochs=3,
        callbacks=[PaceCallback()],
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(TinyRegressor(), loader(160), loader(64))
