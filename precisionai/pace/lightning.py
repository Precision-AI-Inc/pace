# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""PyTorch Lightning callback that reports training progress through Pace.

``PaceCallback`` writes one ``root`` scope for the whole fit (``trainer.global_step`` of
``trainer.estimated_stepping_batches``) and one child scope per validation, test or
predict pass. Only global rank 0 writes, whatever ``PACE_ALL_RANKS`` says, so a
multi-GPU job still produces one record per scope.

Install with ``pip install precisionai-pace[lightning]``. Both ``lightning`` (2.x) and
the older ``pytorch_lightning`` import names are supported. When both are installed (2.x)
the callback is a subclass of both packages' ``Callback``, so either package's ``Trainer``
accepts it.
"""

import math
from typing import Any

from precisionai.pace._adapters import AdapterScopes
from precisionai.pace.core import Logger

_CALLBACK_BASES: list[type] = []
try:
    from lightning.pytorch.callbacks import Callback as _LightningCallback  # type: ignore[import]

    _CALLBACK_BASES.append(_LightningCallback)
except ImportError:
    pass
try:
    import pytorch_lightning as _pytorch_lightning  # type: ignore[import]
    from pytorch_lightning.callbacks import Callback as _PytorchLightningCallback  # type: ignore[import]

    # Hooks the callback does not override come from the first base. pytorch_lightning 2.x is
    # the same code as lightning 2.x, so both bases are safe; a 1.x Trainer calls some hooks
    # with other arguments, so 1.x is a base only when it is the one package installed.
    _major = str(getattr(_pytorch_lightning, "__version__", "2")).split(".")[0]
    if not _CALLBACK_BASES or not _major.isdigit() or int(_major) >= 2:
        _CALLBACK_BASES.append(_PytorchLightningCallback)
except ImportError:
    pass
_LIGHTNING_AVAILABLE = bool(_CALLBACK_BASES)
if not _CALLBACK_BASES:
    _CALLBACK_BASES.append(object)

__all__ = ["PaceCallback"]


def _batches(counts: Any) -> float | None:
    """Return the number of batches in a loop from Lightning's per-dataloader counts, or ``None`` if unknown."""
    try:
        values = list(counts) if isinstance(counts, (list, tuple)) else [counts]
        total = float(sum(values))
    except (TypeError, ValueError):
        return None
    return total if math.isfinite(total) and total > 0 else None


class PaceCallback(*_CALLBACK_BASES):  # type: ignore[misc]
    """Report a Lightning run's progress as Pace records.

    Parameters
    ----------
    description : str
        Label of the fit scope, ``"Train"`` by default.
    logger : Logger, optional
        Pace logger to write to; the module-level one by default. Tests pass their own.
    scope : str, optional
        Explicit scope id for the fit scope.
    root : bool
        Whether the first scope this callback starts stands for the whole job. Pass
        ``False`` when the program already opened a root scope of its own.

    Raises
    ------
    ImportError
        If neither ``lightning`` nor ``pytorch_lightning`` is installed.

    Notes
    -----
    Progress is read in optimizer steps (``trainer.global_step``), the unit
    ``trainer.estimated_stepping_batches`` uses. The total is unknown when Lightning
    cannot estimate it (``max_epochs=-1`` with no ``max_steps``, or an unsized
    dataloader). A fit restored from a checkpoint writes the ``res`` flag and starts at the
    restored step. Evaluation loops run during the fit are child scopes of the fit scope;
    the sanity check is not reported. Each pass is its own scope (``validation``,
    ``validation-2``, ...), because a scope cannot start twice.
    """

    def __init__(
        self,
        description: str = "Train",
        *,
        logger: Logger | None = None,
        scope: str | None = None,
        root: bool = True,
    ) -> None:
        if not _LIGHTNING_AVAILABLE:
            raise ImportError("lightning is required: pip install precisionai-pace[lightning]") from None
        super().__init__()
        self._scopes = AdapterScopes(description, "step", scope=scope, root=root, logger=logger)

    def on_train_start(self, trainer: Any, pl_module: Any) -> None:
        """Start the fit scope at the step the run begins from."""
        if not trainer.is_global_zero:
            return
        step = trainer.global_step
        total = trainer.estimated_stepping_batches
        self._scopes.begin(total if math.isfinite(total) else None, completed=step, resumed=step > 0)

    def on_train_batch_end(self, trainer: Any, pl_module: Any, outputs: Any, batch: Any, batch_idx: int) -> None:
        """Report the optimizer step reached."""
        self._scopes.update(trainer.global_step)

    def on_train_end(self, trainer: Any, pl_module: Any) -> None:
        """End the fit scope."""
        self._scopes.end()

    def on_fit_end(self, trainer: Any, pl_module: Any) -> None:
        """End the fit scope if training never started or its end was not reported."""
        self._scopes.end()

    def on_exception(self, trainer: Any, pl_module: Any, exception: BaseException) -> None:
        """Mark the open scopes failed."""
        self._scopes.fail()

    def on_validation_start(self, trainer: Any, pl_module: Any) -> None:
        """Open a child scope for a validation pass, except the sanity check."""
        if not trainer.sanity_checking:
            self._open(trainer, "Validation", "num_val_batches")

    def on_validation_batch_end(
        self, trainer: Any, pl_module: Any, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Count a validation batch."""
        self._scopes.advance_child()

    def on_validation_end(self, trainer: Any, pl_module: Any) -> None:
        """End the validation scope."""
        self._scopes.end_child()

    def on_test_start(self, trainer: Any, pl_module: Any) -> None:
        """Open a scope for a test pass."""
        self._open(trainer, "Test", "num_test_batches")

    def on_test_batch_end(
        self, trainer: Any, pl_module: Any, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Count a test batch."""
        self._scopes.advance_child()

    def on_test_end(self, trainer: Any, pl_module: Any) -> None:
        """End the test scope."""
        self._scopes.end_child()

    def on_predict_start(self, trainer: Any, pl_module: Any) -> None:
        """Open a scope for a predict pass."""
        self._open(trainer, "Predict", "num_predict_batches")

    def on_predict_batch_end(
        self, trainer: Any, pl_module: Any, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Count a predict batch."""
        self._scopes.advance_child()

    def on_predict_end(self, trainer: Any, pl_module: Any) -> None:
        """End the predict scope."""
        self._scopes.end_child()

    def _open(self, trainer: Any, description: str, attribute: str) -> None:
        if trainer.is_global_zero:
            self._scopes.begin_child(description, _batches(getattr(trainer, attribute, None)), "batch")
