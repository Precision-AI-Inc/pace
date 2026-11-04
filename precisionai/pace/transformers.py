# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Hugging Face Transformers callback that reports training progress through Pace.

``PaceCallback`` writes one ``root`` scope for the whole training run
(``state.global_step`` of ``state.max_steps``) and one child scope per evaluation or
prediction pass. Only the world's main process (``state.is_world_process_zero``) writes.
TRL's trainers subclass ``transformers.Trainer``, so the same callback serves them.

Install with ``pip install precisionai-pace[transformers]``.
"""

from typing import Any

from precisionai.pace._adapters import AdapterScopes
from precisionai.pace.core import Logger

try:
    from transformers import TrainerCallback  # type: ignore[import]

    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    TrainerCallback = object  # type: ignore[assignment,misc]
    _TRANSFORMERS_AVAILABLE = False


def _length(dataloader: Any) -> float | None:
    """Return the number of batches of a dataloader, or ``None`` when it has no length."""
    try:
        return float(len(dataloader))
    except TypeError:
        return None


class PaceCallback(TrainerCallback):  # type: ignore[misc,valid-type]
    """Report a ``transformers.Trainer`` run's progress as Pace records.

    Parameters
    ----------
    description : str
        Label of the training scope, ``"Train"`` by default.
    logger : Logger, optional
        Pace logger to write to; the module-level one by default. Tests pass their own.
    scope : str, optional
        Explicit scope id for the training scope.
    root : bool
        Whether the first scope this callback starts stands for the whole job. Pass
        ``False`` when the program already opened a root scope of its own.

    Raises
    ------
    ImportError
        If ``transformers`` is not installed.

    Notes
    -----
    Progress is read in optimizer steps (``state.global_step``) against
    ``state.max_steps``, which is unknown when the trainer has no step budget. Training
    resumed from a checkpoint writes ``res=1`` and starts at the restored step. An
    evaluation pass is a child scope of the training scope that starts with its first
    batch and ends in ``on_evaluate`` (or ``on_predict``); its total is the length of the
    evaluation dataloader when it has one. The ``Trainer`` reports a batch only after it
    has run and has no hook before the first, so the scope starts with one batch already
    counted: its rate and its "done in" time cover the batches after the first, and leave
    out the first one's duration. The ``Trainer`` has no hook for an exception in
    training, so a crashed run writes no ``fail`` record: the scheduler sees the stream
    go quiet and the process exit. Call :meth:`fail` from your own error handling to
    write it.
    """

    def __init__(
        self,
        description: str = "Train",
        *,
        logger: Logger | None = None,
        scope: str | None = None,
        root: bool = True,
    ) -> None:
        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers is required: pip install precisionai-pace[transformers]") from None
        super().__init__()
        self._scopes = AdapterScopes(description, "step", scope=scope, root=root, logger=logger)

    def fail(self) -> None:
        """Mark the open scopes failed, for use when training raised."""
        self._scopes.fail()

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        """Start the training scope at the step the run begins from."""
        if state.is_world_process_zero:
            step = state.global_step
            self._scopes.begin(state.max_steps, completed=step, resumed=step > 0)

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        """Report the optimizer step reached."""
        self._scopes.update(state.global_step)

    def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        """End the training scope."""
        self._scopes.end()

    def on_prediction_step(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        """Count an evaluation or prediction batch, opening the pass's scope on the first.

        The ``Trainer`` calls this after each batch, and has no hook before the first, so
        the scope opens with that batch already done and its clock starts after it.
        """
        if not state.is_world_process_zero:
            return
        if self._scopes.child is None:
            self._open(kwargs.get("eval_dataloader"), completed=1)
        else:
            self._scopes.advance_child()

    def on_evaluate(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        """End the evaluation scope, writing it first when the pass had no batches."""
        self._close(state)

    def on_predict(self, args: Any, state: Any, control: Any, metrics: Any = None, **kwargs: Any) -> None:
        """End the prediction scope, writing it first when the pass had no batches."""
        self._close(state)

    def _open(self, dataloader: Any, completed: float = 0.0) -> None:
        self._scopes.begin_child("Evaluation", _length(dataloader), "batch", completed=completed)

    def _close(self, state: Any) -> None:
        if state.is_world_process_zero:
            if self._scopes.child is None:
                self._open(None)
            self._scopes.end_child()
