# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""The scope bookkeeping the framework adapters share.

An adapter reports one main scope (the training run) and, while it is open, at most one
child scope (a validation or evaluation pass) whose parent is the main scope. This module
holds that bookkeeping with the standard library only, so it is tested without any
framework installed.
"""

from precisionai.pace.core import Logger
from precisionai.pace.scopes import Scope


class AdapterScopes:
    """Own the main scope of a run and the child scope of the pass currently running inside it.

    Every method is safe to call in any order and any number of times: a scope that was
    never started, or has already ended, is left alone. Only the first scope this object
    starts is marked as the job's root, because a process has at most one root.

    Parameters
    ----------
    description : str
        Label of the main scope.
    unit : str
        Unit of the main scope's counts.
    scope : str, optional
        Explicit scope id for the main scope.
    root : bool
        Whether the first scope this object starts stands for the whole job.
    logger : Logger, optional
        Output to use; the module-level logger by default.
    """

    def __init__(
        self,
        description: str,
        unit: str = "step",
        *,
        scope: str | None = None,
        root: bool = True,
        logger: Logger | None = None,
    ) -> None:
        self.description = description
        self.unit = unit
        self._scope_id = scope
        self._root_allowed = root
        self._logger = logger
        self._main: Scope | None = None
        self._child: Scope | None = None

    @property
    def main(self) -> Scope | None:
        """The main scope while it is open, otherwise ``None``."""
        return self._main

    @property
    def child(self) -> Scope | None:
        """The child scope while it is open, otherwise ``None``."""
        return self._child

    def begin(self, total: float | None, completed: float = 0.0, resumed: bool = False) -> Scope:
        """Start the main scope, ending any main scope that is still open.

        Parameters
        ----------
        total : float, optional
            Total amount; ``None`` when unknown.
        completed : float
            Amount already done, above zero when resuming.
        resumed : bool
            Whether this resumes earlier work.

        Returns
        -------
        Scope
            The started scope.
        """
        self.end()
        scope = Scope(
            self.description,
            total,
            self.unit,
            scope=self._scope_id,
            root=self._take_root(),
            completed=completed,
            resumed=resumed,
            logger=self._logger,
        )
        scope.start()
        self._main = scope
        return scope

    def update(self, completed: float) -> None:
        """Set the main scope's completed amount."""
        if self._main is not None:
            self._main.update(completed)

    def begin_child(self, description: str, total: float | None, unit: str, completed: float = 0.0) -> Scope:
        """Start a child scope, ending the previous one if it is still open.

        Its parent is the main scope while that is open. With no main scope it is a
        top-level scope but never the root: an evaluation run before training must
        not mark the job complete when it ends, so only training can be the root.

        Parameters
        ----------
        description : str
            Label of the child scope.
        total : float, optional
            Total amount; ``None`` when unknown.
        unit : str
            Unit of the child scope's counts.
        completed : float
            Amount already done when the scope starts, for a pass whose first unit has
            finished before the adapter hears of it.

        Returns
        -------
        Scope
            The started scope.
        """
        self.end_child()
        scope = Scope(description, total, unit, completed=completed, logger=self._logger)
        if self._main is None:
            scope.start()
        else:
            self._main.bind()
            try:
                scope.start()
            finally:
                self._main.unbind()
        self._child = scope
        return scope

    def advance_child(self, n: float = 1) -> None:
        """Add ``n`` to the child scope's completed amount."""
        if self._child is not None:
            self._child.advance(n)

    def end_child(self) -> None:
        """End the child scope if one is open."""
        child, self._child = self._child, None
        if child is not None:
            child.end()

    def end(self) -> None:
        """End the child scope and then the main scope, whichever are open."""
        self.end_child()
        main, self._main = self._main, None
        if main is not None:
            main.end()

    def fail(self) -> None:
        """Mark the child scope and then the main scope failed, whichever are open."""
        child, self._child = self._child, None
        if child is not None:
            child.fail()
        main, self._main = self._main, None
        if main is not None:
            main.fail()

    def _take_root(self) -> bool:
        taken, self._root_allowed = self._root_allowed, False
        return taken
