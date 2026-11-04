# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Report a library's own progress bar and log lines through Pace.

Situation: the long part of your job runs inside one call to a library, so nothing you write
around the call can report progress, and the library prints its own log lines. ``Engine``
below stands in for such a library (vLLM is one): it draws a ``tqdm`` bar it is handed
(``use_tqdm=``), and it logs through a logger with its own console handler and
``propagate = False``, so each line would be printed twice, or never reach Pace.

Shows: ``PaceTqdm`` passed as the library's progress-bar class, so its bar becomes the root
scope, and ``capture_logging(take_over=["engine"])``, which removes the library's console
handler and lets its records through to Pace's handler. Call it after the library is
imported. See ``docs/progress.md`` and ``docs/logging.md``.

Run: ``python examples/library_hooks.py``. Add ``PACE_MIN_INTERVAL=0`` to see every progress
line.

Expect: the library's two log lines once each, in Pace's format on stdout, and a ``Rendering``
progress scope that counts the prompts as they finish. Nothing on stderr.
"""

import functools
import logging
import time
from collections.abc import Callable
from typing import Any

from precisionai.pace import capture_logging, logger
from precisionai.pace.tqdm import PaceTqdm

log = logging.getLogger("engine")
log.addHandler(logging.StreamHandler())
log.propagate = False


class Engine:
    """Stand in for a library that takes a ``tqdm``-like class and logs on its own."""

    def render(self, prompts: list[str], use_tqdm: Callable[..., Any]) -> list[str]:
        """Render every prompt, reporting progress to ``use_tqdm`` the way vLLM does."""
        log.info("Rendering %d prompts", len(prompts))
        bar = use_tqdm(total=len(prompts), desc="Rendering", dynamic_ncols=True)
        images = []
        for start in range(0, len(prompts), 2):
            time.sleep(0.01)
            finished = prompts[start : start + 2]
            images.extend(f"image of {prompt}" for prompt in finished)
            bar.update(len(finished))
        bar.close()
        log.warning("Cache is full, %d images were not kept", len(images) // 2)
        return images


def main() -> None:
    """Take over the library's logger, then hand it Pace's progress class."""
    capture_logging(level="INFO", take_over=["engine"])

    images = Engine().render(
        [f"prompt {number}" for number in range(8)],
        use_tqdm=functools.partial(PaceTqdm, root=True),
    )
    logger.success(f"Rendered {len(images)} images")


if __name__ == "__main__":
    main()
