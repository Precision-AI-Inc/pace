# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Log at every level, record an exception, and change the settings.

Situation: a script that only needs good log lines, with no loop to report on. Pace's
``logger`` is ready to use after the import, so there is nothing to set up first.

Shows: ``logger.debug/info/success/warning/error/critical``, ``logger.exception``,
``logger.log``, ``logger.configure`` and ``logger.reset``.

Run: ``python examples/basic_logging.py``

Expect: one line per call, each a UTC timestamp, the level padded to eight characters and
the message. The first ``debug`` line is dropped (the default level is INFO) and the
second one appears after ``configure(level="debug")``. The ``info`` line after
``configure(level="warning")`` is dropped. The failed config lookup is written as an
``ERROR`` line followed by its traceback. There are no progress lines: this script does not
use ``track`` or ``progress``.
"""

from precisionai.pace import logger


def read_learning_rate(config: dict) -> float:
    """Return the learning rate, or log why it is missing and fall back to a default."""
    try:
        return config["lr"]
    except KeyError:
        logger.exception("Config is missing a key")
        return 0.001


def main() -> None:
    """Show each log level, an exception and a settings change."""
    logger.debug("Hidden: the default level is INFO")
    logger.info("Loading model")
    logger.success("Model ready")
    logger.warning("Falling back to cached metadata")
    logger.error("Could not upload the sample images; the job continues")
    logger.critical("Out of disk space; the job cannot continue")

    # Each method takes one argument, so format the message yourself.
    batch_size = 32
    logger.info(f"Batch size is {batch_size}")

    # Pick the level at run time with log(); a name in any case or a number works.
    logger.log("warning", "Chosen at run time")

    # Call exception() inside an except block: an ERROR line plus the traceback.
    read_learning_rate({"epochs": 3})

    # configure() changes only what you pass. It beats a PACE_* environment variable.
    logger.configure(level="debug", color="never")
    logger.debug("Visible now")
    logger.configure(level="warning")
    logger.info("Hidden again")
    logger.warning("Only warnings and above from here")

    # reset() forgets every explicit setting, so the environment and the defaults apply.
    logger.reset()
    logger.info("Back to the defaults")


if __name__ == "__main__":
    main()
