"""
Project-wide logging configuration.

Call `configure_logging()` exactly once at the start of any executable
entrypoint (e.g. build_index.py, app.py, run_eval.py). Library modules
then just do:

    import logging
    log = logging.getLogger(__name__)

and inherit the formatting and level set here. This mirrors good
agent logging style (per-component prefix + colour) but routes through
Python's standard `logging` module instead of raw prints.
"""

from __future__ import annotations

import logging
import sys

from src.config import settings


# ---------------------------------------------------------------------------
# ANSI colour codes — used to tint each module's log lines so the trace is
# readable in the terminal during development. Kept as a class for symmetry
# with Ed's `Agent` colours in Week 8.
# ---------------------------------------------------------------------------

class _Colour:
    RESET = "\033[0m"
    GREY = "\033[90m"
    BLUE = "\033[34m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"


# Map log level → colour. INFO is plain blue; warnings and errors stand out.
_LEVEL_COLOURS: dict[int, str] = {
    logging.DEBUG: _Colour.GREY,
    logging.INFO: _Colour.BLUE,
    logging.WARNING: _Colour.YELLOW,
    logging.ERROR: _Colour.RED,
    logging.CRITICAL: _Colour.RED,
}


# ---------------------------------------------------------------------------
# Custom formatter — adds the colour and trims the logger name to its last
# component so log lines look like:
#
#     [INFO] chunker: chunked housing_act_1988.md into 47 chunks
# ---------------------------------------------------------------------------

class _ColouredFormatter(logging.Formatter):
    """Formatter that colours the level and shortens the logger name."""

    def format(self, record: logging.LogRecord) -> str:
        colour = _LEVEL_COLOURS.get(record.levelno, _Colour.RESET)
        record.short_name = record.name.split(".")[-1]
        record.coloured_level = f"{colour}{record.levelname}{_Colour.RESET}"
        return super().format(record)


# ---------------------------------------------------------------------------
# Public entry point — called once at the top of any script.
# ---------------------------------------------------------------------------

def configure_logging() -> None:
    """Configure the root logger. Idempotent — safe to call multiple times."""
    root = logging.getLogger()

    # Avoid stacking handlers if called twice (e.g. Jupyter reloads).
    if root.handlers:
        return

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(
        _ColouredFormatter(fmt="[%(coloured_level)s] %(short_name)s: %(message)s")
    )
    root.addHandler(handler)
    root.setLevel(settings.log_level.upper())