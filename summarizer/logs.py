"""Logging set-up shared by cli.py and benchmark.py.

The rule for every log message in this project: METADATA ONLY. File sizes,
token counts, timings, chunk counts, the reason a reply was rejected. Never
the document, never the prompt, never the model's reply. A log must be safe
to paste into a bug report.

Every module logs under the "summarizer" name (logging.getLogger(__name__)
gives "summarizer.core", "summarizer.files", ...), so one handler on that
name catches them all, and other libraries' logs (httpx) stay out.
"""

import logging
import sys

LOGGER_NAME = "summarizer"


def setup_logging(verbose: bool = False) -> None:
    """INFO: the short progress notes users always see. --verbose adds DEBUG details, with time and module."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    fmt = "%(asctime)s %(levelname)-7s %(name)s: %(message)s" if verbose else "%(message)s"
    handler = logging.StreamHandler(sys.stderr)  # stderr, so logs never mix with the real output
    handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))

    logger.handlers.clear()  # main() may run more than once (tests); don't print every line twice
    logger.addHandler(handler)
