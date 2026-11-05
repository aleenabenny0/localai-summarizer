"""--watch: keep summarizing .txt files as they appear in a folder.

Plain polling, no extra library: every few seconds, list the .txt files in the
folder and handle any that has no <name>.summary.json yet. The summary file is
the "done" marker, so restarting the watcher never redoes work, and files.py's
no-overwrite rule means it never clobbers a result either.

A file that is still being copied in is left alone until its size has stopped
changing for one interval, so we never summarize half a file.
"""

import logging
import time
from collections.abc import Callable
from pathlib import Path

from summarizer.errors import SummarizerError

log = logging.getLogger(__name__)


def summary_path(text_file: Path) -> Path:
    """notes.txt -> notes.summary.json, in the same folder."""
    return text_file.with_name(text_file.stem + ".summary.json")


def watch_folder(
    folder: Path,
    handle: Callable[[Path], None],
    interval_s: float = 5.0,
    max_polls: int | None = None,
) -> int:
    """Poll `folder`; call `handle(path)` once for every .txt file that has no summary yet.

    A file whose handling fails (too big, binary, the model misbehaved) is
    logged once and skipped until the watcher is restarted. Runs until Ctrl+C,
    or for `max_polls` polls (used by the tests). Returns how many files were handled.
    """
    last_size: dict[Path, int] = {}  # size of each not-yet-handled file at the previous poll
    failed: set[Path] = set()
    handled = 0
    polls = 0
    while max_polls is None or polls < max_polls:
        polls += 1
        for path in sorted(folder.glob("*.txt")):
            if path in failed or summary_path(path).exists():
                continue
            try:
                size = path.stat().st_size
            except OSError:  # deleted or locked between the listing and now; look again next time
                continue
            if last_size.get(path) != size:  # first sight, or still growing: wait one more interval
                last_size[path] = size
                continue
            try:
                handle(path)
                handled += 1
            except SummarizerError as e:
                log.warning("%s: %s", path.name, e)
                failed.add(path)
            last_size.pop(path, None)
        time.sleep(interval_s)
    return handled
