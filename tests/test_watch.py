"""Tests for the --watch loop (summarizer/watch.py), with a fake handler and no waiting."""

import logging
from pathlib import Path

import pytest

from summarizer import watch
from summarizer.errors import SummarizerError


@pytest.fixture(autouse=True)
def no_sleeping(monkeypatch):
    monkeypatch.setattr(watch.time, "sleep", lambda seconds: None)


def recording_handler(done: list[Path]):
    """A handler that records the files it got and writes the summary file, like the real one."""

    def handle(path: Path) -> None:
        done.append(path)
        watch.summary_path(path).write_text("{}", encoding="utf-8")

    return handle


def test_summary_file_sits_next_to_the_text_file():
    assert watch.summary_path(Path("inbox/notes.txt")) == Path("inbox/notes.summary.json")


def test_a_new_file_is_handled_once_its_size_is_stable(tmp_path):
    (tmp_path / "notes.txt").write_text("Ana will send the report.", encoding="utf-8")
    done: list[Path] = []

    # Poll 1 only records the size; poll 2 sees it unchanged and handles the file.
    assert watch.watch_folder(tmp_path, recording_handler(done), max_polls=1) == 0
    assert watch.watch_folder(tmp_path, recording_handler(done), max_polls=2) == 1
    assert done == [tmp_path / "notes.txt"]


def test_a_file_with_a_summary_is_never_redone(tmp_path):
    (tmp_path / "notes.txt").write_text("Ana will send the report.", encoding="utf-8")
    (tmp_path / "notes.summary.json").write_text(
        "{}", encoding="utf-8"
    )  # already done, e.g. before a restart
    done: list[Path] = []

    assert watch.watch_folder(tmp_path, recording_handler(done), max_polls=3) == 0
    assert done == []


def test_a_file_still_being_written_is_left_alone(tmp_path, monkeypatch):
    path = tmp_path / "big.txt"
    path.write_text("part one", encoding="utf-8")
    done: list[Path] = []

    # While the watcher "sleeps" after the first poll, more text arrives.
    def sleep_and_grow(seconds):
        if path.stat().st_size < 20:
            path.write_text("part one, part two, part three", encoding="utf-8")

    monkeypatch.setattr(watch.time, "sleep", sleep_and_grow)

    # poll 1: size 8 recorded; poll 2: size changed -> recorded again; poll 3: stable -> handled.
    assert watch.watch_folder(tmp_path, recording_handler(done), max_polls=2) == 0
    assert watch.watch_folder(tmp_path, recording_handler(done), max_polls=3) == 1


def test_other_files_are_ignored(tmp_path):
    (tmp_path / "photo.png").write_bytes(b"\x89PNG")
    (tmp_path / "notes.summary.json").write_text("{}", encoding="utf-8")
    done: list[Path] = []
    assert watch.watch_folder(tmp_path, recording_handler(done), max_polls=2) == 0


def test_a_failing_file_is_logged_once_and_skipped(tmp_path, caplog):
    (tmp_path / "bad.txt").write_text("x", encoding="utf-8")
    (tmp_path / "good.txt").write_text("Ana will send the report.", encoding="utf-8")
    done: list[Path] = []
    attempts: list[Path] = []

    def handle(path: Path) -> None:
        attempts.append(path)
        if path.name == "bad.txt":
            raise SummarizerError("looks like a binary file")
        recording_handler(done)(path)

    with caplog.at_level(logging.WARNING, logger="summarizer.watch"):
        handled = watch.watch_folder(tmp_path, handle, max_polls=4)

    assert handled == 1
    assert done == [tmp_path / "good.txt"]
    assert attempts.count(tmp_path / "bad.txt") == 1  # not retried every poll
    assert "bad.txt: looks like a binary file" in caplog.text
