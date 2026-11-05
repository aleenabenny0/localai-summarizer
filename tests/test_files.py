"""Tests for reading the input file safely (summarizer/files.py)."""

import os
import sys

import pytest

from summarizer.errors import SummarizerError
from summarizer.files import check_output_path, read_input, write_output


def write(tmp_path, name, data):
    path = tmp_path / name
    path.write_bytes(data)
    return path


def test_reads_utf8(tmp_path):
    path = write(tmp_path, "notes.txt", "Ana to send the report – Friday".encode())
    assert read_input(path) == "Ana to send the report – Friday"


def test_reads_utf8_with_marker(tmp_path):
    # Notepad's "UTF-8 with BOM" puts 3 marker bytes first; they must not end up in the text.
    path = write(tmp_path, "notes.txt", b"\xef\xbb\xbfAna to send the report")
    assert read_input(path) == "Ana to send the report"


def test_reads_utf16_from_powershell(tmp_path):
    # Windows PowerShell 5.1 writes UTF-16 when you do: "some text" > notes.txt
    path = write(tmp_path, "notes.txt", "Ana to send the report".encode("utf-16"))
    assert read_input(path) == "Ana to send the report"


def test_dotdot_in_the_path_is_resolved(tmp_path):
    (tmp_path / "sub").mkdir()
    write(tmp_path, "notes.txt", b"hello")
    assert read_input(tmp_path / "sub" / ".." / "notes.txt") == "hello"


# --- Refused files, each with a message that says what to do -----------------


def test_too_big_is_refused_with_the_sizes(tmp_path):
    path = write(tmp_path, "big.txt", b"x" * 2_000_001)
    with pytest.raises(SummarizerError) as error:
        read_input(path, max_bytes=2_000_000)
    message = str(error.value)
    assert "is 2.0 MB; the limit is 2.0 MB" in message
    assert "--max-mb 3" in message


def test_limit_can_be_raised(tmp_path):
    path = write(tmp_path, "notes.txt", b"word " * 100)  # 500 bytes
    with pytest.raises(SummarizerError):
        read_input(path, max_bytes=100)
    assert read_input(path, max_bytes=1000).startswith("word word")


def test_binary_file_is_refused(tmp_path):
    path = write(tmp_path, "report.pdf", b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n1 0 obj\x00\x00<< /Type /Catalog >>")
    with pytest.raises(SummarizerError, match="binary file"):
        read_input(path)


def test_other_encodings_are_refused_with_a_fix(tmp_path):
    # "café" saved by an old Windows program (code page 1252): 0xe9 is not valid UTF-8.
    path = write(tmp_path, "old.txt", b"caf\xe9 meeting")
    with pytest.raises(SummarizerError) as error:
        read_input(path)
    message = str(error.value)
    assert "isn't UTF-8" in message
    assert "0xe9 at position 3" in message
    assert "Set-Content -Encoding utf8" in message


def test_empty_file_is_refused(tmp_path):
    path = write(tmp_path, "empty.txt", b"  \r\n")
    with pytest.raises(SummarizerError, match="is empty"):
        read_input(path)


def test_missing_file_shows_where_it_looked(tmp_path):
    with pytest.raises(SummarizerError) as error:
        read_input(tmp_path / "sub" / ".." / "missing.txt")
    message = str(error.value)
    assert "doesn't exist" in message
    assert str(tmp_path / "missing.txt") in message  # the '..' is resolved away


def test_folder_is_refused(tmp_path):
    with pytest.raises(SummarizerError, match="is a folder"):
        read_input(tmp_path)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows device names")
def test_windows_device_names_are_refused():
    # NUL looks like a file name but is a device: reading it gives nothing, forever.
    with pytest.raises(SummarizerError, match="not an ordinary file"):
        read_input("NUL")


# --- Writing the output file -------------------------------------------------


def test_writes_a_new_file_and_leaves_no_temporary_file(tmp_path):
    target = tmp_path / "result.json"
    assert write_output(target, '{"summary": "ok"}') == target
    assert target.read_text(encoding="utf-8") == '{"summary": "ok"}'
    assert os.listdir(tmp_path) == ["result.json"]  # the .tmp file is gone


def test_never_overwrites_without_force(tmp_path):
    target = tmp_path / "result.json"
    target.write_text("precious earlier result", encoding="utf-8")
    with pytest.raises(SummarizerError, match="already exists.*--force"):
        write_output(target, "new")
    assert target.read_text(encoding="utf-8") == "precious earlier result"


def test_force_overwrites(tmp_path):
    target = tmp_path / "result.json"
    target.write_text("old", encoding="utf-8")
    write_output(target, "new", force=True)
    assert target.read_text(encoding="utf-8") == "new"


def test_a_failed_write_leaves_the_old_file_and_no_temporary_file(tmp_path, monkeypatch):
    target = tmp_path / "result.json"
    target.write_text("old", encoding="utf-8")

    def rename_fails(src, dst):
        raise PermissionError(13, "The file is being used by another process")

    monkeypatch.setattr(os, "replace", rename_fails)
    with pytest.raises(SummarizerError, match="can't write"):
        write_output(target, "new", force=True)
    assert target.read_text(encoding="utf-8") == "old"  # untouched
    assert os.listdir(tmp_path) == ["result.json"]  # no stray .tmp


def test_output_path_problems_are_found_before_writing(tmp_path):
    with pytest.raises(SummarizerError, match="is a folder"):
        check_output_path(tmp_path)
    with pytest.raises(SummarizerError, match="folder .* doesn't exist"):
        check_output_path(tmp_path / "no_such_folder" / "result.json")
    assert check_output_path(tmp_path / "new.json") == tmp_path / "new.json"
