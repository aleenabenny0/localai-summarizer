"""Reading the input file and writing the output file safely.

Reading: before any text goes to the model we make sure that what the user
pointed at is what they meant: one ordinary text file, of a sensible size, in
an encoding we can read without guessing.

Writing: never over an existing file unless asked (--force), and never
half-written (write a temporary file, then rename it in one step).
"""

import logging
import math
import os
from pathlib import Path

from summarizer import config
from summarizer.errors import SummarizerError

log = logging.getLogger(__name__)  # metadata only (paths and sizes), see logs.py

UTF16_MARKERS = (b"\xff\xfe", b"\xfe\xff")  # the first two bytes of a UTF-16 file

# A file with NUL bytes (0x00) in its first few KB is almost certainly not
# text: PDFs, Word files, images and zips all have them. This is the same
# quick test git uses to tell text from binary.
BINARY_CHECK_BYTES = 8000


def read_input(path: str | Path, max_bytes: int | None = None) -> str:
    """Read the text file at `path`. Anything that isn't a readable text file raises SummarizerError."""
    max_bytes = max_bytes or config.MAX_INPUT_BYTES
    file = resolve_input_path(path)

    # The size comes from the file's metadata, so a huge file is refused
    # without ever being read into memory.
    size = file.stat().st_size
    if size > max_bytes:
        raise SummarizerError(
            f"{file} is {describe_size(size)}; the limit is {describe_size(max_bytes)}.\n"
            f"That is about {max_bytes // 6:,} words of text, so a bigger file is probably not the one "
            "you meant (a log? an export?).\n"
            f"If it is, raise the limit:  --max-mb {math.ceil(size / 1e6)}"
        )

    log.debug("reading %s (%s)", file, describe_size(size))
    text = decode_text(file.read_bytes(), file)
    if not text.strip():
        raise SummarizerError(f"{file} is empty")
    return text


def resolve_input_path(path: str | Path) -> Path:
    """Turn what the user typed into one unambiguous absolute path to an ordinary file.

    resolve() makes the path absolute, removes any '..' and follows shortcuts
    (symbolic links), so every message shows the real location we would read.
    """
    file = Path(path).expanduser().resolve()
    if not file.exists():
        raise SummarizerError(f"{path} doesn't exist (looked for {file})")
    if file.is_dir():
        raise SummarizerError(f"{file} is a folder, not a file. Give the path of one .txt file.")
    if not file.is_file():  # e.g. Windows device names such as NUL
        raise SummarizerError(
            f"{file} is not an ordinary file (a device or special file). Give the path of one .txt file."
        )
    return file


def decode_text(data: bytes, file: Path) -> str:
    """Bytes -> text. UTF-8 (the normal case) or UTF-16 (what PowerShell 5.1 writes with '>').

    We don't guess other encodings: a wrong guess silently corrupts names,
    numbers and dates, exactly the details we extract. Instead we say how to
    convert the file.
    """
    if data.startswith(UTF16_MARKERS):
        encoding = "utf-16"
    elif b"\x00" in data[:BINARY_CHECK_BYTES]:
        raise SummarizerError(
            f"{file} looks like a binary file (a PDF, Word document, image or archive?), not text.\n"
            "Export or save it as a plain .txt file first."
        )
    else:
        encoding = "utf-8-sig"  # strict UTF-8; "-sig" also removes a UTF-8 marker if there is one

    try:
        return data.decode(encoding)
    except UnicodeDecodeError as e:
        converted = file.with_name(f"{file.stem}-utf8.txt")
        raise SummarizerError(
            f"{file} isn't UTF-8 text: byte 0x{data[e.start]:02x} at position {e.start:,} can't be decoded.\n"
            "If an older Windows program saved it, convert it to UTF-8:\n"
            f'    Get-Content "{file}" | Set-Content -Encoding utf8 "{converted}"\n'
            "(or in Notepad: File > Save as > Encoding: UTF-8)"
        ) from None


def check_output_path(path: str | Path, force: bool = False) -> Path:
    """Make sure `path` can be written later. Called BEFORE the model runs, so a
    problem costs the user a second, not a minute (or 30, for the benchmark)."""
    target = Path(path).expanduser().resolve()
    if target.is_dir():
        raise SummarizerError(f"{target} is a folder. Give a file name, e.g. {target / 'result.json'}")
    if target.exists() and not force:
        raise SummarizerError(
            f"{target} already exists. Use --force to overwrite it, or choose another name."
        )
    if not target.parent.is_dir():
        raise SummarizerError(f"the folder {target.parent} doesn't exist")
    return target


def write_output(path: str | Path, text: str, force: bool = False) -> Path:
    """Write `text` to `path`: never over an existing file unless `force`, and never half-written.

    The text goes into a temporary file in the same folder first; then that
    file is renamed over the final name. The rename is one atomic step on
    Windows and Linux, so if the program is interrupted (Ctrl+C, a crash,
    a power cut) the user has either the old file or the complete new one,
    never a partial one.
    """
    target = check_output_path(path, force)
    temporary = target.with_name(target.name + ".tmp")
    try:
        with open(temporary, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()  # Python's buffer -> the operating system
            os.fsync(f.fileno())  # the operating system's buffer -> the disk
        os.replace(temporary, target)
    except OSError as e:
        temporary.unlink(missing_ok=True)  # don't leave a stray .tmp file behind
        raise SummarizerError(f"can't write {target}: {e.strerror}") from None
    log.debug("wrote %s (%s)", target, describe_size(len(text.encode("utf-8"))))
    return target


def describe_size(n_bytes: int) -> str:
    """1 MB = 1,000,000 bytes, like Windows Explorer's 'Size' column (roughly)."""
    if n_bytes >= 1_000_000:
        return f"{n_bytes / 1e6:.1f} MB"
    return f"{n_bytes / 1e3:.0f} KB"
