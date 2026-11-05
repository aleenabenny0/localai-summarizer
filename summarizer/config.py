"""Every tunable number in one place, so each one is easy to find and explain.

Three layers, each overriding the one before:
  1. the built-in defaults in this file;
  2. config.toml next to cli.py, if it exists (load());
  3. command-line flags (apply(), called by cli.py).
Code reads the values from this module at the moment it needs them
(config.CHUNK_TOKENS, ...), so an override made at start-up applies everywhere.
"""

import logging
import sys
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib  # built into Python 3.11+
else:  # pragma: no cover
    import tomli as tomllib  # the same library as a package, for Python 3.10

from summarizer.errors import SummarizerError

log = logging.getLogger(__name__)

# Ollama runs a small HTTP server on this machine. 127.0.0.1 means "this PC
# only". Ollama's own setting, the OLLAMA_HOST environment variable, may
# change the port, but privacy.py refuses any address that is not this PC,
# so a stray setting can never send your text to another computer.
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"

MODEL = "llama3.2:3b"

# --- Input file --------------------------------------------------------------
# The biggest file we accept. 1 MB of text is about 170,000 words (140 chunks),
# which already takes hours on a CPU; anything bigger is probably a mistake
# (a log file, a database export). The --max-mb flag raises it.
MAX_INPUT_BYTES = 1_000_000

# --- Generation settings sent with every request ---------------------------
TEMPERATURE: float = 0  # always pick the most likely next token -> repeatable output
NUM_CTX = 4096  # context window in tokens (prompt + answer must fit inside)
NUM_PREDICT = 1024  # hard cap on how many tokens the model may generate per answer

# How many times we ask the model before giving up on malformed JSON.
MAX_ATTEMPTS = 2  # the first try + one retry

# Hard cap on the action-item list, enforced by the JSON schema (validation.py).
# 20 items x ~40 tokens each still fits inside NUM_PREDICT.
MAX_ACTION_ITEMS = 20

# --- Limits on the model's reply (checked in validation.py) ------------------
# A reply outside these limits is treated like malformed output: the model is
# asked again, with the limit in the message. They are generous for real
# summaries and action items, but stop a model that starts copying the document
# or its instructions into a field.
MAX_SUMMARY_CHARS = 1500  # 3-5 sentences are usually 400-800
MAX_TASK_CHARS = 300
MAX_FIELD_CHARS = 100  # owner, due

# --- Chunking for long documents (see chunking.py) --------------------------
# One request has to fit inside NUM_CTX = 4096 tokens:
#     ~330  system prompt (274, measured) + chat template + "part 2 of 3" text
#    1024  reserved for the answer (NUM_PREDICT)
#     ~40  room for the retry message
#   ------
#   ~2700  left for the document text itself
# We use 2000, keeping ~700 tokens spare because our token count is an estimate.
CHUNK_TOKENS = 2000
OVERLAP_TOKENS = 200  # ~10%: text cut at a chunk boundary appears whole in the next chunk
PROMPT_OVERHEAD_TOKENS = 400  # the ~330 + ~40 above, rounded up

# We estimate tokens from the word count instead of running the real tokenizer.
# Measured with Llama 3.2's tokenizer (Ollama's prompt_eval_count) on our samples:
# meeting notes 1.44 tokens/word, transcript 1.60. We use the worst case.
TOKENS_PER_WORD = 1.6


def max_chunk_tokens() -> int:
    """The biggest chunk that still fits: the window minus the answer and the prompt overhead (2672)."""
    return NUM_CTX - NUM_PREDICT - PROMPT_OVERHEAD_TOKENS


# --- config.toml and command-line overrides ------------------------------------
CONFIG_FILE = Path(__file__).resolve().parent.parent / "config.toml"
SETTINGS = ("model", "max_input_mb", "chunk_tokens", "temperature")


def load(path: str | Path | None = None) -> dict[str, Any]:
    """Apply the settings in config.toml, if the file exists. Returns the settings it found."""
    path = Path(path or CONFIG_FILE)
    if not path.exists():
        return {}
    try:
        settings = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as e:
        raise SummarizerError(f"{path} is not valid TOML: {e}") from None
    apply(settings, source=str(path))
    log.debug("settings from %s: %s", path, ", ".join(f"{key}={value}" for key, value in settings.items()))
    return settings


def apply(settings: dict[str, Any], source: str) -> None:
    """Check `settings` and write them into this module. `source` says where they came from, for errors."""
    global MODEL, MAX_INPUT_BYTES, CHUNK_TOKENS, TEMPERATURE

    unknown = sorted(set(settings) - set(SETTINGS))
    if unknown:  # a typo must be an error, not silently ignored
        raise SummarizerError(
            f"{source}: unknown setting(s): {', '.join(unknown)}. The settings are: {', '.join(SETTINGS)}"
        )
    if "model" in settings:
        MODEL = _text(settings["model"], "model", source)
    if "max_input_mb" in settings:
        MAX_INPUT_BYTES = int(_number(settings["max_input_mb"], "max_input_mb", source, 0.001, 10_000) * 1e6)
    if "chunk_tokens" in settings:
        CHUNK_TOKENS = int(
            _number(settings["chunk_tokens"], "chunk_tokens", source, OVERLAP_TOKENS + 1, max_chunk_tokens())
        )
    if "temperature" in settings:
        TEMPERATURE = _number(settings["temperature"], "temperature", source, 0, 2)


def _text(value: object, key: str, source: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SummarizerError(f"{source}: {key} must be a non-empty string, not {value!r}")
    return value.strip()


def _number(value: object, key: str, source: str, low: float, high: float) -> float:
    # bool is a kind of int in Python, so `true` would otherwise pass as 1.
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high:
        raise SummarizerError(f"{source}: {key} must be a number between {low} and {high}, not {value!r}")
    return value
