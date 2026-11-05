"""Split long text into overlapping chunks that fit in the model's context window."""

import math
import re

from summarizer import config


def estimate_tokens(text: str) -> int:
    """Rough token count, from the word count (see TOKENS_PER_WORD in config.py)."""
    return math.ceil(len(text.split()) * config.TOKENS_PER_WORD)


def split_into_chunks(
    text: str, chunk_tokens: int | None = None, overlap_tokens: int | None = None
) -> list[str]:
    """Split `text` into chunks of about `chunk_tokens` tokens.

    Each chunk repeats the last `overlap_tokens` of the previous chunk, so a
    sentence cut in half at a boundary still appears complete in one chunk.
    We split between words, never inside one. Short text -> one chunk.
    """
    chunk_tokens = chunk_tokens or config.CHUNK_TOKENS
    overlap_tokens = overlap_tokens or config.OVERLAP_TOKENS
    if overlap_tokens >= chunk_tokens:
        raise ValueError("overlap must be smaller than the chunk size")

    # Each piece is one word plus the spaces/newlines after it, so joining
    # pieces back together gives the original text, line breaks included.
    words = re.findall(r"\S+\s*", text.strip())

    chunk_words = int(chunk_tokens / config.TOKENS_PER_WORD)
    overlap_words = int(overlap_tokens / config.TOKENS_PER_WORD)
    step = chunk_words - overlap_words  # how far each new chunk's start moves forward

    chunks = []
    for start in range(0, len(words), step):
        chunks.append("".join(words[start : start + chunk_words]).strip())
        if start + chunk_words >= len(words):  # this chunk already reached the end
            break
    return chunks
