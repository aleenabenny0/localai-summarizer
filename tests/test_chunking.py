"""Tests for splitting long text into overlapping chunks (summarizer/chunking.py).

With the settings in config.py (CHUNK_TOKENS 2000, OVERLAP_TOKENS 200,
TOKENS_PER_WORD 1.6) a chunk is 1,250 words and the overlap is 125 words,
so each new chunk starts 1,125 words after the previous one.
"""

import pytest

from summarizer import config
from summarizer.chunking import estimate_tokens, split_into_chunks


def numbered_words(count):
    """'w0 w1 w2 ...': every word is different, so we can see exactly where chunks start and end."""
    return " ".join(f"w{n}" for n in range(count))


def test_short_text_is_one_chunk():
    text = numbered_words(300)
    assert split_into_chunks(text) == [text]


def test_empty_text_gives_no_chunks():
    assert split_into_chunks("") == []
    assert split_into_chunks("  \n\n  ") == []


def test_long_text_is_split_into_chunks_of_1250_words():
    chunks = split_into_chunks(numbered_words(3000))
    # Chunks start at words 0, 1125 and 2250; the last one runs to the end.
    assert [len(chunk.split()) for chunk in chunks] == [1250, 1250, 750]


def test_consecutive_chunks_share_125_words():
    chunks = split_into_chunks(numbered_words(3000))
    for previous, current in zip(chunks, chunks[1:], strict=False):
        assert previous.split()[-125:] == current.split()[:125]


def test_no_words_are_lost():
    chunks = split_into_chunks(numbered_words(3000))
    assert chunks[0].split()[0] == "w0"
    assert chunks[-1].split()[-1] == "w2999"
    words_seen = {word for chunk in chunks for word in chunk.split()}
    assert len(words_seen) == 3000


def test_text_of_exactly_one_chunk_is_not_split():
    assert len(split_into_chunks(numbered_words(1250))) == 1
    assert len(split_into_chunks(numbered_words(1251))) == 2


def test_every_chunk_fits_the_token_budget():
    for chunk in split_into_chunks(numbered_words(5000)):
        assert estimate_tokens(chunk) <= config.CHUNK_TOKENS


def test_line_breaks_are_kept():
    text = "Agenda:\n- budget\n- hiring\n\nNotes: all good"
    assert split_into_chunks(text) == [text]


def test_overlap_must_be_smaller_than_the_chunk():
    with pytest.raises(ValueError):
        split_into_chunks("some text", chunk_tokens=200, overlap_tokens=200)


def test_estimate_tokens_uses_tokens_per_word():
    # 5 words x 1.6 tokens per word = 8 tokens
    assert estimate_tokens("one two three four five") == 8
