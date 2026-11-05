"""Spotting text that tries to give the model instructions ("prompt injection").

Three layers deal with injected instructions:
  1. the JSON schema and validation.py fix the SHAPE of the reply, whatever the
     document says (this layer is reliable);
  2. the prompt tells the model that the document is data (core.py); a small
     model mostly obeys, but not always;
  3. this file: if the document contains phrases aimed at an AI, we warn the
     user to check the result. We don't remove the phrases: deleting text from
     a document behind the user's back is worse than a warning.
"""

import re

# Phrases that are almost never part of normal meeting notes or emails, but are
# typical of text written to manipulate an AI. Case doesn't matter. In each
# group the longer alternatives come first, so the whole phrase is matched.
WHO = r"(?:ai assistant|language model|assistant|ai|llm)"
SUSPICIOUS_PHRASES = re.compile(
    r"(?:ignore|disregard|forget) (?:all |the |any )?(?:previous|prior|above|earlier) instructions"
    rf"|(?:attention|note to|message to|dear) (?:the |any )?{WHO}"
    rf"|you are (?:now |an? )?{WHO}\b"
    r"|system prompt"
    r"|new instructions?:",
    re.IGNORECASE,
)


def injection_warnings(text: str) -> list[str]:
    """The suspicious phrases found in `text`, lower-cased and without repeats. Empty = nothing found."""
    return sorted({match.group(0).lower() for match in SUSPICIOUS_PHRASES.finditer(text)})
