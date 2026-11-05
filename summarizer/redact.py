"""Masking personal data before the text goes anywhere (the --redact flag).

With --redact, the model only ever sees placeholders such as [EMAIL-1], and so
does anything we save. The placeholders are numbered, so the model can still
tell that two mentions are the same address without seeing the address. The
real values are not kept anywhere: redact() returns only how many it masked.

Limits, honestly: these are patterns, not understanding. Phone and SSN formats
are North American; people's names, addresses and account numbers are not
masked; a 10-digit order number looks like a phone number and will be masked.
"""

import re

# Checked in this order. Emails first, because they can contain digits that
# the phone pattern might otherwise pick up.
PATTERNS = {
    "EMAIL": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # 123-45-6789
    # +1 (555) 014-2290, 555-014-2290, 555.014.2290, 5550142290. (?<!\d) and
    # \b stop it from matching the middle of a longer run of digits.
    "PHONE": re.compile(r"(?<!\d)(?:\+?\d{1,3}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b"),
}


def redact(text: str) -> tuple[str, dict[str, int]]:
    """Replace emails, SSN-like numbers and phone numbers with numbered placeholders.

    Returns (redacted_text, counts); counts is like {"EMAIL": 5, "SSN": 0, "PHONE": 1},
    the number of DIFFERENT values masked of each kind. Safe to log: no values in it.
    """
    counts = {}
    for kind, pattern in PATTERNS.items():
        text, counts[kind] = _mask(text, kind, pattern)
    return text, counts


def _mask(text: str, kind: str, pattern: re.Pattern[str]) -> tuple[str, int]:
    numbers: dict[str, int] = {}  # value -> its number, so the same value always gets the same placeholder

    def placeholder(match: re.Match[str]) -> str:
        number = numbers.setdefault(match.group(0).lower(), len(numbers) + 1)
        return f"[{kind}-{number}]"

    return pattern.sub(placeholder, text), len(numbers)


def describe_counts(counts: dict[str, int]) -> str:
    """'5 emails, 0 SSN-like numbers, 1 phone number' for the progress note."""
    names = {
        "EMAIL": ("email", "emails"),
        "SSN": ("SSN-like number", "SSN-like numbers"),
        "PHONE": ("phone number", "phone numbers"),
    }
    parts = []
    for kind, count in counts.items():
        singular, plural = names[kind]
        parts.append(f"{count} {singular if count == 1 else plural}")
    return ", ".join(parts)
