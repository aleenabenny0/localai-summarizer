"""Tests for masking personal data (summarizer/redact.py)."""

from pathlib import Path

from summarizer.redact import describe_counts, redact

EMAIL_SAMPLE = Path(__file__).parent.parent / "samples" / "email_thread.txt"


def test_emails_become_numbered_placeholders():
    text = "From: Lena Park <lena.park@example.com>\nTo: owen.brooks@example.com\nCC: Lena.Park@example.com"
    redacted, counts = redact(text)
    # The same address (any casing) always gets the same number.
    assert redacted == "From: Lena Park <[EMAIL-1]>\nTo: [EMAIL-2]\nCC: [EMAIL-1]"
    assert counts == {"EMAIL": 2, "SSN": 0, "PHONE": 0}


def test_phone_numbers_in_common_formats():
    text = "Call +1 (555) 014-2290, 555-014-2290, 555.014.2290 or 5550142290."
    redacted, counts = redact(text)
    assert redacted == "Call [PHONE-1], [PHONE-2], [PHONE-3] or [PHONE-4]."
    assert counts["PHONE"] == 4


def test_ssn_like_numbers():
    redacted, counts = redact("SSN on file: 123-45-6789 (confirmed 123-45-6789)")
    assert redacted == "SSN on file: [SSN-1] (confirmed [SSN-1])"
    assert counts["SSN"] == 1


def test_everyday_numbers_are_left_alone():
    text = (
        "Release 2.4 ships Oct 28. PR #4412, ticket #20931, $4,500 refund, 10-11am, "
        "2026-11-04 22:28, 500-seat license, account 8812-00, version 3.2.1."
    )
    assert redact(text) == (text, {"EMAIL": 0, "SSN": 0, "PHONE": 0})


def test_the_email_sample_is_cleaned():
    redacted, counts = redact(EMAIL_SAMPLE.read_text(encoding="utf-8"))
    assert "@" not in redacted
    assert "014-2290" not in redacted
    assert counts == {"EMAIL": 5, "SSN": 0, "PHONE": 1}
    # The names stay, so the model can still work out who owns each task.
    assert "From: Lena Park <[EMAIL-1]>" in redacted


def test_describe_counts():
    assert (
        describe_counts({"EMAIL": 5, "SSN": 0, "PHONE": 1}) == "5 emails, 0 SSN-like numbers, 1 phone number"
    )
