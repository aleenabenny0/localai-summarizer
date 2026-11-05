"""Tests for checking the model's JSON reply (summarizer/validation.py)."""

import json

import pytest

from summarizer import config
from summarizer.validation import (
    InvalidOutputError,
    drop_repeats,
    parse_model_output,
    parse_summary_only,
)


def reply(**fields):
    """The JSON text a model would send back, e.g. reply(summary="...")."""
    return json.dumps(fields)


def item(task, owner=None, due=None):
    return {"task": task, "owner": owner, "due": due}


def test_valid_reply_is_parsed_and_cleaned():
    raw = reply(
        action_items=[{"task": "  Send the report ", "owner": "Ana", "due": "Friday"}],
        summary="  The team planned the report.  ",
    )
    assert parse_model_output(raw) == {
        "summary": "The team planned the report.",
        "action_items": [item("Send the report", "Ana", "Friday")],
    }


def test_no_action_items_is_fine():
    raw = reply(action_items=[], summary="Just a status update.")
    assert parse_model_output(raw)["action_items"] == []


@pytest.mark.parametrize(
    "placeholder", [None, "", "  ", "null", "None", "N/A", "unknown", "TBD", "not specified"]
)
def test_placeholder_owner_and_due_become_none(placeholder):
    raw = reply(
        action_items=[{"task": "Fix the bug", "owner": placeholder, "due": placeholder}],
        summary="S.",
    )
    assert parse_model_output(raw)["action_items"] == [item("Fix the bug")]


@pytest.mark.parametrize(
    "raw, problem",
    [
        ("Sure! Here is the summary.", "not valid JSON"),
        ('{"action_items": [{"task": "Send the', "not valid JSON"),  # cut off mid-reply
        ('["a list", "not an object"]', "must be a JSON object"),
        (reply(action_items=[]), '"summary" must be a non-empty string'),
        (reply(action_items=[], summary="   "), '"summary" must be a non-empty string'),
        (reply(summary="S."), '"action_items" must be a list'),
        (reply(action_items="none", summary="S."), '"action_items" must be a list'),
        (reply(action_items=["Send the report"], summary="S."), "must be an object"),
        (reply(action_items=[item("", "Ana")], summary="S."), 'non-empty "task"'),
    ],
)
def test_bad_replies_are_rejected_with_a_reason(raw, problem):
    # The reason matters: it is sent back to the model when we retry.
    with pytest.raises(InvalidOutputError) as error:
        parse_model_output(raw)
    assert problem in str(error.value)


@pytest.mark.parametrize(
    "field, limit",
    [
        ("summary", config.MAX_SUMMARY_CHARS),
        ("task", config.MAX_TASK_CHARS),
        ("owner", config.MAX_FIELD_CHARS),
        ("due", config.MAX_FIELD_CHARS),
    ],
)
def test_too_long_fields_are_rejected_but_the_limit_itself_is_fine(field, limit):
    def reply_with(length):
        values = {"summary": "S.", "task": "Send it", "owner": "Ana", "due": "Friday"}
        values[field] = "x" * length
        return reply(
            action_items=[item(values["task"], values["owner"], values["due"])],
            summary=values["summary"],
        )

    parse_model_output(reply_with(limit))  # exactly at the limit: accepted
    with pytest.raises(InvalidOutputError, match=f"keep it under {limit}"):
        parse_model_output(reply_with(limit + 1))


def test_exact_repeats_are_dropped():
    items = [
        item("Send the report", "Ana"),
        item("send the REPORT", "ana", "Friday"),  # same task + owner, other case: a repeat
        item("Send the report", "Ben"),  # another owner: a different task
    ]
    # The first copy is the one that's kept.
    assert drop_repeats(items) == [item("Send the report", "Ana"), item("Send the report", "Ben")]


def test_reworded_duplicates_are_kept():
    # A known limitation: only exact repeats are removed.
    items = [item("Send the report", "Ana"), item("Email the report", "Ana")]
    assert drop_repeats(items) == items


def test_summary_only_reply():
    assert parse_summary_only(reply(summary=" The whole document. ")) == {"summary": "The whole document."}


def test_summary_only_reply_needs_a_summary():
    with pytest.raises(InvalidOutputError):
        parse_summary_only(reply(summary=""))
