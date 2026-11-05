"""The JSON shape we want from the model: a schema for Ollama, and our own checks."""

import json
from typing import Any

from summarizer import config

# One action item as we return it. owner and due are None when the text names none.
ActionItem = dict[str, str | None]

# Passed to Ollama as `format`. Ollama turns this schema into a grammar, so the
# model can only produce JSON with these keys and types, in this order.
# "maxItems" forces the list to end: a small model stuck repeating the same
# items is stopped at MAX_ACTION_ITEMS instead of running until it's cut off.
RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "action_items": {
            "type": "array",
            "maxItems": config.MAX_ACTION_ITEMS,
            "items": {
                "type": "object",
                "properties": {
                    "task": {"type": "string"},
                    "owner": {"type": ["string", "null"]},
                    "due": {"type": ["string", "null"]},
                },
                "required": ["task", "owner", "due"],
            },
        },
        "summary": {"type": "string"},
    },
    "required": ["action_items", "summary"],
}

# For the step that combines the chunk summaries of a long document.
SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {"summary": {"type": "string"}},
    "required": ["summary"],
}


class InvalidOutputError(ValueError):
    """The model's reply wasn't the JSON shape we asked for."""


def parse_model_output(raw: str) -> dict[str, Any]:
    """Parse and check one model reply.

    The schema already fixes the keys and types, but we still check, because:
      * a reply cut off at NUM_PREDICT is incomplete JSON;
      * the schema allows empty strings, and we don't want an empty summary;
      * the schema has no length limits; ours (config.MAX_*_CHARS) stop a
        model that copies the document or its instructions into a field;
      * if the schema were ever not applied, bad output must not slip through.

    Returns {"summary": str, "action_items": [{"task", "owner", "due"}, ...]}.
    Raises InvalidOutputError with a short reason, which is sent back to the
    model when we retry.
    """
    data = _load_json_object(raw)
    items = data.get("action_items")
    if not isinstance(items, list):
        raise InvalidOutputError('"action_items" must be a list (use [] if there are none)')

    cleaned = [_clean_action_item(item) for item in items]
    return {"summary": _clean_summary(data), "action_items": drop_repeats(cleaned)}


def parse_summary_only(raw: str) -> dict[str, Any]:
    """Parse and check a reply that only has a summary: {"summary": "..."}."""
    return {"summary": _clean_summary(_load_json_object(raw))}


def _load_json_object(raw: str) -> dict[str, Any]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise InvalidOutputError(f"the reply is not valid JSON ({e.msg})") from None
    if not isinstance(data, dict):
        raise InvalidOutputError("the reply must be a JSON object")
    return data


def _clean_summary(data: dict[str, Any]) -> str:
    summary = data.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        raise InvalidOutputError('"summary" must be a non-empty string')
    return _within_limit(summary.strip(), config.MAX_SUMMARY_CHARS, '"summary"')


def _clean_action_item(item: object) -> ActionItem:
    if not isinstance(item, dict):
        raise InvalidOutputError('each action item must be an object with a "task"')

    task = item.get("task")
    if not isinstance(task, str) or not task.strip():
        raise InvalidOutputError('each action item needs a non-empty "task" string')

    return {
        "task": _within_limit(task.strip(), config.MAX_TASK_CHARS, 'each "task"'),
        "owner": _optional_text(item.get("owner"), '"owner"'),
        "due": _optional_text(item.get("due"), '"due"'),
    }


# Values models write when they mean "not stated". We store all of them as None.
_EMPTY_VALUES = {"", "null", "none", "n/a", "unknown", "tbd", "tba", "not specified", "not stated"}


def _optional_text(value: object, name: str) -> str | None:
    """Return a cleaned string, or None for missing/placeholder values."""
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in _EMPTY_VALUES:
        return None
    return _within_limit(text, config.MAX_FIELD_CHARS, name)


def _within_limit(text: str, limit: int, name: str) -> str:
    if len(text) > limit:
        raise InvalidOutputError(f"{name} is {len(text)} characters; keep it under {limit}")
    return text


def drop_repeats(items: list[ActionItem]) -> list[ActionItem]:
    """Remove exact repeats (same task text and owner), keeping the first one.

    Small models sometimes list the identical item twice; this is cheap,
    deterministic cleanup. Differently-worded duplicates are left alone.
    """
    seen = set()
    unique = []
    for item in items:
        key = ((item["task"] or "").lower(), (item["owner"] or "").lower())
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique
