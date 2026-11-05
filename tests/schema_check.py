"""The shape every result must have, whatever the input text tried to make the model do."""

from summarizer import config


def assert_matches_schema(result):
    summary = result["summary"]
    assert isinstance(summary, str)
    assert 0 < len(summary) <= config.MAX_SUMMARY_CHARS

    items = result["action_items"]
    assert isinstance(items, list)
    assert len(items) <= config.MAX_ACTION_ITEMS
    for item in items:
        assert set(item) == {"task", "owner", "due"}
        assert isinstance(item["task"], str)
        assert 0 < len(item["task"]) <= config.MAX_TASK_CHARS
        for field in ("owner", "due"):
            value = item[field]
            assert value is None or (isinstance(value, str) and 0 < len(value) <= config.MAX_FIELD_CHARS)
