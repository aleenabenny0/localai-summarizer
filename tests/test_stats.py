"""Tests for the performance numbers (summarizer/stats.py)."""

from fakes import make_response

from summarizer.stats import call_stats, format_stats, summarize_stats, tokens_per_second


def test_nanoseconds_become_seconds_and_missing_fields_become_zero():
    response = make_response({"summary": "S.", "action_items": []})  # 1 s prefill, 2 s decode, 3 s total
    response.load_duration = None  # Ollama leaves fields out when they don't apply

    stats = call_stats(response, "whole text")

    assert stats == {
        "label": "whole text",
        "load_s": 0,
        "prompt_tokens": 100,
        "prefill_s": 1.0,
        "output_tokens": 20,
        "decode_s": 2.0,
        "total_s": 3.0,
    }


def test_speeds_are_totals_not_averages_of_averages():
    # A big call and a small call: the combined speed must weight the big one more.
    big = call_stats(make_response("{}"), "part 1/2")
    big.update(prompt_tokens=2000, prefill_s=20.0, output_tokens=300, decode_s=30.0)
    small = call_stats(make_response("{}"), "part 2/2")
    small.update(prompt_tokens=100, prefill_s=2.0, output_tokens=10, decode_s=2.0)

    stats = summarize_stats([big, small], wall_s=60.0)

    assert stats["llm_calls"] == 2
    assert stats["prompt_tokens"] == 2100
    assert stats["prefill_tokens_per_s"] == 2100 / 22.0
    assert stats["decode_tokens_per_s"] == 310 / 32.0
    assert stats["total_s"] == 6.0
    assert stats["wall_s"] == 60.0


def test_no_division_by_zero():
    assert tokens_per_second(0, 0) == 0.0
    assert tokens_per_second(50, 0) == 0.0


def test_format_shows_a_per_call_table_for_several_calls():
    calls = [call_stats(make_response("{}"), "part 1/2"), call_stats(make_response("{}"), "part 2/2")]
    text = format_stats(summarize_stats(calls, wall_s=7.0))

    assert "Decode speed     :     10.0 tokens/s" in text
    assert "Model calls      :        2" in text
    assert "prefill t/s" in text  # the table header...
    assert text.count("part ") == 2  # ...and one row per call


def test_format_has_no_table_for_a_single_call():
    text = format_stats(summarize_stats([call_stats(make_response("{}"), "whole text")], wall_s=3.5))
    assert "Model calls      :        1" in text
    assert "prefill t/s" not in text
