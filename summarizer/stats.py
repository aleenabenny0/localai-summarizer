"""Performance numbers from Ollama's response metadata.

Every Ollama response reports how long each phase took, in NANOSECONDS
(1 second = 1,000,000,000 ns):
    load_duration         loading the model into memory (about 0 if already loaded)
    prompt_eval_count     how many prompt tokens were processed ("prefill")
    prompt_eval_duration  time spent on prefill
    eval_count            how many tokens the model generated ("decode")
    eval_duration         time spent on decode
    total_duration        everything, end to end, on the server
"""

from typing import Any

import ollama

NS_PER_SECOND = 1_000_000_000


def call_stats(response: ollama.ChatResponse, label: str) -> dict[str, Any]:
    """Pull the timing fields out of one Ollama response, converted to seconds."""
    return {
        "label": label,
        "load_s": (response.load_duration or 0) / NS_PER_SECOND,
        "prompt_tokens": response.prompt_eval_count or 0,
        "prefill_s": (response.prompt_eval_duration or 0) / NS_PER_SECOND,
        "output_tokens": response.eval_count or 0,
        "decode_s": (response.eval_duration or 0) / NS_PER_SECOND,
        "total_s": (response.total_duration or 0) / NS_PER_SECOND,
    }


def summarize_stats(calls: list[dict[str, Any]], wall_s: float) -> dict[str, Any]:
    """Add up the stats of all model calls in one run.

    A long document makes several calls (one per chunk, plus combining), so
    we sum tokens and times across calls. Speeds are total tokens / total
    time, NOT the average of each call's speed: a 2,000-token call should
    count more than a 50-token one.
    """
    prompt_tokens = sum(c["prompt_tokens"] for c in calls)
    prefill_s = sum(c["prefill_s"] for c in calls)
    output_tokens = sum(c["output_tokens"] for c in calls)
    decode_s = sum(c["decode_s"] for c in calls)
    return {
        "llm_calls": len(calls),
        "load_s": sum(c["load_s"] for c in calls),
        "prompt_tokens": prompt_tokens,
        "prefill_tokens_per_s": tokens_per_second(prompt_tokens, prefill_s),
        "output_tokens": output_tokens,
        "decode_tokens_per_s": tokens_per_second(output_tokens, decode_s),
        "total_s": sum(c["total_s"] for c in calls),
        "wall_s": wall_s,
        "calls": calls,
    }


def tokens_per_second(tokens: int, seconds: float) -> float:
    return tokens / seconds if seconds > 0 else 0.0


def format_stats(stats: dict[str, Any]) -> str:
    """Turn the stats dict into the text block printed by --stats."""
    lines = [
        "=== Performance (from Ollama's response metadata) ===",
        f"Model load time  : {stats['load_s']:8.2f} s",
        f"Prompt tokens    : {stats['prompt_tokens']:8,d}",
        f"Prefill speed    : {stats['prefill_tokens_per_s']:8.1f} tokens/s   (reading the prompt)",
        f"Output tokens    : {stats['output_tokens']:8,d}",
        f"Decode speed     : {stats['decode_tokens_per_s']:8.1f} tokens/s   (writing the answer)",
        f"Total time       : {stats['total_s']:8.2f} s   (Ollama, all calls)",
        f"Wall-clock time  : {stats['wall_s']:8.2f} s   (measured in Python, end to end)",
        f"Model calls      : {stats['llm_calls']:8d}",
    ]
    if stats["llm_calls"] > 1:
        lines.append("")
        lines.append(
            f"  {'call':<18}{'load s':>8}{'prompt tok':>12}{'prefill t/s':>13}"
            f"{'out tok':>9}{'decode t/s':>12}{'total s':>9}"
        )
        for c in stats["calls"]:
            prefill = tokens_per_second(c["prompt_tokens"], c["prefill_s"])
            decode = tokens_per_second(c["output_tokens"], c["decode_s"])
            lines.append(
                f"  {c['label']:<18}{c['load_s']:8.2f}{c['prompt_tokens']:12,d}{prefill:13.1f}"
                f"{c['output_tokens']:9,d}{decode:12.1f}{c['total_s']:9.2f}"
            )
    return "\n".join(lines)
