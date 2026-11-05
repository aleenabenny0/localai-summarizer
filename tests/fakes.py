"""A fake Ollama for the tests, so they run without Ollama or a model.

The whole program talks to the model through ONE function,
ollama_client.chat_json(). install_fake() swaps that function for a fake that
returns replies we wrote ourselves, and records every request it receives.
"""

import json

import ollama

from summarizer import ollama_client


def make_response(content, done_reason="stop"):
    """Build a response shaped like Ollama's.

    `content` is the model's reply text; a dict is turned into JSON text first.
    done_reason "stop" = finished normally; "length" = cut off at NUM_PREDICT.
    The timings are made-up round numbers so the stats are easy to check:
    100 prompt tokens in 1 s (100 tok/s), 20 output tokens in 2 s (10 tok/s).
    """
    if isinstance(content, dict):
        content = json.dumps(content)
    return ollama.ChatResponse(
        model="fake-model",
        done=True,
        done_reason=done_reason,
        message=ollama.Message(role="assistant", content=content),
        load_duration=0,
        prompt_eval_count=100,
        prompt_eval_duration=1_000_000_000,  # Ollama reports nanoseconds: 1 s
        eval_count=20,
        eval_duration=2_000_000_000,  # 2 s
        total_duration=3_000_000_000,  # 3 s
    )


class FakeOllama:
    """Hands out the prepared replies one by one, in order."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []  # what the program sent: one entry per model call

    def chat_json(self, model, messages, schema):
        self.requests.append({"model": model, "messages": messages, "schema": schema})
        if not self.replies:
            raise AssertionError("the program made more model calls than the test expected")
        reply = self.replies.pop(0)
        if isinstance(reply, ollama.ChatResponse):
            return reply  # already a full response, e.g. one that was cut off
        return make_response(reply)


def install_fake(monkeypatch, *replies):
    """Make the program use a FakeOllama with these replies, for one test."""
    fake = FakeOllama(replies)
    # monkeypatch is pytest's tool for replacing something temporarily:
    # the real chat_json is put back automatically when the test ends.
    monkeypatch.setattr(ollama_client, "chat_json", fake.chat_json)
    return fake
