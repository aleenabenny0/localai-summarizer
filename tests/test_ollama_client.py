"""Tests for the "here's how to fix it" errors (summarizer/ollama_client.py)."""

from types import SimpleNamespace

import ollama
import pytest

from summarizer import ollama_client
from summarizer.errors import SummarizerError


def error_message(model="llama3.2:3b"):
    """Call chat_json, expect a SummarizerError, return its message."""
    with pytest.raises(SummarizerError) as error:
        ollama_client.chat_json(model, messages=[], schema={})
    return str(error.value)


def fake_generate(monkeypatch, result=None, error=None):
    """Pretend Ollama answered a /api/generate request (used to load and unload models)."""

    def generate(*args, **kwargs):
        if error:
            raise error
        return result

    monkeypatch.setattr(ollama.Client, "generate", generate)


def test_load_model_returns_the_load_time_in_seconds(monkeypatch):
    fake_generate(monkeypatch, result=SimpleNamespace(load_duration=2_500_000_000))  # nanoseconds
    assert ollama_client.load_model("llama3.2:3b") == 2.5


def test_load_model_explains_a_missing_model(monkeypatch):
    fake_generate(monkeypatch, error=ollama.ResponseError("model not found", 404))
    with pytest.raises(SummarizerError, match="ollama pull llama3.2:1b"):
        ollama_client.load_model("llama3.2:1b")


def test_unload_model_explains_a_server_that_is_down(monkeypatch):
    fake_generate(monkeypatch, error=ConnectionError("refused"))
    with pytest.raises(SummarizerError, match="ollama serve"):
        ollama_client.unload_model("llama3.2:3b")


def test_loaded_models_lists_what_ollama_has_in_memory(monkeypatch):
    class FakeClient:
        def ps(self):
            return SimpleNamespace(
                models=[SimpleNamespace(model="llama3.2:1b"), SimpleNamespace(model="llama3.2:3b")]
            )

    monkeypatch.setattr(ollama_client, "_client", FakeClient)
    assert ollama_client.loaded_models() == ["llama3.2:1b", "llama3.2:3b"]


def test_ollama_version_when_nothing_answers(monkeypatch):
    monkeypatch.setattr(ollama_client.privacy, "server_version", lambda url: None)
    assert ollama_client.ollama_version() == "unknown"


def test_ollama_not_running(monkeypatch):
    # A real connection attempt to a port where nothing listens, so this checks
    # the error the real ollama package raises (takes about 2 s on Windows).
    monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:9")
    message = error_message()
    assert "isn't running" in message
    assert "ollama serve" in message


def make_ollama_answer_with(monkeypatch, error):
    """Pretend the Ollama server answered our request with this error."""

    def fake_chat(*args, **kwargs):
        raise error

    monkeypatch.setattr(ollama.Client, "chat", fake_chat)


def test_model_not_downloaded(monkeypatch):
    # Ollama answers 404 when the model hasn't been pulled.
    make_ollama_answer_with(monkeypatch, ollama.ResponseError("model 'llama3.2:3b' not found", 404))
    assert "ollama pull llama3.2:3b" in error_message("llama3.2:3b")


def test_other_ollama_errors_are_passed_on(monkeypatch):
    make_ollama_answer_with(monkeypatch, ollama.ResponseError("out of memory", 500))
    assert "Ollama returned an error: out of memory" in error_message()


def test_describe_model_includes_the_digest(monkeypatch):
    """The facts for the benchmark report, from the three Ollama calls `ollama show/list/ps` use."""

    class FakeClient:
        def show(self, model):
            return SimpleNamespace(
                details=SimpleNamespace(parameter_size="3.2B", quantization_level="Q4_K_M")
            )

        def list(self):
            digest = "a80c4f17acd55265feec403c7aef86be0c25983ab279d83f3bcd3abbcb5b8b72"
            return SimpleNamespace(models=[SimpleNamespace(model="llama3.2:3b", digest=digest)])

        def ps(self):
            return SimpleNamespace(
                models=[SimpleNamespace(model="llama3.2:3b", size=2_600_000_000, size_vram=0)]
            )

    monkeypatch.setattr(ollama_client, "_client", FakeClient)

    assert ollama_client.describe_model("llama3.2:3b") == {
        "parameters": "3.2B",
        "quantization": "Q4_K_M",
        "digest": "a80c4f17acd5",
        "memory_gb": 2.6,
        "gpu_percent": 0,
    }
