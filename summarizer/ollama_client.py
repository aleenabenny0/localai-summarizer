"""Thin wrapper around the official `ollama` Python package.

All communication with the Ollama server happens in this file. That gives us:
  * one place to turn low-level errors into friendly "here's the fix" messages;
  * one function (`chat_json`) that tests can swap for a fake, so the tests
    run without Ollama or a model.
"""

import logging
from typing import Any

import ollama

from summarizer import config, privacy
from summarizer.errors import SummarizerError
from summarizer.stats import NS_PER_SECOND

log = logging.getLogger(__name__)  # metadata only, see logs.py


def _client() -> ollama.Client:
    # privacy.ollama_url() refuses any address that is not this PC.
    return ollama.Client(host=privacy.ollama_url())


def chat_json(model: str, messages: list[dict[str, str]], schema: dict[str, Any]) -> ollama.ChatResponse:
    """Send a chat request in JSON mode and return Ollama's full response.

    `schema` is a JSON schema. Passing it as `format` is Ollama's structured
    output mode: plain format="json" only guarantees valid JSON syntax; a
    schema also fixes which keys and types the model may produce.

    The response object holds the model's reply (`response.message.content`),
    why it stopped (`response.done_reason`), and timing metadata.
    """
    log.debug(
        "asking %s (%d message(s), num_ctx %d, temperature %s)",
        model,
        len(messages),
        config.NUM_CTX,
        config.TEMPERATURE,
    )
    try:
        return _client().chat(
            model=model,
            messages=messages,
            format=schema,
            options={
                "temperature": config.TEMPERATURE,
                "num_ctx": config.NUM_CTX,
                "num_predict": config.NUM_PREDICT,
            },
        )
    except (ConnectionError, ollama.ResponseError) as e:
        raise friendly_error(e, model) from None


def load_model(model: str) -> float:
    """Load `model` into memory with our settings; return the load time in seconds.

    Generating a single token is the simplest request that makes Ollama load
    the model AND report how long loading took. It must use the same num_ctx
    as chat_json, otherwise Ollama would reload the model on the next chat.
    """
    try:
        response = _client().generate(
            model=model, prompt="hi", options={"num_ctx": config.NUM_CTX, "num_predict": 1}
        )
    except (ConnectionError, ollama.ResponseError) as e:
        raise friendly_error(e, model) from None
    load_s = (response.load_duration or 0) / NS_PER_SECOND
    log.debug("loaded %s in %.2f s", model, load_s)
    return load_s


def unload_model(model: str) -> None:
    """Remove `model` from memory now (this also empties Ollama's prompt cache)."""
    log.debug("unloading %s", model)
    try:
        _client().generate(model=model, keep_alive=0)
    except (ConnectionError, ollama.ResponseError) as e:
        raise friendly_error(e, model) from None


def loaded_models() -> list[str]:
    """The models Ollama has in memory right now (what `ollama ps` lists).

    Ollama keeps a model loaded for 5 minutes after its last use, so after
    benchmarking one model the next one would share the memory with it. The
    benchmark unloads all of these before each run.
    """
    try:
        return [m.model for m in _client().ps().models if m.model]
    except (ConnectionError, ollama.ResponseError) as e:
        raise friendly_error(e, "") from None


def describe_model(model: str) -> dict[str, Any]:
    """Facts about a model for the benchmark report. Call it while the model is loaded."""
    client = _client()
    details = client.show(model).details  # may be None for an unusual model file
    info: dict[str, Any] = {
        "parameters": details.parameter_size if details else "?",
        "quantization": details.quantization_level if details else "?",
    }

    # The digest is the SHA-256 of the model file: it names the EXACT model, which
    # the tag alone doesn't (after another `ollama pull`, "llama3.2:3b" may point
    # at a different file). The first 12 characters are enough to tell models
    # apart, the same convention Docker uses for image ids.
    for m in client.list().models:
        if m.model == model:
            info["digest"] = (m.digest or "")[:12]

    loaded = [m for m in client.ps().models if m.model == model]
    if loaded:
        size = loaded[0].size or 0
        # size_vram = the part of the model in GPU memory. 0 means it runs on the CPU only.
        info["memory_gb"] = size / 1e9
        info["gpu_percent"] = round(100 * (loaded[0].size_vram or 0) / size) if size else 0
    return info


def ollama_version() -> str:
    """The Ollama server's version, for the benchmark report."""
    return privacy.server_version(privacy.ollama_url()) or "unknown"


def friendly_error(error: ConnectionError | ollama.ResponseError, model: str) -> SummarizerError:
    """Turn a low-level Ollama error into a SummarizerError that says how to fix it."""
    if isinstance(error, ConnectionError):
        # The ollama package raises Python's built-in ConnectionError when
        # nothing is listening on the port, i.e. the Ollama server isn't running.
        return SummarizerError(
            f"Can't reach Ollama at {privacy.ollama_url()} - it isn't running.\n"
            "Start it with:   ollama serve\n"
            "(or open 'Ollama' from the Start menu).\n"
            "Not installed yet? Run:   winget install Ollama.Ollama"
        )
    # Otherwise the server is up but refused the request. 404 = model not downloaded.
    if error.status_code == 404:
        return SummarizerError(f"Model '{model}' isn't downloaded yet. Fix it with:\n    ollama pull {model}")
    return SummarizerError(f"Ollama returned an error: {error.error}")
