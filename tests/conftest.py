"""Settings pytest applies to every test (pytest looks for this file name)."""

from pathlib import Path

import pytest

from summarizer import config


@pytest.fixture(autouse=True)
def no_ollama_host_setting(monkeypatch):
    """Start every test with OLLAMA_HOST unset, whatever this PC has in its environment.

    Tests that want a particular setting set it themselves.
    """
    monkeypatch.delenv("OLLAMA_HOST", raising=False)


@pytest.fixture(autouse=True)
def isolated_config(monkeypatch):
    """Keep the settings the same for every test.

    config.toml and the command-line flags write into the config module, so a
    test that changes them must not leak into the next one. We point
    CONFIG_FILE at a file that doesn't exist (so the project's own config.toml
    is ignored) and put every value back afterwards.
    """
    monkeypatch.setattr(config, "CONFIG_FILE", Path(__file__).parent / "no-such-config.toml")
    saved = {name: getattr(config, name) for name in dir(config) if name.isupper()}
    yield
    for name, value in saved.items():
        setattr(config, name, value)
