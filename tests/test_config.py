"""Tests for config.toml and the command-line overrides (summarizer/config.py)."""

import pytest
from fakes import install_fake

from cli import main
from summarizer import config
from summarizer.errors import SummarizerError

GOOD_REPLY = {"action_items": [], "summary": "The team planned the report."}


def write_toml(tmp_path, text):
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_no_file_means_the_built_in_defaults(tmp_path):
    assert config.load(tmp_path / "missing.toml") == {}
    assert config.MODEL == "llama3.2:3b"
    assert config.CHUNK_TOKENS == 2000


def test_all_four_settings_are_applied(tmp_path):
    path = write_toml(
        tmp_path, 'model = "llama3.2:1b"\nmax_input_mb = 2.5\nchunk_tokens = 1500\ntemperature = 0.3\n'
    )

    config.load(path)

    assert config.MODEL == "llama3.2:1b"
    assert config.MAX_INPUT_BYTES == 2_500_000
    assert config.CHUNK_TOKENS == 1500
    assert config.TEMPERATURE == 0.3


def test_a_partial_file_only_changes_what_it_names(tmp_path):
    config.load(write_toml(tmp_path, "chunk_tokens = 1000\n"))
    assert config.CHUNK_TOKENS == 1000
    assert config.MODEL == "llama3.2:3b"


def test_a_typo_is_an_error_not_ignored(tmp_path):
    path = write_toml(tmp_path, 'modle = "llama3.2:1b"\n')
    with pytest.raises(
        SummarizerError, match="unknown setting.*modle.*The settings are: model, max_input_mb"
    ):
        config.load(path)


def test_broken_toml_is_an_error(tmp_path):
    path = write_toml(tmp_path, 'model = "llama3.2:1b\n')  # missing closing quote
    with pytest.raises(SummarizerError, match="not valid TOML"):
        config.load(path)


@pytest.mark.parametrize(
    "line, problem",
    [
        ("chunk_tokens = 3000", "between 201 and 2672"),  # wouldn't fit in the context window
        ("chunk_tokens = 100", "between 201 and 2672"),  # smaller than the overlap
        ("temperature = 3", "between 0 and 2"),
        ("temperature = true", "between 0 and 2"),  # a bool is not a number here
        ('temperature = "0"', "between 0 and 2"),  # a string is not a number
        ("max_input_mb = 0", "between 0.001 and 10000"),
        ('model = ""', "non-empty string"),
        ("model = 3", "non-empty string"),
    ],
)
def test_bad_values_are_rejected_with_the_allowed_range(tmp_path, line, problem):
    with pytest.raises(SummarizerError, match=problem):
        config.load(write_toml(tmp_path, line + "\n"))


def test_the_max_chunk_size_follows_the_token_budget():
    # 4096 window - 1024 answer - 400 prompt overhead
    assert config.max_chunk_tokens() == 2672


# --- Precedence: defaults < config.toml < command line -------------------------


def test_command_line_flags_override_config_toml(tmp_path, monkeypatch):
    monkeypatch.setattr(
        config, "CONFIG_FILE", write_toml(tmp_path, 'model = "llama3.2:1b"\nchunk_tokens = 1000\n')
    )
    fake = install_fake(monkeypatch, GOOD_REPLY)
    notes = tmp_path / "notes.txt"
    notes.write_text("Ana will send the report.", encoding="utf-8")

    assert main([str(notes), "--model", "llama3.2:3b"]) == 0

    assert fake.requests[0]["model"] == "llama3.2:3b"  # the flag won
    assert config.CHUNK_TOKENS == 1000  # the file's value stayed where no flag was given


def test_config_toml_is_used_when_no_flag_is_given(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_FILE", write_toml(tmp_path, 'model = "llama3.2:1b"\n'))
    fake = install_fake(monkeypatch, GOOD_REPLY)
    notes = tmp_path / "notes.txt"
    notes.write_text("Ana will send the report.", encoding="utf-8")

    assert main([str(notes)]) == 0

    assert fake.requests[0]["model"] == "llama3.2:1b"


def test_a_bad_flag_value_gets_the_same_check(tmp_path, capsys, monkeypatch):
    install_fake(monkeypatch)  # any model call would fail the test
    notes = tmp_path / "notes.txt"
    notes.write_text("Ana will send the report.", encoding="utf-8")

    assert main([str(notes), "--chunk-tokens", "5000"]) == 1

    assert "the command line: chunk_tokens must be a number between 201 and 2672" in capsys.readouterr().err


def test_a_broken_config_toml_stops_the_cli_with_a_clear_error(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_FILE", write_toml(tmp_path, "chunk_tokens = 'oops'\n"))
    assert main(["--check-privacy"]) == 1
    assert "chunk_tokens must be a number" in capsys.readouterr().err
