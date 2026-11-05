"""Tests for benchmark.py, with Ollama faked (the model calls, loads and model facts)."""

import pytest
from fakes import install_fake

import benchmark
from summarizer import memory

GOOD_REPLY = {
    "action_items": [{"task": "Send the report", "owner": "Ana", "due": "Friday"}],
    "summary": "The team planned the report.",
}


def test_existing_results_file_is_refused_at_the_start(tmp_path, capsys, monkeypatch):
    results = tmp_path / "benchmark_results.md"
    results.write_text("30 minutes of earlier measurements", encoding="utf-8")

    # Refused before a single model call. If the check were missing, the benchmark
    # would start for real, so make the first Ollama-related calls fail at once.
    monkeypatch.setattr(
        benchmark.ollama_client, "loaded_models", lambda: pytest.fail("the benchmark started")
    )
    monkeypatch.setattr(
        benchmark.ollama_client, "unload_model", lambda model: pytest.fail("the benchmark started")
    )

    assert benchmark.main(["--output", str(results)]) == 1

    assert "already exists" in capsys.readouterr().err
    assert results.read_text(encoding="utf-8") == "30 minutes of earlier measurements"


def test_mean_sd_formatting():
    assert benchmark.mean_sd([]) == "-"
    assert benchmark.mean_sd([12.34]) == "12.3"
    assert benchmark.mean_sd([10.0, 12.0, 14.0]) == "12.0 ± 2.0"
    assert benchmark.mean_sd([1.2, 1.3], digits=2) == "1.25 ± 0.07"


def test_report_ties_results_to_the_exact_model(monkeypatch):
    monkeypatch.setattr(benchmark.ollama_client, "ollama_version", lambda: "0.40.14")
    info = {
        "llama3.2:3b": {
            "parameters": "3.2B",
            "quantization": "Q4_K_M",
            "digest": "a80c4f17acd5",
            "memory_gb": 2.6,
            "gpu_percent": 0,
        }
    }
    run = {
        "load_s": 1.0,
        "prefill_tokens_per_s": 100.0,
        "decode_tokens_per_s": 10.0,
        "total_s": 30.0,
        "peak_ram_gb": 2.65,
    }
    row = benchmark.make_row("llama3.2:3b", "meeting_notes", "word " * 317, [run])

    report = benchmark.build_report([row], info, runs=1)

    assert "| llama3.2:3b | 3.2B | Q4_K_M | a80c4f17acd5 | 2.6 GB | 100% CPU |" in report
    assert "Ollama 0.40.14" in report
    assert "| Peak RAM (GB) |" in report
    assert "| llama3.2:3b | meeting_notes | 317 | 1 | 1.00 | 100.0 | 10.0 | 30.0 | 2.65 | 1/1 |" in report


def test_make_row_counts_only_successful_runs():
    runs = [
        {
            "load_s": 1.0,
            "prefill_tokens_per_s": 100.0,
            "decode_tokens_per_s": 10.0,
            "total_s": 30.0,
            "peak_ram_gb": 2.5,
        },
        None,  # a failed run
        {
            "load_s": 1.0,
            "prefill_tokens_per_s": 120.0,
            "decode_tokens_per_s": 12.0,
            "total_s": 26.0,
            "peak_ram_gb": 2.7,
        },
    ]
    row = benchmark.make_row("llama3.2:3b", "meeting_notes", "word " * 317, runs)
    assert row["words"] == 317
    assert row["chunks"] == 1
    assert row["decode"] == "11.0 ± 1.4"
    assert row["peak_ram"] == "2.60 ± 0.14"
    assert row["ok_runs"] == "2/3"


# --- A whole benchmark run, with everything Ollama-related faked ----------------


def fake_ollama_server(monkeypatch):
    """Everything benchmark.py asks Ollama for, apart from the model calls (those: install_fake).

    Returns the list of models the benchmark asked to unload, in order.
    """
    unloaded: list[str] = []
    # Pretend a model from an earlier run is still in memory next to the one being measured.
    monkeypatch.setattr(benchmark.ollama_client, "loaded_models", lambda: ["other-model", "fake-model"])
    monkeypatch.setattr(benchmark.ollama_client, "unload_model", unloaded.append)
    monkeypatch.setattr(benchmark.ollama_client, "load_model", lambda model: 1.5)
    monkeypatch.setattr(benchmark.ollama_client, "ollama_version", lambda: "0.40.14")
    facts = {
        "parameters": "1.2B",
        "quantization": "Q8_0",
        "digest": "baf6a787fdff",
        "memory_gb": 1.5,
        "gpu_percent": 0,
    }
    monkeypatch.setattr(benchmark.ollama_client, "describe_model", lambda model: facts)
    monkeypatch.setattr(memory, "ollama_memory_bytes", lambda: 1_650_000_000)  # a steady 1.65 GB
    return unloaded


def test_a_whole_benchmark_run_with_a_fake_model(tmp_path, monkeypatch):
    unloaded = fake_ollama_server(monkeypatch)
    # 4 samples: three take one call each; the transcript takes 3 chunks + 1 combine.
    fake = install_fake(monkeypatch, *[GOOD_REPLY] * 7)
    results = tmp_path / "results.md"

    assert benchmark.main(["--runs", "1", "--models", "fake-model", "--output", str(results)]) == 0

    # Before every run, EVERY loaded model is unloaded, not just the one being measured.
    assert unloaded == ["other-model", "fake-model"] * 4
    report = results.read_text(encoding="utf-8")
    assert len(fake.requests) == 7
    assert "| fake-model | 1.2B | Q8_0 | baf6a787fdff | 1.5 GB | 100% CPU |" in report
    for sample in ["email_thread", "injection_attempt", "meeting_notes", "transcript"]:
        assert f"| fake-model | {sample} |" in report
    assert report.count("| 1.50 | 100.0 | 10.0 |") == 4  # load, prefill, decode: the fake's round numbers
    assert report.count("| 1.65 | 1/1 |") == 4  # peak RAM, OK runs


def test_a_failing_sample_does_not_stop_the_benchmark(tmp_path, monkeypatch, capsys):
    fake_ollama_server(monkeypatch)
    # The first sample (email_thread) gets two bad replies, so its run fails; the rest succeed.
    install_fake(monkeypatch, "not json", "still not json", *[GOOD_REPLY] * 6)
    results = tmp_path / "results.md"

    assert benchmark.main(["--runs", "1", "--models", "fake-model", "--output", str(results)]) == 0

    report = results.read_text(encoding="utf-8")
    assert "run failed" in capsys.readouterr().err
    assert "| fake-model | email_thread | 838 | 1 | - | - | - | - | - | 0/1 |" in report
    assert report.count("| 1/1 |") == 3


def test_cpu_name_falls_back_when_the_registry_is_unavailable(monkeypatch):
    winreg = pytest.importorskip("winreg")  # Windows only; elsewhere the ImportError path runs anyway

    def no_registry(*args):
        raise OSError("no registry")

    monkeypatch.setattr(winreg, "OpenKey", no_registry)
    assert benchmark.cpu_name()  # some non-empty name, never a crash
