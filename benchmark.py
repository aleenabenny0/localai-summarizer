"""Benchmark llama3.2:1b vs llama3.2:3b on every file in samples/.

Each model summarizes each sample RUNS times. Every run:
  1. unloads every model    -> empties Ollama's prompt cache, so an earlier run
     Ollama has in memory      can't make prefill look faster than it really is,
                               and a previous model can't count towards peak RAM
  2. loads the model        -> reported separately as "load time"
  3. summarizes the sample  -> prefill speed, decode speed, total time
while a background thread watches the memory of the Ollama processes and
keeps the peak. The mean (± standard deviation) for each model/sample pair
goes into benchmark_results.md.

Usage:
    python benchmark.py                        # both models, 3 runs (slow on a CPU: ~25 min)
    python benchmark.py --runs 1               # quick check
    python benchmark.py --models llama3.2:1b   # one model only
    python benchmark.py --force                # overwrite an existing benchmark_results.md
"""

import argparse
import logging
import os
import platform
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from summarizer import config, files, ollama_client
from summarizer.chunking import split_into_chunks
from summarizer.core import summarize
from summarizer.errors import SummarizerError
from summarizer.logs import setup_logging
from summarizer.memory import MemoryMonitor

DEFAULT_MODELS = ["llama3.2:1b", "llama3.2:3b"]
SAMPLES_DIR = Path(__file__).parent / "samples"

log = logging.getLogger("summarizer.benchmark")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark local models on the sample files.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="models to compare")
    parser.add_argument("--runs", type=int, default=3, help="runs per model per sample (default 3)")
    parser.add_argument("--output", default="benchmark_results.md", help="where to write the results table")
    parser.add_argument("--force", action="store_true", help="overwrite an existing results file")
    parser.add_argument("--verbose", action="store_true", help="show per-call tokens, timings and loads")
    args = parser.parse_args(argv)
    setup_logging(args.verbose)

    samples = sorted(SAMPLES_DIR.glob("*.txt"))
    rows = []
    model_info = {}
    try:
        # Refuse an existing results file NOW, not after half an hour of runs.
        files.check_output_path(args.output, force=args.force)
        for model in args.models:
            for sample in samples:
                text = sample.read_text(encoding="utf-8")
                runs = []
                for run in range(1, args.runs + 1):
                    log.info("%s | %s | run %d/%d", model, sample.name, run, args.runs)
                    runs.append(benchmark_once(model, text))
                rows.append(make_row(model, sample.stem, text, runs))
            model_info[model] = ollama_client.describe_model(model)  # model is still loaded here
    except SummarizerError as e:
        # Ollama not running or model not downloaded: no point continuing.
        print(f"\nError: {e}", file=sys.stderr)
        return 1

    report = build_report(rows, model_info, args.runs)
    written = files.write_output(args.output, report, force=args.force)
    print(f"\nWrote {written}")
    return 0


def benchmark_once(model: str, text: str) -> dict[str, Any] | None:
    """One measured run. Returns its stats, or None if the model's output was unusable."""
    # Not just `model`: Ollama keeps the previous model in memory for 5 minutes,
    # and it would count towards this run's peak RAM.
    for loaded in ollama_client.loaded_models():
        ollama_client.unload_model(loaded)
    memory = MemoryMonitor().start()  # watches Ollama's memory from before the load to the end
    load_s = ollama_client.load_model(model)
    try:
        result = summarize(text, model=model)
    except SummarizerError as e:
        memory.stop()
        log.warning("  run failed: %s", e)
        return None
    peak_gb = memory.stop() / 1e9
    stats = result["stats"]
    log.info(
        "  decode %.1f tok/s, total %.1f s, peak RAM %.2f GB",
        stats["decode_tokens_per_s"],
        stats["total_s"],
        peak_gb,
    )
    # The model was already loaded when summarize() ran, so use the load we measured.
    return {**stats, "load_s": load_s, "peak_ram_gb": peak_gb}


def make_row(model: str, sample_name: str, text: str, runs: list[dict[str, Any] | None]) -> dict[str, Any]:
    """Turn several runs of one model on one sample into one table row."""
    ok = [r for r in runs if r is not None]
    return {
        "model": model,
        "sample": sample_name,
        "words": len(text.split()),
        "chunks": len(split_into_chunks(text)),
        "load": mean_sd([r["load_s"] for r in ok], digits=2),
        "prefill": mean_sd([r["prefill_tokens_per_s"] for r in ok]),
        "decode": mean_sd([r["decode_tokens_per_s"] for r in ok]),
        "total": mean_sd([r["total_s"] for r in ok]),
        "peak_ram": mean_sd([r["peak_ram_gb"] for r in ok], digits=2),
        "ok_runs": f"{len(ok)}/{len(runs)}",
    }


def mean_sd(values: list[float], digits: int = 1) -> str:
    """Format as 'mean ± standard deviation', e.g. '12.3 ± 0.4'."""
    if not values:
        return "-"
    if len(values) == 1:
        return f"{values[0]:.{digits}f}"
    return f"{statistics.mean(values):.{digits}f} ± {statistics.stdev(values):.{digits}f}"


def build_report(rows: list[dict[str, Any]], model_info: dict[str, dict[str, Any]], runs: int) -> str:
    lines = [
        "# Benchmark results",
        "",
        f"Generated by `python benchmark.py` on {datetime.now():%Y-%m-%d %H:%M}.",
        "",
        "## Setup",
        "",
        f"- CPU: {cpu_name()} ({os.cpu_count()} logical cores)",
        f"- OS: {platform.platform()}",
        f"- Python {platform.python_version()}; Ollama {ollama_client.ollama_version()}; "
        f"context window {config.NUM_CTX} tokens; temperature {config.TEMPERATURE}",
        "",
        "| Model | Parameters | Quantization | Digest | Memory when loaded | Runs on |",
        "|---|---|---|---|---:|---|",
    ]
    for model, info in model_info.items():
        memory = f"{info['memory_gb']:.1f} GB" if "memory_gb" in info else "?"
        where = f"{info['gpu_percent']}% GPU" if info.get("gpu_percent") else "100% CPU"
        lines.append(
            f"| {model} | {info['parameters']} | {info['quantization']} | {info.get('digest', '?')} | "
            f"{memory} | {where} |"
        )
    lines += [
        "",
        "*Digest* = the first 12 characters of the model file's SHA-256 (`ollama list` shows it), "
        "so these results are tied to the exact model version, not just its name.",
    ]

    lines += [
        "",
        "## Method",
        "",
        f"Each model summarized each sample {runs} time(s). Before every run, every model Ollama "
        "had in memory was unloaded (so Ollama's prompt cache is empty and no other model counts "
        "towards *Peak RAM*) and the model was loaded again; that load is the "
        "*Load* column. The model file was usually still in the operating system's file cache, so "
        "this is a reload, not a first load from disk. *Prefill*, *Decode* and *Total time* come from "
        "Ollama's response metadata for the summarization itself (all chunk calls added up, any "
        "retries included), so *Total time* does not include the load. *Peak RAM* is the highest "
        "memory in use by Ollama's processes and everything they started (the server, the model "
        "runner `llama-server.exe` that holds the model, the tray app), sampled once a second from "
        "before the load to the end of the run; it includes the model's weights, its KV cache and "
        "Ollama's own baseline. *Runs on* comes from the same data `ollama ps` "
        "shows: how much of the model sits in GPU memory. Values are mean ± standard deviation "
        "over the successful runs.",
        "",
        "## Results",
        "",
        "| Model | Sample | Words | Chunks | Load (s) | Prefill (tok/s) | Decode (tok/s) "
        "| Total time (s) | Peak RAM (GB) | OK runs |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['sample']} | {r['words']:,} | {r['chunks']} | {r['load']} | "
            f"{r['prefill']} | {r['decode']} | {r['total']} | {r['peak_ram']} | {r['ok_runs']} |"
        )
    return "\n".join(lines) + "\n"


def cpu_name() -> str:
    """A readable CPU name. Windows keeps it in the registry; elsewhere use Python's guess."""
    try:
        import winreg

        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    except (ImportError, OSError):
        return platform.processor() or "unknown CPU"


if __name__ == "__main__":
    sys.exit(main())
