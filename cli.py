"""Command-line entry point.

Usage:
    python cli.py samples\\meeting_notes.txt
    python cli.py samples\\meeting_notes.txt --model llama3.2:1b --out result.json --stats
    python cli.py --watch C:\\inbox          # keep summarizing new .txt files in a folder
    python cli.py --check-privacy
"""

import argparse
import json
import logging
import sys
import textwrap
from pathlib import Path
from typing import Any

from summarizer import config, files, privacy
from summarizer.core import summarize
from summarizer.errors import SummarizerError
from summarizer.logs import setup_logging
from summarizer.redact import describe_counts, redact
from summarizer.stats import format_stats
from summarizer.watch import summary_path, watch_folder

log = logging.getLogger("summarizer.cli")  # metadata only, see summarizer/logs.py


def main(argv: list[str] | None = None) -> int:
    setup_logging(verbose="--verbose" in (argv if argv is not None else sys.argv[1:]))
    try:
        config.load()  # config.toml, if there is one. The flags below override it.
    except SummarizerError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    parser = argparse.ArgumentParser(
        description="Summarize messy text with a local LLM (Ollama). Nothing leaves your PC."
    )
    parser.add_argument(
        "input", nargs="?", help="text file to summarize (meeting notes, email thread, transcript)"
    )
    parser.add_argument(
        "--model",
        default=config.MODEL,
        help="Ollama model to use (default: %(default)s; smaller/faster: llama3.2:1b)",
    )
    parser.add_argument("--out", help="also save the result (including stats) to this JSON file")
    parser.add_argument("--force", action="store_true", help="let --out overwrite an existing file")
    parser.add_argument(
        "--stats", action="store_true", help="print performance stats (load, prefill, decode)"
    )
    parser.add_argument(
        "--redact",
        action="store_true",
        help="mask emails, phone numbers and SSN-like numbers before the model or any file sees the text",
    )
    parser.add_argument(
        "--max-mb",
        type=float,
        default=config.MAX_INPUT_BYTES / 1e6,
        help="biggest input file to accept, in MB (default: %(default)s)",
    )
    parser.add_argument(
        "--chunk-tokens",
        type=int,
        default=config.CHUNK_TOKENS,
        help=f"chunk size for long documents, in tokens (default: %(default)s, max {config.max_chunk_tokens()})",  # noqa: E501
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=config.TEMPERATURE,
        help="0 = repeatable output (default: %(default)s); higher = more varied, up to 2",
    )
    parser.add_argument(
        "--watch",
        metavar="FOLDER",
        help="keep running: summarize every .txt file that appears in FOLDER, "
        "saving <name>.summary.json next to it (Ctrl+C to stop)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="seconds between looks at the --watch folder (default: %(default)s)",
    )
    parser.add_argument(
        "--check-privacy", action="store_true", help="check that Ollama is private to this PC, then exit"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="show details: settings, chunks, per-call tokens and timings, files read and written "
        "(metadata only, never the text)",
    )
    args = parser.parse_args(argv)

    if args.check_privacy:
        return print_privacy_report()
    if args.input and args.watch:
        parser.error("give either a text file or --watch FOLDER, not both")
    if not args.input and not args.watch:
        parser.error("give a text file to summarize, --watch FOLDER, or --check-privacy")

    try:
        # The flags win over config.toml; the same checks apply to both.
        config.apply(
            {
                "model": args.model,
                "max_input_mb": args.max_mb,
                "chunk_tokens": args.chunk_tokens,
                "temperature": args.temperature,
            },
            source="the command line",
        )
        return watch_mode(args) if args.watch else run(args)
    except SummarizerError as e:
        # Every problem the user can fix ends up here, with its "how to fix it" message.
        print(f"\nError: {e}", file=sys.stderr)
        return 1


def run(args: argparse.Namespace) -> int:
    """One summarization, start to finish."""
    # Anything that can be refused is refused now, before the model spends a minute working.
    privacy.ollama_url()  # OLLAMA_HOST points at another computer?
    if args.out:
        files.check_output_path(args.out, force=args.force)  # output file already exists?

    log.info("Summarizing with %s (the first call also loads the model)...", config.MODEL)
    result = summarize_path(args.input, args)

    print_result(result)
    if args.stats:
        print()
        print(format_stats(result["stats"]))
    if args.out:
        saved = files.write_output(
            args.out, json.dumps(result, indent=2, ensure_ascii=False), force=args.force
        )
        print(f"\nSaved to {saved}")
    return 0


def summarize_path(path: str | Path, args: argparse.Namespace) -> dict[str, Any]:
    """Read, optionally redact, and summarize one file. Shared by the one-file and --watch modes."""
    text = files.read_input(path)  # size limit: config.MAX_INPUT_BYTES
    # Progress notes are log messages: they go to stderr, so they never mix with
    # the real output, and they carry metadata only (counts, timings), never the text.
    redacted = None
    if args.redact:
        text, redacted = redact(text)  # from here on, nothing sees the real values
        log.info("Redacted before anything else saw the text: %s.", describe_counts(redacted))
    result = summarize(text, model=config.MODEL)
    result["redacted"] = redacted  # the counts only; None when --redact wasn't used
    return result


def watch_mode(args: argparse.Namespace) -> int:
    """--watch FOLDER: summarize every .txt file that appears there, until Ctrl+C."""
    folder = Path(args.watch).expanduser().resolve()
    if not folder.is_dir():
        raise SummarizerError(f"{folder} is not a folder")
    privacy.ollama_url()  # refuse now if OLLAMA_HOST points at another computer

    def summarize_file(path: Path) -> None:
        output = summary_path(path)
        files.check_output_path(output, force=args.force)  # never clobber an earlier result
        log.info("%s: summarizing with %s...", path.name, config.MODEL)
        result = summarize_path(path, args)
        files.write_output(output, json.dumps(result, indent=2, ensure_ascii=False), force=args.force)
        log.info("%s -> %s (%d action items)", path.name, output.name, len(result["action_items"]))
        if result["warnings"]:  # the count only: the phrases themselves come from the document
            log.warning(
                "%s: contains %d phrase(s) that look like instructions to an AI; check its result",
                path.name,
                len(result["warnings"]),
            )

    log.info(
        "Watching %s for .txt files, looking every %g s. Results go next to each file as "
        "<name>.summary.json. Press Ctrl+C to stop.",
        folder,
        args.interval,
    )
    try:
        watch_folder(folder, summarize_file, interval_s=args.interval)
    except KeyboardInterrupt:
        log.info("Stopped.")
    return 0


def print_privacy_report() -> int:
    """The --check-privacy command. Exit code 0 only if every check passed."""
    lines, ok = privacy.check_privacy()
    print("=== Privacy check: does the text stay on this PC? ===")
    for status, message in lines:
        print(f"[{status}] {message}")
    print("\nResult: PASS - Ollama is private to this PC." if ok else "\nResult: NOT PASSED - see above.")
    return 0 if ok else 1


def print_result(result: dict[str, Any]) -> None:
    chunks = "1 chunk" if result["chunks"] == 1 else f"{result['chunks']} chunks"
    print(f"\n=== Summary ({result['model']}, {chunks}) ===")
    print(textwrap.fill(result["summary"], width=88))

    items = result["action_items"]
    print(f"\n=== Action items ({len(items)}) ===")
    if not items:
        print("(none found)")
    for number, item in enumerate(items, start=1):
        details = []
        if item["owner"]:
            details.append(f"owner: {item['owner']}")
        if item["due"]:
            details.append(f"due: {item['due']}")
        suffix = f"  [{' | '.join(details)}]" if details else ""
        print(f"{number}. {item['task']}{suffix}")

    if result["warnings"]:
        print("\n=== Warning ===")
        print("The document contains text that looks like instructions to an AI:")
        print("  " + ", ".join(f'"{phrase}"' for phrase in result["warnings"]))
        print(
            textwrap.fill(
                "It was treated as data, but a small model can still be misled: check the summary "
                "and the action items for anything nobody actually agreed to.",
                width=88,
            )
        )


if __name__ == "__main__":
    sys.exit(main())
