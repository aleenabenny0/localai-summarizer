# Security and privacy

LocalAI-Summarizer reads one text file, sends it to a language model running on the
**same PC** (through Ollama), and prints a summary and action items. This page says what
that design protects you from, how, and, just as important, what it does not protect you
from.

## What it protects against

### 1. Your text leaving the device

- The only network address in the code is Ollama's local address. Ollama's own setting,
  the `OLLAMA_HOST` environment variable, is honoured so you can change the port, but
  [summarizer/privacy.py](summarizer/privacy.py) refuses any address that is not this PC
  (`127.0.0.1`, `localhost`, `::1`), and it does so on **every connection**, not just at
  start-up. Pointing it at another machine stops the program with an explanation.
- No API keys, no cloud SDKs. The `ollama` package is an HTTP client for the local server.
- `python cli.py --check-privacy` reports (a) where the client connects, (b) which address
  the Ollama server is listening on, read from the operating system's table of open ports,
  and (c) that it is really Ollama answering there.
- Verified by hand: with Wi-Fi off, the tool works the same. Only `ollama pull` needs the
  internet.

### 2. An Ollama server that is reachable from the network by accident

Ollama listens on `127.0.0.1` by default, but `OLLAMA_HOST=0.0.0.0` makes it accept
requests from **every computer on the network**: anyone on the same Wi-Fi could send it
prompts and read the answers. `--check-privacy` detects a server listening on `0.0.0.0`
(or any non-local address) and prints the fix, and the summarizer refuses to run with that
setting.

### 3. Prompt injection: text that tries to give the model instructions

Three layers, in decreasing order of reliability:

1. **Shape.** The reply must match a JSON schema (Ollama turns it into a grammar the model
   cannot escape) and then pass Python validation with length limits. A reply that fails is
   retried once with the reason, then refused with an error. A failed reply is never
   printed. In testing, the model never managed to reply "PWNED" or dump its prompt.
2. **Prompt.** The document is sent between `<<<TEXT>>>` and `<<<END TEXT>>>` markers with
   the instruction that everything inside is data, never instructions; marker look-alikes
   inside the document are defused so it can't "close" the data section early.
3. **Detection.** Phrases typical of injection ("ignore previous instructions", "system
   prompt", "new instructions:") produce a warning that tells the user to check the result.

**Honest limit:** layer 2 is not reliable with a 3B model. On
[samples/injection_attempt.txt](samples/injection_attempt.txt) the model still turned an
injected "translate everything into French" into an action item, and in one run it listed
the injected payment task. The schema guarantees the *shape* of the output, not that every
item is real. Read the action items before acting on them. `python -m pytest -m live` runs
this sample against the real model so you can see for yourself.

### 4. Sensitive data in logs and output files

- Progress messages carry metadata only: model name, counts, timings, and the reason a reply
  was rejected. Never the text, never the model's reply. The result is printed to the
  terminal and saved only when you ask (`--out`).
- `--redact` masks emails, phone numbers and SSN-like numbers with numbered placeholders
  (`[EMAIL-1]`) **before** the model or any file sees the text. Only the counts are kept.
- `--out` never overwrites an existing file unless you pass `--force`, and it writes
  atomically (temporary file, then rename), so an interrupted run never leaves a
  half-written file.

**Honest limit:** redaction is pattern matching. Phone and SSN formats are North American;
names, street addresses and account numbers are not masked; a 10-digit order number looks
like a phone number.

### 5. Unexpected input and dependencies

- Input files: the path is resolved to one unambiguous location; folders and devices are
  refused; the size limit (1 MB by default, `--max-mb`) is checked from the file's metadata
  before anything is read; binary files are refused; only UTF-8 and UTF-16 are accepted,
  with no guessing, because a wrong guess silently corrupts names and dates.
- Dependencies are pinned to exact versions ([requirements.txt](requirements.txt)), the
  full tree is recorded in [requirements.lock](requirements.lock), and `pip-audit` checks
  them against the Python vulnerability database.
- Benchmark results record each model's digest (SHA-256 of the model file), so numbers are
  tied to an exact model version, not just a name.

## What it does NOT protect against

- **A compromised machine.** Malware running with your user rights can read the input file,
  the terminal, the result file and the memory of the Ollama process. Nothing in this tool
  can help with that.
- **A malicious or tampered model file.** `ollama pull` fetches models from Ollama's
  registry over HTTPS, and the digest lets you confirm the same file was used, but the tool
  does not verify where a model came from, and a model can be trained to behave badly. Only
  pull models you trust.
- **Ollama itself.** Its network behaviour (the Windows app checks for updates), how it
  stores models, and its security fixes are outside this project. Keep it updated.
- **Someone with write access to your disk racing the output check.** "Already exists" is
  checked, then the file is written; a file created in between would be overwritten. The
  no-overwrite rule is for accidents, not attackers.
- **Mistakes in the summary.** The model can miss, invent or misattribute tasks. Temperature
  0, the strict format and the warnings reduce this; they can't remove it.
- **Input on a network drive.** A file on a network share travels over the network to this
  PC when it is read. That is inbound and visible to you, but it is not "the text never
  touches the network".
- **What lands in a watched folder.** `--watch` summarizes every `.txt` file that appears,
  unattended. If other people or programs can write to that folder, they choose the model's
  input, injection attempts included. Watch only folders you control.

## Reporting a problem

Open an issue in the repository. Please don't paste real documents into the report.
