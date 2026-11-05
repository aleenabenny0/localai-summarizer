# LocalAI-Summarizer

Turn messy meeting notes, email threads and transcripts into a short summary and a list of
action items, using a small LLM that runs **on your own PC**. Nothing is sent to the cloud,
and it works with Wi-Fi turned off.

```powershell
python cli.py samples\email_thread.txt
```

```
=== Summary (llama3.2:3b, 1 chunk) ===
The customer webinar has been rescheduled to Thursday Dec 4, 10-11am. The team will work
on the slides, case study, and Q&A session. ...

=== Action items (10) ===
1. Move the platform booking to Dec 4 instead of cancelling it  [owner: Owen Brooks | due: by the end of this week]
2. Send the invite once the booking is moved  [owner: Owen Brooks | due: by the end of this week]
3. Pull adoption numbers from the customer dashboard for the slides  [owner: Farid Haddad | due: by end of next week]
...
```

(An excerpt of a real run. The full run also made mistakes; see [Known limitations](#known-limitations).)

## What it does

- Reads a text file: meeting notes, an email thread, a call transcript.
- Prints a 3-5 sentence **summary** and the **action items**, each with a task, an owner and
  a deadline (when the text names them).
- `--out result.json` saves the result as JSON; `--stats` prints performance numbers (load
  time, prefill and decode speed); `--redact` masks personal data before the model sees it.
- Long documents are split into overlapping chunks, summarized chunk by chunk, then combined.
- Runs the model locally with [Ollama](https://ollama.com): `llama3.2:3b` by default,
  `llama3.2:1b` as a faster, less accurate option.
- Warns when a document contains text that tries to give the model instructions.

## Why on-device

- **Privacy.** Meeting notes and email threads are full of names, customers and plans. Here
  the text never leaves the laptop: the code only talks to Ollama at `127.0.0.1` (this PC),
  and refuses to run if that is changed. See [Security & privacy](#security--privacy).
- **Works offline.** On a plane, on bad hotel Wi-Fi, or on a network that blocks AI services.
- **No cloud cost.** No API key, no account, no per-token bill. The cost is your own CPU time.
- **The trade-off:** a 3B model on a laptop CPU is slower and less capable than a large cloud
  model. The [Performance](#performance) section and the limitations below are honest about that.

## Setup (Windows, PowerShell)

1. **Install Ollama** (it runs the model and starts automatically with Windows):

   ```powershell
   winget install Ollama.Ollama
   ```

   Then open a **new** PowerShell window, so the `ollama` command is found.

2. **Download the models** (one time, about 2.0 GB and 1.3 GB):

   ```powershell
   ollama pull llama3.2:3b
   ollama pull llama3.2:1b
   ```

3. **Set up Python** (3.10 or newer) in the project folder:

   ```powershell
   cd localai-summarizer
   py -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -r requirements.txt          # to run it
   pip install -r requirements-dev.txt      # also tests and pip-audit
   ```

   If PowerShell says "running scripts is disabled", run
   `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then activate again.

## Usage

```powershell
python cli.py samples\meeting_notes.txt                          # summary + action items
python cli.py samples\meeting_notes.txt --model llama3.2:1b      # smaller, faster model
python cli.py samples\transcript.txt --stats                     # long text (3 chunks) + performance numbers
python cli.py samples\email_thread.txt --out result.json         # also save the result as JSON
python cli.py samples\email_thread.txt --out result.json --force # ...overwriting an existing file
python cli.py samples\email_thread.txt --redact                  # mask emails/phones/SSNs before the model sees them
python cli.py big_export.txt --max-mb 5                          # accept a file bigger than the 1 MB default
python cli.py samples\transcript.txt --chunk-tokens 1500         # smaller chunks for this run (see Configuration)
python cli.py samples\transcript.txt --verbose                   # log details: settings, chunks, per-call tokens and timings
python cli.py --watch C:\inbox                                   # keep summarizing .txt files that appear in a folder
python cli.py --check-privacy                                    # is Ollama private to this PC?

python benchmark.py                                              # 1b vs 3b, every sample, 3 runs each
python benchmark.py --runs 1 --models llama3.2:1b                # quick check
python benchmark.py --force                                      # overwrite an existing benchmark_results.md

python -m pytest -q                                              # tests (no Ollama needed)
python -m pytest -m live -v -s                                   # the one test that runs the real model
pip-audit -r requirements.txt                                    # known vulnerabilities in the dependencies
```

If something is missing, the error says exactly how to fix it:

```
Error: Can't reach Ollama at http://127.0.0.1:11434 - it isn't running.
Start it with:   ollama serve
(or open 'Ollama' from the Start menu).
```

```
Error: Model 'llama3.2:3b' isn't downloaded yet. Fix it with:
    ollama pull llama3.2:3b
```

### Watching a folder

`python cli.py --watch C:\inbox` keeps running and summarizes every `.txt` file in that folder,
saving `<name>.summary.json` next to it. It looks every 5 seconds (`--interval`), waits until a
file's size has stopped changing so it never reads half a file, and uses the summary file as
the "done" marker: restarting the watcher never redoes a file, and the no-overwrite rule means
it never replaces an earlier result (`--force` to allow that). A file it can't handle (too big,
binary, a model failure) is logged once and skipped. `--redact` and the other flags apply to
every file. Treat the watched folder as untrusted input: anything dropped there goes to the
model.

### Configuration

Settings come from three layers, each overriding the one before:

1. the built-in defaults in [summarizer/config.py](summarizer/config.py), with the reasoning
   behind each number;
2. [config.toml](config.toml) next to `cli.py`, optional, for the four settings you are most
   likely to change: `model`, `max_input_mb`, `chunk_tokens`, `temperature`;
3. the command-line flags `--model`, `--max-mb`, `--chunk-tokens`, `--temperature`, for one run.

Both layers get the same checks: a misspelled key or an out-of-range value (a chunk that would
not fit in the context window, a temperature of 3) is an error with the allowed range, never
silently ignored.

## Security & privacy

The threat model, what the tool protects against and what it doesn't, is in
**[SECURITY.md](SECURITY.md)**. In short:

| Risk | What the tool does |
|---|---|
| Text leaving the device | Talks only to Ollama on this PC; refuses any other `OLLAMA_HOST`, on every connection. `--check-privacy` verifies the client address, what the Ollama server is listening on, and that Ollama answers. |
| Ollama exposed to the network (`0.0.0.0`) | Detected by `--check-privacy` with the fix; the summarizer refuses to run. |
| Prompt injection | The document is sent between markers as data; the reply must match a JSON schema with length limits or it is rejected; suspicious phrases trigger a warning. The shape is guaranteed; the content of a 3B model's reply is not (see SECURITY.md). |
| Personal data in outputs | `--redact` masks emails, phones and SSN-like numbers *before* inference. Logs (also with `--verbose`) carry metadata only: sizes, counts, timings, never text. A test proves it. |
| Lost or half-written results | `--out` never overwrites without `--force`; files are written atomically. |
| Bad input | Resolved paths, size limit, binary detection, strict UTF-8/UTF-16. |
| Dependencies | Exact pins, a full lock file, `pip-audit`; benchmark results record the model digest. |

```powershell
python cli.py --check-privacy
```

```
=== Privacy check: does the text stay on this PC? ===
[PASS] Client: connects to http://127.0.0.1:11434, an address on this PC (the default; OLLAMA_HOST is not set).
[PASS] Server: ollama.exe, PID 21804 listens on 127.0.0.1:11434, reachable from this PC only.
[PASS] Server: Ollama 0.40.14 answers at http://127.0.0.1:11434.

Result: PASS - Ollama is private to this PC.
```

## How it works

```
text file
  -> refuse to start if OLLAMA_HOST is not this PC, or --out already exists   privacy.py, files.py
  -> read the file: resolved path, size limit, not binary, UTF-8/UTF-16         files.py
  -> --redact: mask emails, phones, SSNs (numbered placeholders)                redact.py
  -> split into overlapping chunks (only if the text is long)                   chunking.py
  -> one model call per chunk: summary + action items as JSON                   core.py -> ollama_client.py -> Ollama
       - the document goes between <<<TEXT>>> markers, as data
       - JSON schema output, temperature 0
       - check the reply (shape, lengths); if invalid, retry once               validation.py
  -> if there were several chunks:
       - summaries: one more model call combines them
       - action items: merged in plain Python
  -> warn if the document contained instructions aimed at an AI                 injection.py
  -> print it; --out saves it atomically; --stats shows the timings             cli.py, stats.py
```

### Getting reliable JSON from a small model

- **Structured outputs.** Each request passes a JSON schema as Ollama's `format`. Ollama turns
  the schema into a grammar, so the model can only produce the right keys and types.
  *Note:* the original spec asked for Ollama's JSON mode (`format="json"`). This uses the
  stricter form of the same feature: plain JSON mode only guarantees valid JSON *syntax*,
  not the keys we need.
- **A cap on the list.** The schema allows at most 20 action items. Without the cap, the 1B
  model sometimes got stuck repeating the same items until it hit the output limit.
- **Temperature 0.** The model always picks the most likely next token, so the same input gives
  (almost) the same output. That suits extraction and makes benchmarks comparable.
- **Our own checks.** Python checks the reply: a non-empty summary, a non-empty task for every
  item, placeholders like "N/A" or "TBD" turned into `null`, and length limits (1,500
  characters for the summary, 300 per task, 100 for owner and due) that stop a model from
  copying the document or its instructions into a field. A reply that hit the 1,024-token
  output limit is incomplete JSON, so it counts as invalid.
- **One retry, then a clear error.** At temperature 0, sending the identical request again would
  most likely return the identical bad reply. So the retry adds a message saying what was
  wrong. If the second reply is also invalid, the program stops with an error instead of
  printing bad data.

### Chunking: how the chunk size was chosen

The model gets a 4,096-token context window (`num_ctx`). The prompt and the answer must both
fit inside it:

| Part of one request | Tokens |
|---|---:|
| System prompt (measured: 274) + chat template + "part 2 of 3" | ~330 |
| Reserved for the answer (`num_predict`) | 1,024 |
| Room for the retry message | ~40 |
| **Left for the document text** | **~2,700** |
| **Chunk size used** | **2,000** (about 1,250 words) |

- **Why 2,000 and not 2,700:** chunks are measured in words, and tokens are estimated as
  1.6 tokens per word. That ratio was measured with Ollama's own token counts on the samples
  (1.44 for the meeting notes, 1.60 for the transcript); the higher one is used. The spare
  ~700 tokens cover text that splits into more tokens than usual (names, numbers, codes).
- **Overlap:** each chunk repeats the last 200 tokens (about 125 words) of the previous one, so
  a sentence cut at a boundary appears complete in one of the two chunks.
- **Why a 4,096-token window when Llama 3.2 supports 128K:** memory. The KV cache (the stored
  keys and values for every token in the context) of the 3B model takes about 112 KB per token:
  roughly 0.45 GB at 4K, but about 14 GB at 128K.

**Map-reduce.** Each chunk is summarized on its own (map), with the note "This is part *n* of
*N*". Then the chunk summaries go to one more model call that writes a single summary (reduce);
if there are too many of them to fit in one call, each half is combined first. The **action
items are merged in Python, not by the model**: exact repeats are removed and the items are
grouped by owner. When the 3B model was asked to merge the lists, it either kept the duplicates
or silently dropped real tasks. A visible duplicate is better than a missing task.

### Performance stats

`--stats` reads the timing metadata that Ollama returns with every response (in nanoseconds):

| Stat | From Ollama | What it means |
|---|---|---|
| Model load time | `load_duration` | loading the model into memory (about 0 if already loaded) |
| Prompt tokens, prefill speed | `prompt_eval_count`, `prompt_eval_duration` | reading the prompt; all prompt tokens are processed in parallel, so it is limited by compute |
| Output tokens, decode speed | `eval_count`, `eval_duration` | writing the answer, one token at a time; every token reads all the weights from memory, so it is limited by memory bandwidth |
| Total time | `total_duration` | everything on the server, including the load |

For a long document (several calls), tokens and times are added up, and the speed is total
tokens / total time, so a big call counts more than a small one.

*Caveat:* if you run the same text twice within a few minutes, Ollama reuses its cached copy of
the prompt and prefill speed looks unrealistically high. The benchmark avoids this (see below).

## Performance

> **Results come from [benchmark_results.md](benchmark_results.md)**, produced by
> `python benchmark.py` on my laptop on 2026-11-05 (3 runs per row, mean ± standard deviation).

Measured on: a laptop with an 11th Gen Intel Core i5-11320H (4 cores / 8 threads, 3.2 GHz),
Intel Iris Xe graphics (not used by Ollama), 16 GB RAM, Windows 11, Ollama 0.40.14, Python
3.13. Everything ran on the CPU.

| Model | Sample | Decode (tok/s) | Total time (s) | Peak RAM (GB) |
|---|---|---:|---:|---:|
| llama3.2:1b | meeting_notes | 20.6 ± 0.8 | 12.5 ± 0.3 | 1.65 ± 0.00 |
| llama3.2:1b | email_thread | 18.9 ± 0.0 | 27.5 ± 0.3 | 1.65 ± 0.00 |
| llama3.2:1b | injection_attempt | 22.3 ± 0.2 | 13.0 ± 0.5 | 1.65 ± 0.00 |
| llama3.2:1b | transcript | 17.0 ± 0.1 | 150.9 ± 1.1 | 1.89 ± 0.02 |
| llama3.2:3b | meeting_notes | 11.8 ± 1.3 | 38.4 ± 4.9 | 2.70 ± 0.00 |
| llama3.2:3b | email_thread | 9.2 ± 0.1 | 76.8 ± 1.4 | 2.70 ± 0.00 |
| llama3.2:3b | injection_attempt | 11.5 ± 0.6 | 26.1 ± 0.7 | 2.70 ± 0.00 |
| llama3.2:3b | transcript | 9.2 ± 0.4 | 233.1 ± 10.5 | 3.48 ± 0.09 |

What the numbers say:

- **The 1B model decodes about twice as fast as the 3B** (roughly 17-22 vs 9-12 tokens/s).
  Decode is memory-bandwidth-bound: every token reads all the weights, and the 3B's weights are
  a bigger file. The price is accuracy: the 1B invents deadlines and picks wrong owners far
  more often, which is why 3B is the default.
- **Decode slows down as the context grows.** Both models are slowest on the transcript (three
  2,000-token chunks), because each new token also reads the KV cache of everything before it.
- **Peak RAM is weights plus KV cache plus Ollama's baseline.** The 1B needs 1.65 GB loaded and
  1.89 GB on the long transcript; the 3B 2.70 GB and 3.48 GB. `ollama ps` reports 1.5 and
  2.6 GB for the same models; the difference is the running process's own memory and the KV
  cache filling up on long inputs. The memory figures are stable to within 0.1 GB across runs.
- **Prefill is far faster than decode** (about 95 vs 65 tokens/s for 1B/3B in the full table,
  against 17-22 and 9-12 for decode), because the whole prompt is processed in parallel.
- **The 3B's timings vary more** (± up to 4.9 s on the meeting notes) than the 1B's. It ran
  second, after 20 minutes of full CPU load, so heat is the likely cause; it isn't proven.

**How the benchmark works:** each model summarizes each file in `samples/` three times. Before
every run, every model Ollama has in memory is unloaded (which clears Ollama's prompt cache and
makes sure a previous model doesn't count towards the memory figure) and the model is loaded
again; the load is reported in its own column and is not part of *Total time*. A background thread samples
the memory of Ollama's processes (server, model runner, tray app) once a second and keeps the
peak: that is *Peak RAM*, and it includes the model's weights, its KV cache and Ollama's own
baseline. Whether the model ran on the CPU or a GPU comes from the same data `ollama ps` shows.
The table reports mean ± standard deviation, and records each model's digest (SHA-256 of the
model file) so the numbers are tied to an exact model version. The models used:

| Model | Parameters | Quantization | Download size |
|---|---|---|---:|
| llama3.2:3b (default) | 3.2B | Q4_K_M (about 4-5 bits per weight) | 2.0 GB |
| llama3.2:1b | 1.2B | Q8_0 (8 bits per weight) | 1.3 GB |

## Verifying it's offline

1. **In the code:** the only network address is Ollama's local one, `http://127.0.0.1:11434`
   by default ([summarizer/config.py](summarizer/config.py)). Ollama's `OLLAMA_HOST` setting
   can change the port, but [summarizer/privacy.py](summarizer/privacy.py) refuses any address
   that is not this PC, so a stray setting can't send your text to another machine. There are
   no API keys or cloud SDKs; the `ollama` package is just an HTTP client for that local server.
2. **With a command:** `python cli.py --check-privacy` checks the client address and what the
   Ollama server is listening on.
3. **In practice:** turn Wi-Fi off (or switch on airplane mode) and run
   `python cli.py samples\meeting_notes.txt`. It works the same.
4. Only *downloading* a model (`ollama pull`) needs the internet, as does `pip-audit`. The
   Ollama app may also go online by itself, for example to check for updates. Summarizing
   never does.

## Tests

```powershell
python -m pytest -q
```

168 tests, which run in a few seconds **without Ollama or a model**. The program talks to the
model through a single function (`ollama_client.chat_json`), and the tests replace it with a
fake that returns prepared replies ([tests/fakes.py](tests/fakes.py)). They cover chunking
(sizes, overlap, nothing lost), JSON validation (good replies, nine kinds of bad ones, length
limits), the retry logic, map-reduce and the action-item merge, the localhost-only rules and
the privacy report, input validation, redaction, the injection markers and warning, safe
output (no overwrite, atomic write), configuration precedence, metadata-only logging, the
performance stats, the memory monitor, a whole benchmark run with a fake model, the folder
watcher, the "how to fix it" error messages, and the command line. Line coverage of the code is 99%; the two
unreached lines are the `if __name__ == "__main__"` entry points. Each core behaviour was also
broken on purpose, in a scratch copy, to check that a test fails.

One more test is marked `live` and skipped by default: `python -m pytest -m live -v -s` runs
the real model on `samples/injection_attempt.txt` and shows what it produced.

## Code quality

```powershell
ruff check .            # linting: unused imports, likely bugs, import order, missing type hints
ruff format .           # formatting, so every file looks the same
mypy                    # are the type hints correct?
```

- **Type hints on every function** in `summarizer/`, `cli.py` and `benchmark.py`. Ruff's `ANN`
  rules enforce that they are *there* ([pyproject.toml](pyproject.toml)), so a new function
  without hints fails the lint; `mypy` checks that they are *right*. Tests are exempt; the
  hints matter in the code they test.
- **CI** ([.github/workflows/ci.yml](.github/workflows/ci.yml)) runs on Windows with Python 3.10
  and 3.13: `ruff check`, `ruff format --check`, `mypy`, the test suite (with the fake model, so
  no Ollama on the runner), and `pip-audit` on the pinned dependencies, which turns the build
  red the day an advisory is published for one of them.

## Project structure

```
localai-summarizer/
├── cli.py                 command line: flags, read the file, print the result
├── benchmark.py           1b vs 3b on every sample -> benchmark_results.md
├── benchmark_results.md   the measured results
├── SECURITY.md            threat model: what is and isn't protected
├── config.toml            optional settings (model, size limit, chunk size, temperature)
├── summarizer/
│   ├── config.py          every setting in one place, plus reading config.toml and the flags
│   ├── core.py            the pipeline: prompts, markers, map-reduce, retry
│   ├── chunking.py        split long text into overlapping chunks
│   ├── validation.py      the JSON schema and the checks on the model's reply
│   ├── ollama_client.py   the only file that talks to Ollama; friendly error messages
│   ├── privacy.py         localhost-only rule and the --check-privacy report
│   ├── files.py           safe reading (validation) and writing (no overwrite, atomic)
│   ├── redact.py          --redact: mask emails, phones, SSNs
│   ├── injection.py       warn about text that gives the model instructions
│   ├── watch.py           --watch: poll a folder, summarize new .txt files
│   ├── stats.py           performance numbers from Ollama's metadata
│   ├── logs.py            logging set-up; the rule: metadata only, never text
│   └── errors.py          SummarizerError: a problem the user can fix
├── samples/               messy test texts: meeting notes, email thread, transcript, injection attempt
├── tests/                 pytest tests with a fake Ollama (+ one live test)
├── .github/workflows/     CI: ruff + pytest + pip-audit on Windows
├── pytest.ini
├── pyproject.toml         ruff settings (line length, rules, type hints required)
├── requirements.txt       runtime dependencies, exact versions
├── requirements-dev.txt   + pytest, ruff, pip-audit
└── requirements.lock      the complete package tree (pip freeze)
```

## Known limitations

- **Injected instructions can become action items.** The output's shape is guaranteed, but a
  3B model may still list an injected command as a task (see SECURITY.md). The warning tells
  you when to look closely.
- **Changes of plan in email threads.** Small models tend to read each email on its own and miss
  that a later one cancels an earlier plan. In one run on `samples/email_thread.txt`, the 3B
  model listed "Send the invite to the team" (due Wednesday morning), although the newest email
  says not to, and "Send the slide template", which the thread says is already done.
- **Reworded duplicates.** Only exact repeats are removed. If a long document repeats a task in
  other words (the transcript ends with a recap), both versions stay; grouping by owner puts
  them next to each other.
- **Wrong owners and invented deadlines** happen, much more often with the 1B model.
- **Redaction is pattern-based**: North American phone/SSN formats; names and addresses stay.
- **At most 20 action items per model call** (the schema cap). A very dense chunk could lose
  items beyond that.
- **Temperature 0 is not perfectly repeatable:** the output can differ slightly between a run
  with a freshly loaded model and one that reuses Ollama's cache.
- **Tiny last chunks:** a text just over one chunk (e.g. 1,251 words) makes a second chunk with
  only one new word, which costs an extra model call.
- **CPU only on the test laptop**, so long documents take minutes. **Plain text input only**
  (no PDF or Word files).
