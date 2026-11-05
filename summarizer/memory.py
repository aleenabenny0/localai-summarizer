"""Peak memory of Ollama during a benchmark run.

Ollama is several processes: the server (ollama.exe), on Windows the tray app
(ollama app.exe), and a model "runner" that the server starts to hold the
loaded model. The runner's name has changed between Ollama versions
(llama-server.exe in 0.40), so instead of matching names we count every
process named ollama* AND everything those processes started.

"Memory" here is the working set (RSS): the bytes of a process that are in
RAM right now. For the runner that is the model's weights, the KV cache and
its buffers.
"""

import threading

import psutil


def ollama_memory_bytes() -> int:
    """Memory used right now by Ollama's processes and everything they started."""
    # One sweep of the process table: name, parent and memory of every process.
    processes = {p.pid: p.info for p in psutil.process_iter(["name", "ppid", "memory_info"])}

    ollama_pids = {
        pid for pid, info in processes.items() if (info["name"] or "").lower().startswith("ollama")
    }
    # Add children, grandchildren, ...: repeat until a pass finds nothing new.
    found_more = True
    while found_more:
        found_more = False
        for pid, info in processes.items():
            if pid not in ollama_pids and info["ppid"] in ollama_pids:
                ollama_pids.add(pid)
                found_more = True

    # memory_info is None when the OS wouldn't let us look at a process.
    return sum(processes[pid]["memory_info"].rss for pid in ollama_pids if processes[pid]["memory_info"])


class MemoryMonitor:
    """Samples Ollama's memory in a background thread and remembers the peak.

    monitor = MemoryMonitor().start()
    ... load the model, run it ...
    peak_bytes = monitor.stop()
    """

    def __init__(self, interval_s: float = 1.0) -> None:
        self.interval_s = interval_s
        self.peak = 0
        self._stop = threading.Event()
        # daemon=True: if the program exits early, this thread doesn't keep it alive.
        self._thread = threading.Thread(target=self._sample_until_stopped, daemon=True)

    def start(self) -> "MemoryMonitor":
        self._thread.start()
        return self

    def stop(self) -> int:
        """Stop sampling and return the highest value seen, in bytes."""
        self._stop.set()
        self._thread.join()
        self._record()  # one last reading, in case the peak is right now
        return self.peak

    def _sample_until_stopped(self) -> None:
        while not self._stop.is_set():
            self._record()
            self._stop.wait(self.interval_s)  # sleeps, but wakes at once when stop() is called

    def _record(self) -> None:
        self.peak = max(self.peak, ollama_memory_bytes())
