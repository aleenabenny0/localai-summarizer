"""Tests for the memory monitor (summarizer/memory.py), with a fake process table."""

import time
from types import SimpleNamespace

import psutil

from summarizer import memory


def fake_process(pid, ppid, name, rss):
    """Looks enough like a psutil.Process for our code: .pid and .info with name, ppid, memory_info."""
    memory_info = SimpleNamespace(rss=rss) if rss is not None else None
    return SimpleNamespace(pid=pid, info={"name": name, "ppid": ppid, "memory_info": memory_info})


def test_ollama_processes_and_their_children_are_counted(monkeypatch):
    table = [
        fake_process(10, 1, "ollama app.exe", 27_000_000),  # the tray app
        fake_process(20, 10, "ollama.exe", 18_000_000),  # the server, started by the tray app
        fake_process(30, 20, "llama-server.exe", 1_850_000_000),  # the model runner, started by the server
        fake_process(31, 30, "conhost.exe", 5_000_000),  # grandchild
        fake_process(40, 1, "chrome.exe", 900_000_000),  # not ours
        fake_process(41, 40, "chrome.exe", 300_000_000),  # not ours either
        fake_process(50, 1, None, 1),  # a process whose name the OS wouldn't tell us
    ]
    monkeypatch.setattr(psutil, "process_iter", lambda attrs: table)
    assert memory.ollama_memory_bytes() == 1_900_000_000


def test_a_process_we_may_not_look_at_is_skipped(monkeypatch):
    table = [fake_process(20, 1, "ollama.exe", 18_000_000), fake_process(30, 20, "llama-server.exe", None)]
    monkeypatch.setattr(psutil, "process_iter", lambda attrs: table)
    assert memory.ollama_memory_bytes() == 18_000_000


def test_no_ollama_means_zero(monkeypatch):
    monkeypatch.setattr(
        psutil, "process_iter", lambda attrs: [fake_process(40, 1, "chrome.exe", 900_000_000)]
    )
    assert memory.ollama_memory_bytes() == 0


def test_monitor_remembers_the_peak(monkeypatch):
    readings = iter([100, 500, 300])  # then 200 forever
    monkeypatch.setattr(memory, "ollama_memory_bytes", lambda: next(readings, 200))

    monitor = memory.MemoryMonitor(interval_s=0.01).start()
    time.sleep(0.1)

    assert monitor.stop() == 500
