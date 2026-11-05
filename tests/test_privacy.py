"""Tests for the localhost-only rules (summarizer/privacy.py)."""

from types import SimpleNamespace

import httpx
import psutil
import pytest

from summarizer import ollama_client, privacy
from summarizer.errors import SummarizerError

# --- The two small helpers that talk to the OS and to Ollama -----------------


def test_process_name_handles_unknown_and_vanished_processes(monkeypatch):
    assert privacy.process_name(None) == "unknown program"

    def vanished(pid):
        raise psutil.NoSuchProcess(pid)

    monkeypatch.setattr(psutil, "Process", vanished)
    assert privacy.process_name(4242) == "unknown program"

    monkeypatch.setattr(psutil, "Process", lambda pid: SimpleNamespace(name=lambda: "ollama.exe"))
    assert privacy.process_name(4242) == "ollama.exe"


def test_server_version_reads_ollamas_answer(monkeypatch):
    monkeypatch.setattr(
        httpx, "get", lambda url, timeout: SimpleNamespace(json=lambda: {"version": "0.40.14"})
    )
    assert privacy.server_version("http://127.0.0.1:11434") == "0.40.14"


def test_server_version_is_none_when_nothing_ollama_like_answers(monkeypatch):
    def refused(url, timeout):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "get", refused)
    assert privacy.server_version("http://127.0.0.1:11434") is None

    monkeypatch.setattr(httpx, "get", lambda url, timeout: SimpleNamespace(json=lambda: {"hello": "world"}))
    assert privacy.server_version("http://127.0.0.1:11434") is None  # an answer, but not Ollama's


# --- Where our client connects ----------------------------------------------


def test_default_is_this_pc():
    assert privacy.ollama_url() == "http://127.0.0.1:11434"


@pytest.mark.parametrize(
    "setting, url",
    [
        ("127.0.0.1:11500", "http://127.0.0.1:11500"),  # another port is fine
        ("localhost", "http://localhost:11434"),
        ("http://127.0.0.1:11434", "http://127.0.0.1:11434"),
        ("[::1]:11434", "http://[::1]:11434"),  # IPv6 "this PC"
    ],
)
def test_addresses_on_this_pc_are_accepted(monkeypatch, setting, url):
    monkeypatch.setenv("OLLAMA_HOST", setting)
    assert privacy.ollama_url() == url


@pytest.mark.parametrize(
    "setting",
    [
        "0.0.0.0",  # every network interface
        "http://0.0.0.0:11434",
        "192.168.1.20:11434",  # another PC on the network
        "http://my-nas.local:11434",  # a name, not this PC
        "https://api.example.com",  # a cloud server
    ],
)
def test_other_computers_are_refused(monkeypatch, setting):
    monkeypatch.setenv("OLLAMA_HOST", setting)
    with pytest.raises(SummarizerError, match="Refusing to run"):
        privacy.ollama_url()


def test_nonsense_setting_is_refused(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:abc")
    with pytest.raises(SummarizerError, match="not a valid address"):
        privacy.ollama_url()


def test_model_calls_refuse_too(monkeypatch):
    # The rule applies where it matters: on every connection, not just in --check-privacy.
    monkeypatch.setenv("OLLAMA_HOST", "192.168.1.20:11434")
    with pytest.raises(SummarizerError, match="Refusing to run"):
        ollama_client.chat_json("llama3.2:3b", messages=[], schema={})


# --- What the Ollama server listens on ---------------------------------------


def listener(ip, port=11434, pid=4242):
    """One entry of the operating system's open-port table, shaped like psutil returns it."""
    return SimpleNamespace(status=psutil.CONN_LISTEN, laddr=SimpleNamespace(ip=ip, port=port), pid=pid)


def fake_ports(monkeypatch, *listeners, version="0.40.14"):
    """Pretend these are the open ports, and that Ollama answers with this version (None = no answer)."""
    monkeypatch.setattr(psutil, "net_connections", lambda kind: list(listeners))
    monkeypatch.setattr(privacy, "process_name", lambda pid: "ollama.exe")
    monkeypatch.setattr(privacy, "server_version", lambda url: version)


def statuses(lines):
    return [status for status, _ in lines]


def test_report_passes_for_a_server_private_to_this_pc(monkeypatch):
    fake_ports(monkeypatch, listener("127.0.0.1"))
    lines, ok = privacy.check_privacy()
    assert ok
    assert statuses(lines) == ["PASS", "PASS", "PASS"]
    assert "ollama.exe, PID 4242 listens on 127.0.0.1:11434" in lines[1][1]


def test_report_fails_for_a_server_open_to_the_network(monkeypatch):
    fake_ports(monkeypatch, listener("0.0.0.0"))
    lines, ok = privacy.check_privacy()
    assert not ok
    status, message = lines[1]
    assert status == "FAIL"
    assert "other computers on your network" in message
    assert "Fix:" in message


def test_report_fails_when_the_client_points_elsewhere(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "192.168.1.20:11434")
    lines, ok = privacy.check_privacy()
    assert not ok
    assert statuses(lines) == ["FAIL"]


def test_report_warns_when_ollama_is_not_running(monkeypatch):
    fake_ports(monkeypatch)  # nothing listens on the port
    lines, ok = privacy.check_privacy()
    assert not ok
    assert statuses(lines) == ["PASS", "WARN"]
    assert "isn't running" in lines[1][1]


def test_report_warns_when_the_port_table_cannot_be_read(monkeypatch):
    def denied(kind):
        raise psutil.AccessDenied()

    monkeypatch.setattr(psutil, "net_connections", denied)
    lines, ok = privacy.check_privacy()
    assert not ok
    assert statuses(lines) == ["PASS", "WARN"]
    assert "netstat" in lines[1][1]


def test_report_warns_when_something_else_answers(monkeypatch):
    fake_ports(monkeypatch, listener("127.0.0.1"), version=None)
    lines, ok = privacy.check_privacy()
    assert not ok
    assert statuses(lines) == ["PASS", "PASS", "WARN"]


def test_listening_sockets_ignores_other_ports_and_connections():
    table = [
        listener("127.0.0.1"),
        listener("0.0.0.0", port=445),  # another program, another port
        SimpleNamespace(status="ESTABLISHED", laddr=SimpleNamespace(ip="127.0.0.1", port=11434), pid=1),
    ]
    psutil.net_connections, original = (lambda kind: table), psutil.net_connections
    try:
        assert privacy.listening_sockets(11434) == [("127.0.0.1", 4242)]
    finally:
        psutil.net_connections = original
