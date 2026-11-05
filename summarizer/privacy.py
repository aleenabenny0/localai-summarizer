"""Keep the text on this PC.

Two different things could send text to another computer:
  * our CLIENT could be pointed at another machine through OLLAMA_HOST, the
    environment variable Ollama uses for its address;
  * the Ollama SERVER could be listening on every network interface (0.0.0.0),
    so other computers on the network could use it and read what it's given.

ollama_url() rules out the first one every time we connect.
check_privacy() reports on both, for the --check-privacy command.
"""

import ipaddress
import os
from urllib.parse import urlsplit

import httpx
import psutil

from summarizer import config
from summarizer.errors import SummarizerError

DEFAULT_PORT = 11434


def ollama_url() -> str:
    """The address of the Ollama server we talk to, checked to be on this PC.

    OLLAMA_HOST is honoured so a user can change the port, but the address
    must be local. Ollama accepts the variable in several spellings
    ("0.0.0.0", "localhost:11500", "http://127.0.0.1:11434"), so we
    normalize it before checking.
    """
    setting = os.environ.get("OLLAMA_HOST", "").strip()
    raw = setting or config.DEFAULT_OLLAMA_HOST
    if "://" not in raw:
        raw = "http://" + raw
    parts = urlsplit(raw)
    try:
        host, port = parts.hostname, parts.port or DEFAULT_PORT
    except ValueError:  # e.g. a port that isn't a number
        host = None
    if not host or parts.scheme not in ("http", "https"):
        raise SummarizerError(f"OLLAMA_HOST='{setting}' is not a valid address. Use 127.0.0.1:11434.")

    if host in ("0.0.0.0", "::"):
        raise SummarizerError(
            f"Refusing to run: OLLAMA_HOST is '{setting}'. That address means \"every network "
            'interface": Ollama would accept requests from other computers on your network.\n'
            "This tool only works with an Ollama that is private to this PC.\n"
            "Fix: remove the OLLAMA_HOST variable, or set it to 127.0.0.1:11434, then restart Ollama."
        )
    if not is_local(host):
        raise SummarizerError(
            f"Refusing to run: OLLAMA_HOST is '{setting}', which is another computer.\n"
            "This tool only talks to an Ollama server on this PC (127.0.0.1 or localhost), "
            "so your text can never leave it.\n"
            "Fix: remove the OLLAMA_HOST variable, or set it to 127.0.0.1:11434."
        )
    host_part = f"[{host}]" if ":" in host else host  # an IPv6 address needs brackets in a URL
    return f"{parts.scheme}://{host_part}:{port}"


def is_local(host: str) -> bool:
    """True for addresses that can only be reached from this PC."""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback  # 127.x.x.x and ::1
    except ValueError:  # a name such as my-nas.local: some other computer
        return False


def check_privacy() -> tuple[list[tuple[str, str]], bool]:
    """Run the checks for --check-privacy.

    Returns (lines, ok). Each line is a (status, message) pair; status is
    PASS, FAIL or WARN (= could not be checked). ok is True only if every
    line is a PASS.
    """
    lines: list[tuple[str, str]] = []

    # 1. Where does our client connect?
    try:
        url = ollama_url()
    except SummarizerError as e:
        lines.append(("FAIL", f"Client: {e}"))
        return lines, False
    setting = os.environ.get("OLLAMA_HOST")
    source = f"OLLAMA_HOST={setting}" if setting else "the default; OLLAMA_HOST is not set"
    lines.append(("PASS", f"Client: connects to {url}, an address on this PC ({source})."))

    # 2. Which addresses does the Ollama server listen on?
    port = urlsplit(url).port or DEFAULT_PORT  # ollama_url() always includes the port
    try:
        sockets = listening_sockets(port)
    except psutil.AccessDenied:
        lines.append(
            (
                "WARN",
                "Server: couldn't read the list of open ports (permission denied). "
                f"Check by hand with:  netstat -ano | findstr :{port}",
            )
        )
        return lines, False
    if not sockets:
        lines.append(
            (
                "WARN",
                f"Server: nothing is listening on port {port}, so Ollama isn't running and can't "
                "be checked. Start Ollama and run this again.",
            )
        )
        return lines, False
    for ip, pid in sockets:
        who = f"{process_name(pid)}, PID {pid}"
        if is_local(ip):
            lines.append(("PASS", f"Server: {who} listens on {ip}:{port}, reachable from this PC only."))
        else:
            where = "every network interface" if ip in ("0.0.0.0", "::") else f"the network address {ip}"
            lines.append(
                (
                    "FAIL",
                    f"Server: {who} listens on {ip}:{port}, that is {where}: other computers on your "
                    "network can reach it.\n"
                    f"       Fix: remove the OLLAMA_HOST variable (or set it to 127.0.0.1:{DEFAULT_PORT}), "
                    "then quit Ollama from the taskbar tray and start it again.",
                )
            )

    # 3. Is it really Ollama answering there?
    version = server_version(url)
    if version:
        lines.append(("PASS", f"Server: Ollama {version} answers at {url}."))
    else:
        lines.append(
            ("WARN", f"Server: {url} does not answer like Ollama (is another program using port {port}?).")
        )

    return lines, all(status == "PASS" for status, _ in lines)


def listening_sockets(port: int) -> list[tuple[str, int | None]]:
    """(ip, pid) for every program listening on `port`.

    psutil reads the operating system's table of open ports, the same
    information `netstat -ano` prints, and needs no administrator rights on
    Windows for this.
    """
    return [
        (c.laddr.ip, c.pid)
        for c in psutil.net_connections(kind="tcp")  # "tcp" = IPv4 and IPv6
        if c.status == psutil.CONN_LISTEN and c.laddr and c.laddr.port == port
    ]


def process_name(pid: int | None) -> str:
    """The program behind a process id, e.g. 'ollama.exe'."""
    if pid is None:  # the OS wouldn't tell us
        return "unknown program"
    try:
        return psutil.Process(pid).name()
    except psutil.Error:  # the process ended, or we may not look at it
        return "unknown program"


def server_version(url: str) -> str | None:
    """Ollama's version, or None if nothing Ollama-like answers at `url`."""
    try:
        return httpx.get(f"{url}/api/version", timeout=5).json()["version"]
    except (httpx.HTTPError, ValueError, KeyError):
        return None
