"""
Launch the RAG Internal Agent Streamlit apps.

Run with no arguments for an interactive menu:

  python run.py

Or pick a mode directly (skips the menu):

  python run.py --langchain     # User chat (app.py) + admin        [default]
  python run.py --adk           # experimental Google ADK chat + admin
  python run.py --chat-only     # chat without the admin UI
  python run.py --admin-only    # admin UI only

Extra flags:

  --open        open each app in your browser once it's ready
  -h / --help   show argparse help

Defaults: user chat = 8501, admin = 8502. If a port is taken, the next free
port is used.
"""
import argparse
import os
import signal
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

DEFAULT_USER_PORT = 8501
DEFAULT_ADMIN_PORT = 8502

ROOT = Path(__file__).parent
SRC = ROOT / "src"


def find_free_port(preferred: int, avoid: set[int]) -> int:
    """Return `preferred` if free and not in `avoid`, else an OS-picked port."""
    if preferred not in avoid:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("", preferred))
            sock.close()
            return preferred
        except OSError:
            sock.close()
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            port = s.getsockname()[1]
        if port not in avoid:
            return port


def streamlit_cmd(script: Path, port: int) -> list[str]:
    return [
        sys.executable, "-m", "streamlit", "run",
        str(script),
        "--server.port", str(port),
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
    ]


def port_is_serving(port: int) -> bool:
    """True once something accepts connections on the port (app is up)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


# ── preflight checks ─────────────────────────────────────────────────────────

def check_api_key() -> bool:
    """Warn (don't block) if GOOGLE_API_KEY looks missing or unfilled."""
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except Exception:
        pass
    key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not key or key == "your-key-here":
        print("  ⚠  GOOGLE_API_KEY is not set in .env — the agent and embeddings")
        print("     will fail until you copy .env.example to .env and fill it in.")
        return False
    return True


def check_adk_installed() -> bool:
    """ADK track needs the extra requirements; tell the user how to fix it."""
    import importlib.util
    if importlib.util.find_spec("google.adk") is None:
        print("  ⚠  The ADK track needs extra deps. Install them with:")
        print("       pip install -r requirements-adk.txt")
        return False
    return True


# ── mode selection ───────────────────────────────────────────────────────────

# A mode is (backend, want_admin). backend is "langchain", "adk", or None.
MODES = {
    "1": ("User chat + Admin   (LangChain, default)", ("langchain", True)),
    "2": ("User chat + Admin   (experimental Google ADK)", ("adk", True)),
    "3": ("User chat only      (LangChain)", ("langchain", False)),
    "4": ("ADK chat only       (experimental)", ("adk", False)),
    "5": ("Admin only", (None, True)),
}


def prompt_for_mode() -> tuple[str | None, bool]:
    print("\n" + "─" * 52)
    print("  RAG Internal Agent — what do you want to run?")
    print("─" * 52)
    for key, (label, _) in MODES.items():
        print(f"   {key}) {label}")
    print("   q) Quit")
    print("─" * 52)
    while True:
        try:
            choice = input("  Select [1]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)
        if choice in ("q", "quit", "exit"):
            sys.exit(0)
        if choice == "":
            choice = "1"
        if choice in MODES:
            return MODES[choice][1]
        print("  Please choose one of the listed options.")


def resolve_mode(args: argparse.Namespace) -> tuple[str | None, bool]:
    """Turn CLI flags into (backend, want_admin), or fall back to the menu."""
    if args.admin_only:
        return (None, True)
    backend = "adk" if args.adk else "langchain"
    if args.langchain or args.adk or args.chat_only:
        return (backend, not args.chat_only)
    # No mode flag given → interactive menu.
    return prompt_for_mode()


# ── launch ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch the RAG Internal Agent Streamlit apps.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--langchain", action="store_true",
                       help="User chat (LangChain) + admin")
    group.add_argument("--adk", action="store_true",
                       help="experimental Google ADK chat + admin")
    group.add_argument("--admin-only", action="store_true",
                       help="admin UI only")
    parser.add_argument("--chat-only", action="store_true",
                        help="chat without the admin UI")
    parser.add_argument("--open", dest="open_browser", action="store_true",
                        help="open each app in your browser once it's ready")
    args = parser.parse_args()

    backend, want_admin = resolve_mode(args)

    # Build the list of (label, script) to launch.
    services: list[tuple[str, str]] = []
    if backend == "adk":
        services.append(("ADK chat ", "adk_app.py"))
    elif backend == "langchain":
        services.append(("User chat", "app.py"))
    if want_admin:
        services.append(("Admin UI ", "admin.py"))

    print("\n" + "─" * 52)
    print("  RAG Internal Agent — starting up")
    print("─" * 52)

    # Preflight: surface common setup problems before spawning subprocesses.
    check_api_key()
    if backend == "adk" and not check_adk_installed():
        sys.exit(1)

    # Assign non-colliding ports.
    used: set[int] = set()
    ports: list[int] = []
    for label, _ in services:
        preferred = DEFAULT_ADMIN_PORT if label.startswith("Admin") else DEFAULT_USER_PORT
        port = find_free_port(preferred, used)
        used.add(port)
        ports.append(port)

    procs: list[subprocess.Popen] = []

    def shutdown(sig=None, frame=None):
        print("\nShutting down…")
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    urls: list[str] = []
    for i, ((label, script), port) in enumerate(zip(services, ports), 1):
        preferred = DEFAULT_ADMIN_PORT if label.startswith("Admin") else DEFAULT_USER_PORT
        note = "" if port == preferred else f"  (port {preferred} was taken)"
        procs.append(subprocess.Popen(
            streamlit_cmd(SRC / script, port),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ))
        url = f"http://localhost:{port}"
        urls.append(url)
        print(f"  [{i}/{len(services)}] {label}  {url}{note}")

    # Wait for the apps to start serving, then optionally open browsers.
    print("─" * 52)
    print("  Waiting for apps to come up…", end="", flush=True)
    deadline = time.time() + 30
    pending = set(ports)
    while pending and time.time() < deadline:
        for p in procs:
            if p.poll() is not None:
                print(f"\n  Process exited unexpectedly (code {p.returncode}).")
                shutdown()
        pending = {pt for pt in pending if not port_is_serving(pt)}
        if pending:
            time.sleep(0.5)
    print(" ready." if not pending else " (still starting).")

    if args.open_browser:
        for url in urls:
            webbrowser.open(url)

    print("─" * 52)
    print("  Running. Press Ctrl+C to stop.\n")

    # Wait until any process exits.
    while True:
        for p in procs:
            if p.poll() is not None:
                print(f"\n  Process exited unexpectedly (code {p.returncode}). Shutting down.")
                shutdown()
        time.sleep(1)


if __name__ == "__main__":
    main()
