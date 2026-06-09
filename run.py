"""
Launch both Streamlit apps (user chat + admin) in one command.

  python run.py            # LangChain chat (app.py) + admin
  python run.py --adk      # experimental Google ADK chat (adk_app.py) + admin

Defaults: user=8501, admin=8502. If a port is taken, the next free port is used.
"""
import signal
import socket
import subprocess
import sys
from pathlib import Path

DEFAULT_USER_PORT = 8501
DEFAULT_ADMIN_PORT = 8502

ROOT = Path(__file__).parent
SRC = ROOT / "src"


def find_free_port(preferred: int) -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("", preferred))
        sock.close()
        return preferred
    except OSError:
        sock.close()
    # preferred taken — let OS pick
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def streamlit_cmd(script: Path, port: int) -> list[str]:
    return [
        sys.executable, "-m", "streamlit", "run",
        str(script),
        "--server.port", str(port),
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
    ]


def main() -> None:
    # --adk swaps the LangChain chat (app.py) for the experimental ADK chat
    # (adk_app.py). The admin UI is unchanged in both modes.
    use_adk = "--adk" in sys.argv[1:]
    chat_script = "adk_app.py" if use_adk else "app.py"
    chat_label = "ADK chat " if use_adk else "User chat"

    user_port = find_free_port(DEFAULT_USER_PORT)
    admin_port = find_free_port(DEFAULT_ADMIN_PORT)

    # Ensure the two ports don't collide when both defaults are taken
    if admin_port == user_port:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            admin_port = s.getsockname()[1]

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

    user_note = "" if user_port == DEFAULT_USER_PORT else f"  (port {DEFAULT_USER_PORT} was taken)"
    admin_note = "" if admin_port == DEFAULT_ADMIN_PORT else f"  (port {DEFAULT_ADMIN_PORT} was taken)"

    print("\n" + "─" * 44)
    print("  RAG Internal Agent — starting up")
    print("─" * 44)

    procs.append(subprocess.Popen(
        streamlit_cmd(SRC / chat_script, user_port),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ))
    print(f"  [1/2] {chat_label}  http://localhost:{user_port}{user_note}")

    procs.append(subprocess.Popen(
        streamlit_cmd(SRC / "admin.py", admin_port),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ))
    print(f"  [2/2] Admin UI    http://localhost:{admin_port}{admin_note}")

    print("─" * 44)
    print("  Both running. Press Ctrl+C to stop.\n")

    # Wait until either process exits
    while True:
        for p in procs:
            if p.poll() is not None:
                print(f"\n  Process exited unexpectedly (code {p.returncode}). Shutting down.")
                shutdown()
        signal.pause() if hasattr(signal, "pause") else __import__("time").sleep(1)


if __name__ == "__main__":
    main()
