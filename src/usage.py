"""Self-tracked API usage + rate-limit view.

Google exposes NO API for current free-tier consumption, so we count our own
calls. The user chat app and the admin app run as SEPARATE processes, so an
in-memory counter wouldn't be shared — usage is appended to a JSONL file on
disk that both processes read. Aggregates (requests/day, requests/min,
tokens/min) are computed on read against the known free-tier limits below.
"""
import json
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

try:
    from zoneinfo import ZoneInfo
    _PT = ZoneInfo("America/Los_Angeles")  # free-tier RPD resets midnight PT
except Exception:  # pragma: no cover
    from datetime import timezone
    _PT = timezone(timedelta(hours=-7))

_LOG = Path(__file__).parent.parent / "usage_events.jsonl"
_LOCK = threading.Lock()

# Known free-tier limits (from the AI Studio rate-limit dashboard). Models not
# listed are still tracked — their limit just shows as unknown.
LIMITS = {
    "gemini-2.5-flash":            {"rpm": 5,   "tpm": 250000, "rpd": 20},
    "gemini-2.5-flash-lite":       {"rpm": 10,  "tpm": 250000, "rpd": 20},
    "gemini-3.5-flash":            {"rpm": 5,   "tpm": 250000, "rpd": 20},
    "gemini-3.1-flash-lite":       {"rpm": 10,  "tpm": 250000, "rpd": 20},
    "models/gemini-embedding-001": {"rpm": 100, "tpm": 30000,  "rpd": 1000},
}

# Drop events older than this on write so the log can't grow forever.
_RETAIN_SECONDS = 2 * 86400


def record(model: str, kind: str, requests: int = 1, tokens: int = 0) -> None:
    """Append one usage event. `kind` is 'chat' or 'embed'. Never raises — usage
    tracking must not break the actual request path."""
    try:
        line = json.dumps(
            {"t": time.time(), "model": model, "kind": kind,
             "req": int(requests), "tok": int(tokens)}
        )
        with _LOCK:
            with open(_LOG, "a") as f:
                f.write(line + "\n")
    except Exception:
        pass


def _load() -> list[dict]:
    if not _LOG.exists():
        return []
    out: list[dict] = []
    try:
        with open(_LOG) as f:
            for ln in f:
                ln = ln.strip()
                if ln:
                    try:
                        out.append(json.loads(ln))
                    except json.JSONDecodeError:
                        pass
    except OSError:
        pass
    return out


def _day_start_pt_epoch() -> float:
    now = datetime.now(_PT)
    return now.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()


def seconds_to_reset() -> int:
    """Seconds until the next midnight-PT RPD reset."""
    now = datetime.now(_PT)
    nxt = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    return int((nxt - now).total_seconds())


def stats() -> dict[str, dict]:
    """Per-model usage: requests today (rpd), requests last 60s (rpm), tokens
    last 60s (tpm), plus the known limit for each."""
    evts = _load()
    now = time.time()
    day0 = _day_start_pt_epoch()
    minute0 = now - 60
    by: dict[str, dict] = {}
    for e in evts:
        m = e.get("model", "?")
        d = by.setdefault(m, {"rpd": 0, "rpm": 0, "tpm": 0, "kind": e.get("kind", "")})
        req = e.get("req", 1)
        if e["t"] >= day0:
            d["rpd"] += req
        if e["t"] >= minute0:
            d["rpm"] += req
            d["tpm"] += e.get("tok", 0)
    for m, d in by.items():
        d["limit"] = LIMITS.get(m, {})
    return by


try:
    from langchain_core.callbacks import BaseCallbackHandler

    class UsageCallback(BaseCallbackHandler):
        """Records one chat event per LLM call, reading the real model name and
        token counts off the response. Attach to every ChatGoogleGenerativeAI."""

        def on_llm_end(self, response, **kwargs) -> None:
            try:
                gen = response.generations[0][0]
                msg = getattr(gen, "message", None)
                model = "?"
                tokens = 0
                if msg is not None:
                    model = (msg.response_metadata or {}).get("model_name", "?")
                    um = getattr(msg, "usage_metadata", None) or {}
                    tokens = um.get("total_tokens", 0)
                record(model, "chat", requests=1, tokens=tokens)
            except Exception:
                pass

except Exception:  # pragma: no cover — langchain missing
    UsageCallback = None  # type: ignore


def prune() -> None:
    """Rewrite the log keeping only recent events. Cheap; call occasionally."""
    cutoff = time.time() - _RETAIN_SECONDS
    keep = [e for e in _load() if e.get("t", 0) >= cutoff]
    try:
        with _LOCK:
            with open(_LOG, "w") as f:
                for e in keep:
                    f.write(json.dumps(e) + "\n")
    except OSError:
        pass
