"""Provider chain for the asinccontext compressor.

Fast free models first (with warm-up + cooldown), the paid model last.
Only stdlib + httpx; no imports of other plugin modules.

API keys must never leak into exceptions, meta, health or logs.
"""

import contextlib
import fcntl
import json
import os
import re
import time
import uuid
from urllib.parse import urlparse

import httpx

WARM_TTL = 600.0
COOLDOWNS = (60.0, 300.0, 1800.0)
WARMUP_TIMEOUT = 20.0
DEFAULT_TIMEOUT = 120.0
REQUIRED_HEADERS = (
    "## Historical Task Snapshot",
    "## Goal",
    "## Constraints & Preferences",
    "## Completed Actions",
)


class ChainError(Exception):
    """Chain failure with a short, key-free reason."""

    def __init__(self, reason):
        self.reason = str(reason)
        super().__init__(self.reason)


def health_path() -> str:
    env = os.environ.get("ASINC_HEALTH_PATH")
    if env:
        return env
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "health.json")


def load_health() -> dict:
    try:
        with open(health_path(), "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {"entries": {}, "updated": 0.0}
        entries = data.get("entries")
        if not isinstance(entries, dict):
            data["entries"] = {}
        if "updated" not in data:
            data["updated"] = 0.0
        return data
    except Exception:
        return {"entries": {}, "updated": 0.0}


def save_health(h) -> None:
    try:
        h["updated"] = time.time()
        path = health_path()
        tmp = path + ".tmp.%d" % os.getpid()
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(h, f)
        os.replace(tmp, path)
    except Exception:
        try:
            if os.path.exists(tmp):
                os.unlink(tmp)
        except Exception:
            pass


@contextlib.contextmanager
def _health_lock():
    """Cross-process flock around the read-modify-write cycle of the health file (worker, warmer, several sessions).

    Never blocks the work on failure: if the lock cannot be taken, run without it (health is auxiliary).
    """
    fd = None
    try:
        fd = os.open(health_path() + ".wlock", os.O_CREAT | os.O_RDWR, 0o666)
        fcntl.flock(fd, fcntl.LOCK_EX)
    except Exception:
        pass
    try:
        yield
    finally:
        try:
            if fd is not None:
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)
        except Exception:
            pass


def is_available(name, h, now) -> bool:
    entry = (h.get("entries") or {}).get(name)
    if not entry:
        return True
    return now >= float(entry.get("skip_until") or 0.0)


def is_fresh(name, h, now, ttl=None) -> bool:
    entry = (h.get("entries") or {}).get(name)
    if not entry:
        return False
    if not entry.get("ok"):
        return False
    last_ok = entry.get("last_ok")
    if last_ok is None:
        return False
    return (now - float(last_ok)) <= (WARM_TTL if ttl is None else ttl)


def mark_ok(name, lat, now=None) -> None:
    now = time.time() if now is None else now
    with _health_lock():
        h = load_health()
        entries = h.setdefault("entries", {})
        entries[name] = {
            "ok": True,
            "last_ok": now,
            "last_fail": (entries.get(name) or {}).get("last_fail"),
            "fails": 0,
            "skip_until": 0.0,
            "lat": lat,
            "reason": "",
        }
        save_health(h)


def mark_warm(name, lat, now=None) -> None:
    """Successful WARM-UP: marks the model alive and fresh but does NOT reset fails.

    Only a successful real call (mark_ok) resets the counter. Otherwise a model that passes
    warm-up but fails real work would always get the minimal cool-down with no escalation.
    """
    now = time.time() if now is None else now
    with _health_lock():
        h = load_health()
        entries = h.setdefault("entries", {})
        prev = entries.get(name) or {}
        entries[name] = {
            "ok": True,
            "last_ok": now,
            "last_fail": prev.get("last_fail"),
            "fails": int(prev.get("fails") or 0),
            "skip_until": 0.0,
            "lat": lat,
            "reason": prev.get("reason") or "",
        }
        save_health(h)


def mark_fail(name, reason, now=None) -> None:
    now = time.time() if now is None else now
    reason = (reason or "")[:120]
    with _health_lock():
        h = load_health()
        entries = h.setdefault("entries", {})
        prev = entries.get(name) or {}
        fails = int(prev.get("fails") or 0) + 1
        cooldown = COOLDOWNS[min(fails - 1, len(COOLDOWNS) - 1)]
        if "429" in reason:
            cooldown = min(cooldown, COOLDOWNS[0])
        entries[name] = {
            "ok": False,
            "last_ok": prev.get("last_ok"),
            "last_fail": now,
            "fails": fails,
            "skip_until": now + cooldown,
            "lat": prev.get("lat", 0.0),
            "reason": reason,
        }
        save_health(h)


def _host_of(base_url):
    try:
        return urlparse(base_url or "").netloc or ""
    except Exception:
        return ""


def build_chain(cfg) -> list:
    cfg = cfg or {}
    chain = []
    seen = set()
    for e in (cfg.get("free_chain") or []):
        if not isinstance(e, dict):
            continue
        model = e.get("model")
        base_url = e.get("base_url")
        if not model or not base_url:
            continue
        name = e.get("name") or "%s:%s" % (_host_of(base_url), model)
        if name in seen:
            continue
        seen.add(name)
        chain.append({
            "name": name,
            "model": model,
            "base_url": base_url,
            "api_key": e.get("api_key"),
            "timeout": float(e.get("timeout") or DEFAULT_TIMEOUT),
            "headers": e.get("headers") if isinstance(e.get("headers"), dict) else None,
            "unique_session": bool(e.get("unique_session")),
            "reasoning_effort": e.get("reasoning_effort"),
            "paid": False,
        })
    model = cfg.get("model")
    if model:
        name = "primary:" + str(model)
        if name not in seen:
            seen.add(name)
            chain.append({
                "name": name,
                "model": model,
                "base_url": cfg.get("base_url") or "https://openrouter.ai/api/v1",
                "api_key": cfg.get("api_key"),
                "timeout": float(cfg.get("http_timeout") or cfg.get("timeout") or 600),
                "paid": True,
            })
    return chain


_THINK_RE = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.S | re.I)
_THINKING_RE = re.compile(r"<thinking\b[^>]*>.*?</thinking\s*>", re.S | re.I)


def sections_ok(text) -> bool:
    if not text:
        return False
    stripped = _THINK_RE.sub(" ", text)
    stripped = _THINKING_RE.sub(" ", stripped)
    hits = sum(1 for h in REQUIRED_HEADERS if h in stripped)
    return hits >= 3


def _headers(entry):
    h = {"Content-Type": "application/json"}
    if entry.get("api_key"):
        h["Authorization"] = "Bearer %s" % entry["api_key"]
    if isinstance(entry.get("headers"), dict):
        h.update({str(k): str(v) for k, v in entry["headers"].items()})
    if entry.get("unique_session"):
        # Some web proxies bind a chat/account to x-agent-session; without the header all clients of the machine
        # share one chat. A unique id per call = a fresh chat, other conversations never mix.
        h["x-agent-session"] = "asinc-" + uuid.uuid4().hex[:12]
    return h

def _url(entry):
    return entry["base_url"].rstrip("/") + "/chat/completions"


def _extract_message(data, allow_reasoning=True):
    try:
        choices = data.get("choices") or []
        if not choices:
            return None, None
        msg = choices[0].get("message") or {}
        content = msg.get("content")
        finish = choices[0].get("finish_reason")
        if not content and allow_reasoning:
            alt = msg.get("reasoning_content") or msg.get("reasoning")
            if alt:
                content = alt
        return content, finish
    except Exception:
        return None, None


def warmup_entry(entry, timeout=WARMUP_TIMEOUT):
    body = {
        "model": entry["model"],
        "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
        "max_tokens": 1024,
        "temperature": 0,
    }
    t0 = time.time()
    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.post(_url(entry), json=body, headers=_headers(entry))
        lat = round(time.time() - t0, 1)
        if r.status_code == 429:
            return False, lat, "http_429"
        if r.status_code != 200:
            return False, lat, "http_%d" % r.status_code
        try:
            data = r.json()
        except Exception:
            return False, lat, "empty"
        content, _finish = _extract_message(data)
        if content and str(content).strip():
            return True, lat, ""
        return False, lat, "empty"
    except httpx.TimeoutException:
        return False, round(time.time() - t0, 1), "timeout"
    except Exception as ex:
        return False, round(time.time() - t0, 1), "net:%s" % type(ex).__name__


def call_entry(entry, payload, timeout=None):
    body = {
        "model": entry["model"],
        "messages": payload,
        "max_tokens": 32768,
        "temperature": 0.2,
    }
    # Optional: reasoning_effort (thinking) for models that support it
    # (e.g. Gemini via generativelanguage.googleapis.com). Only when set explicitly in the
    # chain entry, so providers that do not understand the field are not broken.
    if entry.get("reasoning_effort"):
        body["reasoning_effort"] = entry["reasoning_effort"]
    timeout = timeout or entry["timeout"]
    t0 = time.time()
    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.post(_url(entry), json=body, headers=_headers(entry))
    except httpx.TimeoutException:
        raise ChainError("timeout")
    except Exception as ex:
        raise ChainError("net:%s" % type(ex).__name__)
    lat = round(time.time() - t0, 1)
    if r.status_code == 429:
        raise ChainError("http_429")
    if r.status_code != 200:
        raise ChainError("http_%d" % r.status_code)
    try:
        data = r.json()
    except Exception:
        raise ChainError("empty finish=None")
    # model reasoning is NOT a summary: empty content = error, as in the old worker
    content, finish = _extract_message(data, allow_reasoning=False)
    if not content or not str(content).strip():
        raise ChainError("empty finish=%s" % finish)
    if not entry.get("paid") and not sections_ok(content):
        raise ChainError("bad_format")
    return content, lat


def summarize_via_chain(payload, cfg, now=None):
    chain = build_chain(cfg)
    tried = []
    t_all = time.time()
    now_arg = now
    for entry in chain:
        name = entry["name"]
        h = load_health()
        now = now_arg if now_arg is not None else time.time()  # fresh time on every write
        if not entry["paid"] and not is_available(name, h, now):
            tried.append({"name": name, "status": "skipped_cooldown"})
            continue
        warmed = False
        if not entry["paid"] and not is_fresh(name, h, now):
            ok, lat, reason = warmup_entry(entry)
            if not ok:
                mark_fail(name, "warmup:" + reason)
                tried.append({"name": name, "status": "warmup_fail", "reason": reason})
                continue
            mark_warm(name, lat)
            warmed = True
        try:
            content, lat = call_entry(entry, payload)
            mark_ok(name, lat)
            tried.append({"name": name, "status": "ok", "lat": lat, "warmed": warmed})
            return content, {
                "model_used": name,
                "tried": tried,
                "total_s": round(time.time() - t_all, 1),
            }
        except ChainError as ce:
            if entry["paid"]:
                _mark_fail_paid(name, ce.reason)
            else:
                mark_fail(name, ce.reason)
            tried.append({"name": name, "status": "fail", "reason": ce.reason})
            continue
    raise RuntimeError(
        "chain exhausted: "
        + ", ".join("%s=%s" % (t["name"], t.get("reason") or t["status"]) for t in tried)
    )


def _mark_fail_paid(name, reason) -> None:
    """Paid entry: record fails/reason but never set a cooldown (last resort)."""
    now = time.time()
    reason = (reason or "")[:120]
    with _health_lock():
        h = load_health()
        entries = h.setdefault("entries", {})
        prev = entries.get(name) or {}
        entries[name] = {
            "ok": False,
            "last_ok": prev.get("last_ok"),
            "last_fail": now,
            "fails": int(prev.get("fails") or 0) + 1,
            "skip_until": float(prev.get("skip_until") or 0.0),
            "lat": prev.get("lat", 0.0),
            "reason": reason,
        }
        save_health(h)


# ------------------------------------------------------------ private config hand-off
# Profile secrets are resolved by the TRUSTED process (the engine, inside the profile secret scope) and passed
# to the child process (worker, warmer) ready-made: in multiplex mode profile keys do not
# reach os.environ, and the child could not find them itself. File is 0600, the consumer deletes it at once.

def stash_cfg(cfg, directory, tag):
    """Write the config (with keys) to a private file <tag>.cfg (0600). Path, or None on failure."""
    try:
        os.makedirs(str(directory), exist_ok=True)
        path = os.path.join(str(directory), "%s.cfg" % tag)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(cfg, f, default=str)
        return path
    except Exception:
        return None


def take_cfg(path):
    """Read and DELETE the private config file. None on any error."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None
    finally:
        try:
            os.remove(path)
        except Exception:
            pass


def sweep_stale_cfg(directory, max_age=3600.0):
    """Remove forgotten *.cfg older than max_age: a safety net against secrets left on disk."""
    try:
        now = time.time()
        for name in os.listdir(str(directory)):
            if name.endswith(".cfg"):
                p = os.path.join(str(directory), name)
                try:
                    if now - os.path.getmtime(p) > max_age:
                        os.remove(p)
                except Exception:
                    pass
    except Exception:
        pass
