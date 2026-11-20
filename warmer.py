"""warmer.py — automatic warm-up of the free models of the chain.

Run: python -m asinccontext.warmer [--force] [--status]

Keeps the chain "hot" BEFORE compression is needed: the engine starts this process
at the end of a turn, and it warms up only stale/available entries.
Within one host — strictly sequential (web proxies accept one request
at a time), different hosts — in parallel. Never raises: exit code 0.
"""

import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # plugins root

from asinccontext.bg_cfg import load_bg_config  # noqa: E402
from asinccontext.chain import (  # noqa: E402
    build_chain,
    health_path,
    is_available,
    is_fresh,
    load_health,
    mark_fail,
    mark_warm,
    save_health,
    take_cfg,
    warmup_entry,
    WARM_TTL,
)
from asinccontext.logsetup import trace  # noqa: E402

# Metrics: absolute import (the warmer is a separate subprocess), relative as a fallback.
try:
    from asinccontext.metrics import record as _mrecord  # noqa: E402
except Exception:  # noqa: BLE001
    try:
        from .metrics import record as _mrecord  # noqa: E402
    except Exception:  # noqa: BLE001
        _mrecord = None


def _m(event, **f):
    """Record a metric; a metrics failure/absence never affects the work."""
    try:
        from asinccontext.metrics import record as _r
    except Exception:  # noqa: BLE001
        try:
            from .metrics import record as _r
        except Exception:  # noqa: BLE001
            return None
    try:
        return _r(event, **f)
    except Exception:  # noqa: BLE001
        return None


LOCK_STALE_S = 120.0

# health.json is read-modify-write in chain.save_health; with parallel hosts
# two mark_* calls could overwrite each other. The write is short, so only it
# is serialized (HTTP warm-up per host stays parallel).
_HEALTH_LOCK = threading.Lock()


def _acquire_lock(path):
    """Take the lock file. A fresh foreign lock -> False; a stale one -> take over."""
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        return True
    except FileExistsError:
        pass
    except Exception:
        return False
    try:
        age = time.time() - os.path.getmtime(path)
    except Exception:
        return False
    if age < LOCK_STALE_S:
        return False
    try:
        os.unlink(path)
    except Exception:
        return False
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        return True
    except Exception:
        return False


def _release_lock(path):
    try:
        os.unlink(path)
    except Exception:
        pass


def _warm_host(entries):
    """Warm up the entries of one host sequentially."""
    for e in entries:
        name = e.get("name")
        try:
            ok, lat, reason = warmup_entry(e)
        except Exception as ex:
            ok, lat, reason = False, 0.0, "exc:%s" % type(ex).__name__
        _m("warm", name=name, ok=bool(ok), lat=lat, reason=(reason or ""))
        try:
            with _HEALTH_LOCK:
                if ok:
                    mark_warm(name, lat)
                    trace("WARM name=%s ok=1 lat=%.1f" % (name, lat))
                else:
                    mark_fail(name, "warmup:" + (reason or "unknown"))
                    trace("WARM name=%s ok=0 reason=%s" % (name, reason))
        except Exception:
            pass


def _select(entries, force):
    """Entries to warm up: --force -> all; otherwise only stale and available ones."""
    if force:
        return list(entries)
    now = time.time()
    h = load_health()
    out = []
    for e in entries:
        name = e.get("name")
        try:
            if is_fresh(name, h, now, ttl=WARM_TTL / 2):  # refresh at half the TTL so the chain does not "cool down" between cycles
                continue
            if not is_available(name, h, now):
                continue
        except Exception:
            pass
        out.append(e)
    return out


def _status():
    h = load_health()
    entries = h.get("entries") or {}
    now = time.time()
    print("%-40s %-5s %-10s %-6s %-10s %s" % ("name", "ok", "last_ok_c", "fails", "cooldown_c", "reason"))
    for name in sorted(entries):
        e = entries.get(name) or {}
        last_ok = e.get("last_ok")
        try:
            age = "-" if last_ok is None else "%.0f" % max(0.0, now - float(last_ok))
        except Exception:
            age = "-"
        try:
            cd = "%.0f" % max(0.0, float(e.get("skip_until") or 0.0) - now)
        except Exception:
            cd = "-"
        print("%-40s %-5s %-10s %-6s %-10s %s" % (
            name, e.get("ok"), age, int(e.get("fails") or 0), cd,
            (e.get("reason") or "")[:60],
        ))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--status" in argv:
        try:
            _status()
        except Exception:
            pass
        return 0
    force = "--force" in argv
    cfg_path = argv[argv.index("--cfg") + 1] if "--cfg" in argv and argv.index("--cfg") + 1 < len(argv) else None
    try:
        cfg = take_cfg(cfg_path) if cfg_path else None  # secrets resolved by the engine
        if cfg is None:
            cfg = load_bg_config()
        entries = [e for e in build_chain(cfg) if not e.get("paid")]
        if not entries:
            return 0
        lock = health_path() + ".lock"
        if not _acquire_lock(lock):
            return 0
        try:
            todo = _select(entries, force)
            groups = {}
            for e in todo:
                host = urlparse(e.get("base_url") or "").netloc or ""
                groups.setdefault(host, []).append(e)
            if groups:
                with ThreadPoolExecutor(max_workers=len(groups)) as pool:
                    futures = [pool.submit(_warm_host, g) for g in groups.values()]
                    for f in futures:
                        try:
                            f.result()
                        except Exception:
                            pass
            # "checked" mark: touch the health file mtime so the engine does not spawn the warmer for nothing
            try:
                save_health(load_health())
            except Exception:
                pass
        finally:
            _release_lock(lock)
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
