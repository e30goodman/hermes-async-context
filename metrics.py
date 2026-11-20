"""metrics.record — atomic write of a compression metric to compress.jsonl.

Never raises and never blocks longer than ~1 s. Rotation .jsonl -> .1 -> .2 -> .3
above 5 MB. Re-exports anchors/retention/depth_of.
"""

import fcntl
import json
import os
import time

from .metrics_anchors import anchors, retention, depth_of  # noqa: F401

_STR_LIMIT = 200
_MAX_BYTES = 5 * 1024 * 1024
_ROT_SHIFTS = 3


def _metrics_dir():
    d = os.environ.get("ASINC_METRICS_DIR")
    if d:
        return d
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "metrics")


def _profile():
    # In a multiplexed gateway HERMES_HOME is always the main one; the request profile comes from a contextvar in get_hermes_home().
    try:
        from hermes_constants import get_hermes_home
        home = str(get_hermes_home())
    except Exception:
        home = os.environ.get("HERMES_HOME") or ""
    if not home:
        return "main"
    return os.path.basename(os.path.normpath(home)) or "main"


def _trunc(obj, limit=_STR_LIMIT):
    if isinstance(obj, str):
        return obj[:limit]
    if isinstance(obj, dict):
        return {k: _trunc(v, limit) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_trunc(x, limit) for x in obj]
    return obj


def _rotate(path):
    """compress.jsonl -> .1 -> .2 -> .3; the oldest shift is dropped."""
    for i in range(_ROT_SHIFTS, 0, -1):
        src = path if i == 1 else "%s.%d" % (path, i - 1)
        dst = "%s.%d" % (path, i)
        if i == _ROT_SHIFTS:
            if os.path.exists(dst):
                try:
                    os.remove(dst)
                except OSError:
                    pass
        if os.path.exists(src):
            try:
                os.replace(src, dst)
            except OSError:
                pass


def record(event, **fields):
    """Append one JSON line. Never raises / blocks >1 s."""
    try:
        d = _metrics_dir()
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, "compress.jsonl")
        ent = {"ts": time.time(), "event": event, "profile": _profile()}
        ent.update(_trunc(fields))
        line = json.dumps(ent, default=str, ensure_ascii=False) + "\n"
        try:
            size = os.path.getsize(path)
        except OSError:
            size = 0
        if size > _MAX_BYTES:
            _rotate(path)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            # Do not block the turn: wait for the lock at most ~0.5 s, otherwise silently skip the write.
            locked = False
            for _ in range(10):
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    locked = True
                    break
                except OSError:
                    time.sleep(0.05)
            if locked:
                os.write(fd, line.encode("utf-8"))
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)
    except Exception:
        pass