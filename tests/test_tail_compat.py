#!/usr/bin/env python3
"""[claude] 2026-11-20 tests: live tail, host-API compat, deepcopy, housekeeping."""
import copy
import os
import sys
import tempfile
import time

PLUGINS_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PLUGINS_ROOT)
sys.path.insert(0, os.environ.get("HERMES_AGENT_ROOT", "."))

from asinccontext import summarize as S  # noqa: E402
from asinccontext import background as B  # noqa: E402

FAILS = []


def check(name, cond, extra=""):
    print(("[PASS] " if cond else "[FAIL] ") + name + (f" :: {extra}" if extra and not cond else ""))
    if not cond:
        FAILS.append(name)


def conv():
    return [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "t1", "type": "function",
                                                             "function": {"name": "x", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "t1", "content": "r1"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "u3"},
        {"role": "assistant", "content": "a3"},
    ]


m = conv()
to_sum, tail = S.split_tail(m, turns=2, max_chars=100000)
check("tail starts at the 2nd-last user turn", tail and tail[0]["content"] == "u2", tail[:1])
check("tool call + result kept together in tail", any(x.get("role") == "tool" for x in tail))
check("summarized part = system + older turns", [x["content"] for x in to_sum] == ["sys", "u1", "a1"])
out = S.assemble_with_tail(m, "SUM", tail, True)
check("system head kept", out[0]["role"] == "system")
check("summary folded into first tail user row", out[1]["role"] == "user" and out[1]["content"].startswith("SUM")
      and out[1]["content"].endswith("u2") and out[1].get("_compressed_summary"))
check("rest of tail verbatim", out[2:] == tail[1:])
check("input not mutated", m == conv())

_, tail_small = S.split_tail(m, turns=2, max_chars=150)
check("budget too small for 2 turns -> 1 turn", tail_small and tail_small[0]["content"] == "u3", tail_small)
_, tail_none = S.split_tail(m, turns=2, max_chars=5)
check("budget too small for any turn -> legacy", tail_none == [])
legacy = S.assemble_with_tail(m, "SUM", [], True)
check("legacy path still works", legacy[0]["role"] == "system" and any("SUM" in (x.get("content") or "") for x in legacy))
_, t1 = S.split_tail([{"role": "system", "content": "s"}, {"role": "user", "content": "only"}])
check("single user turn -> nothing to split", t1 == [])
lst = [{"role": "system", "content": "s"}, {"role": "user", "content": "old"},
       {"role": "assistant", "content": "x"}, {"role": "user", "content": [{"type": "text", "text": "new"}]}]
_, tl = S.split_tail(lst, turns=1)
o = S.assemble_with_tail(lst, "SUM", tl, True)
check("list content gets a text part prepended", isinstance(o[1]["content"], list) and o[1]["content"][0]["text"].startswith("SUM"))

# housekeeping
with tempfile.TemporaryDirectory() as d:
    jd, sd = os.path.join(d, "jobs"), os.path.join(d, "snapshots")
    os.makedirs(jd), os.makedirs(sd)
    old_s, new_s, old_j = os.path.join(sd, "a.json"), os.path.join(sd, "b.json"), os.path.join(jd, "c.json")
    for p in (old_s, new_s, old_j):
        open(p, "w").write("{}")
    t = time.time()
    os.utime(old_s, (t - B.SNAPSHOT_TTL_S - 10,) * 2)
    os.utime(old_j, (t - B.JOB_FILE_TTL_S - 10,) * 2)
    B.sweep_housekeeping(jd, sd, now=t)
    check("old snapshot removed", not os.path.exists(old_s))
    check("fresh snapshot kept", os.path.exists(new_s))
    check("stale job file removed", not os.path.exists(old_j))

env = B.worker_env()
check("worker env carries plugins root first", env["PYTHONPATH"].split(os.pathsep)[0] == str(B.PLUGINS_ROOT))
check("worker cmd uses this interpreter by default",
      B.worker_cmd("x")[0] == (os.environ.get("ASINCCONTEXT_PYTHON") or sys.executable))

# engine: host API + deepcopy
try:
    from asinccontext.engine import AsyncContextEngine  # noqa: E402
    e = AsyncContextEngine()
    check("coerce cap: positive int", e._coerce_threshold_tokens_cap("5000") == 5000)
    check("coerce cap: 0/None/junk -> None", e._coerce_threshold_tokens_cap(0) is None
          and e._coerce_threshold_tokens_cap(None) is None and e._coerce_threshold_tokens_cap("x") is None)
    e.threshold_tokens = 100000
    e.threshold_tokens_cap = 40000
    os.environ.pop("X", None)
    got = e._resolved_threshold_tokens()
    check("cap lowers the host threshold", got in (40000,) or e._last_threshold_source == "override", got)
    e._threshold_tokens = None  # what the host's live-config apply does
    e.threshold_tokens = 0
    check("nulled cache falls back to class default", e._resolved_threshold_tokens() > 0)
    e._session_db = object()
    e._sync_fallback_disabled = True
    c = copy.deepcopy(e)
    check("deepcopy drops db handle", c._session_db is None)
    check("deepcopy keeps every other field", c._sync_fallback_disabled is True and c.threshold_tokens_cap == 40000)
except ImportError as ex:
    print("[SKIP] engine tests (host package not importable):", ex)

print("RESULT:", "ALL PASSED" if not FAILS else "FAILED " + ", ".join(FAILS))
sys.exit(1 if FAILS else 0)
