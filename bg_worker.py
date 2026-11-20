"""bg_worker.py — background process: summarization via auxiliary.compression.

Contract mirrors ContextCompressor._generate_summary / _call_summary_llm
(agent/context_compressor) and _assemble_compressed:
- system prompt: _summarizer_preamble;
- user message: fresh form (no previous summary) + section template;
- parsing: content -> strip_think_blocks -> redact -> SUMMARY_PREFIX + text;
- compacted list: head(protected system) + summary + tail(live tail, last turns verbatim);
- summary message metadata: _compressed_summary=True (+ has_user_turn).

Output contract (--out): {"compacted_messages": [...], "watermark": N, "ok": true}.
exit=0 only on full success.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # plugins root
from asinccontext.bg_cfg import load_bg_config  # noqa: E402
from asinccontext.chain import summarize_via_chain, take_cfg  # noqa: E402
from asinccontext.summarize import (  # noqa: E402
    _assemble_compacted,
    _SUMMARY_END_MARKER,
    assemble_with_tail,
    split_tail,
    build_summary_messages,
    parse_summary_response,
    snapshot_anchor,
)

# Metrics: absolute import (the worker is a separate subprocess), relative as a fallback.
# A metrics failure/absence NEVER affects the worker (_m swallows everything).
try:
    from asinccontext.metrics import record as _mrecord  # noqa: E402
except Exception:  # noqa: BLE001
    try:
        from .metrics import record as _mrecord  # noqa: E402
    except Exception:  # noqa: BLE001
        _mrecord = None
try:
    from asinccontext.metrics import retention as _mretention  # noqa: E402
except Exception:  # noqa: BLE001
    try:
        from .metrics import retention as _mretention  # noqa: E402
    except Exception:  # noqa: BLE001
        _mretention = None
try:
    from asinccontext.metrics import depth_of as _mdepth  # noqa: E402
except Exception:  # noqa: BLE001
    try:
        from .metrics import depth_of as _mdepth  # noqa: E402
    except Exception:  # noqa: BLE001
        _mdepth = None


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


def _role_alternation(messages) -> str:
    """Kept for backward compatibility; canonical — summarize._role_alternation."""
    from asinccontext.summarize import _role_alternation as _ra

    return _ra(messages)


def _assemble_compacted(messages, summary: str, has_user_turn: bool):
    """Kept for backward compatibility; canonical — summarize._assemble_compacted."""
    from asinccontext.summarize import _assemble_compacted as _ac

    return _ac(messages, summary, has_user_turn)


# Optional notifications about background compression: OFF by default. Turn on
# with env ASINCCONTEXT_TG_NOTIFY=1 (or true/yes/on) in the gateway environment.
# (The notifier itself is not part of the public build; the gate is kept so
# downstream forks can plug one in.) The gate lives in the worker, and
# the worker is a fresh subprocess per job, so a change applies without
# a gateway restart.
def _tg_notify_enabled() -> bool:
    return (os.environ.get("ASINCCONTEXT_TG_NOTIFY") or "").strip().lower() in ("1", "true", "yes", "on")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--msgs", required=True)
    ap.add_argument("--watermark", type=int)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cfg", default=None)  # private file with the ready config (secrets resolved by the engine)
    # Extra arguments used ONLY by notifications; without them the worker runs
    # normally (older calls stay fully compatible).
    ap.add_argument("--home", default=None)  # profile HERMES_HOME (for notifications)
    ap.add_argument("--session-id", default=None)  # session_id (for notifications)
    a = ap.parse_args()

    _t0 = time.monotonic()  # wall_s metric of the whole worker run
    _n_msgs = None  # number of incoming messages; may stay unknown on an early failure
    _prof_kw = {}  # {'profile': ...} from cfg['_profile'], if the engine passed it
    try:
        cfg = take_cfg(a.cfg) if a.cfg else None  # reads and immediately deletes the file
        if cfg is None:
            cfg = load_bg_config()
        if isinstance(cfg, dict) and cfg.get("_profile"):
            _prof_kw["profile"] = cfg["_profile"]
        with open(a.msgs) as _f:
            msgs = json.load(_f)
        # The engine snapshot is a full transcript copy: drop it as soon as it is read.
        # Only files inside our own snapshots/ dir are removed (callers may pass their own files).
        _snap_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snapshots")
        if os.path.dirname(os.path.realpath(a.msgs)) == os.path.realpath(_snap_dir):
            try:
                os.remove(a.msgs)
            except OSError:
                pass
        if not isinstance(msgs, list):
            raise ValueError("--msgs must be a JSON list")
        _n_msgs = len(msgs)
        if not cfg.get("free_chain") and not cfg.get("model"):
            sys.stderr.write("bg_worker not_configured: no free_chain and no model\n")
            return 1
        to_sum, tail = split_tail(msgs)
        payload = build_summary_messages(to_sum)
        content, meta = summarize_via_chain(payload, cfg)
        summary = parse_summary_response(content)
        has_user_turn = any((m or {}).get("role") == "user" for m in msgs)
        compacted = assemble_with_tail(msgs, summary, tail, has_user_turn)
        # Atomic write (tmp + os.replace): the engine poll() treats the job as done only
        # when it sees VALID JSON with ok=True. A partial write (kill during dump)
        # must never pass for a ready result (a manual deletion of out.json once
        # wiped a live result before it was consumed).
        tmp_out = a.out + ".tmp"
        with open(tmp_out, "w") as f:
            json.dump(
                {"compacted_messages": compacted, "watermark": a.watermark, "ok": True, "anchor": snapshot_anchor(msgs), "meta": meta},
                f, ensure_ascii=False,
            )
        os.replace(tmp_out, a.out)
        # Guard (manual deletion of out.json before the consuming turn):
        # the result stays read-only until the engine marks it consumed (collect
        # removes 0o444). Accidental truncation/overwrite of live work is blocked;
        # if the file is deleted anyway, background.poll on a dead pid restarts
        # the job on the next should_compress instead of a silent "resumed".
        try:
            os.chmod(a.out, 0o444)
        except OSError:
            pass  # chmod must not break a successful write
        # Metric of a successful background compression (never breaks the result).
        try:
            src = payload[1].get("content", "") if isinstance(payload, list) and len(payload) > 1 else ""
            tried = []
            for t in (meta or {}).get("tried") or []:
                if isinstance(t, dict):
                    tried.append({k: t.get(k) for k in ("name", "status", "reason", "lat") if k in t})
            _at, _ak = (_mretention(src, summary) if _mretention else (0, 0))
            _kw = {}
            if isinstance(cfg, dict) and cfg.get("_profile") is not None:
                _kw["profile"] = cfg.get("_profile")
            _m("bg_job",
               ok=True,
               model_used=(meta or {}).get("model_used"),
               tried=(tried or None),
               total_s=(meta or {}).get("total_s"),
               msgs_in=_n_msgs,
               input_chars=len(src),
               summary_chars=len(summary),
               anchors_total=_at,
               anchors_kept=_ak,
               depth=(_mdepth(msgs) if _mdepth else 0),
               wall_s=round(time.monotonic() - _t0, 3),
               **_kw)
        except Exception:  # noqa: BLE001
            pass
        return 0
    except RuntimeError as e:
        sys.stderr.write("bg_worker chain_failed: %s\n" % e)
        _m("bg_job", ok=False, reason=str(e)[:160], tried=None, msgs_in=_n_msgs,
           wall_s=round(time.monotonic() - _t0, 3), **_prof_kw)
        return 1
    except Exception as e:
        sys.stderr.write(f"bg_worker error: {e}\n")
        _m("bg_job", ok=False, reason="exc:%s" % type(e).__name__, tried=None, msgs_in=_n_msgs,
           wall_s=round(time.monotonic() - _t0, 3), **_prof_kw)
        return 1


if __name__ == "__main__":
    sys.exit(main())