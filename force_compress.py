"""force_compress.py — synchronous /compress branch.

A blocking call to auxiliary.compression in the CURRENT turn (no background),
the compacted list is built the same way as in bg_worker. Does not write the DB — returns
a list of messages, the host commits it (conversation_compression).
"""

import json

# Relative imports: the plugin is loaded as hermes_plugins.asinccontext (plugins_loader);
# an absolute `import asinccontext.*` does not resolve inside the hermes process.
from .bg_cfg import load_bg_config  # noqa: E402
from .chain import summarize_via_chain  # noqa: E402
from .summarize import assemble_with_tail, build_summary_messages, parse_summary_response, split_tail  # noqa: E402


def run_sync(messages, memory_context=""):
    """Synchronous summarization. Returns the compacted list, or the original messages on failure."""
    cfg = load_bg_config()
    if not cfg.get("free_chain") and not (cfg.get("model") and cfg.get("api_key")):
        raise RuntimeError("auxiliary.compression is not configured (free_chain or model/api_key) for /compress")
    msgs = [m for m in (messages or []) if isinstance(m, dict)]
    to_sum, tail = split_tail(msgs)
    payload = build_summary_messages(to_sum, memory_context=memory_context)
    content, _meta = summarize_via_chain(payload, cfg)
    try:  # metric: which model answered on the blocking path (a metrics failure never affects compression)
        from .metrics import record as _rec
        _rec("sync_chain", model_used=(_meta or {}).get("model_used"), tried=[
            {k: t.get(k) for k in ("name", "status", "reason", "lat") if k in t}
            for t in ((_meta or {}).get("tried") or []) if isinstance(t, dict)] or None,
            total_s=(_meta or {}).get("total_s"), msgs_in=len(messages))
    except Exception:
        pass
    summary = parse_summary_response(content)
    has_user_turn = any((m or {}).get("role") == "user" for m in msgs)
    # same assembly as in bg_worker: summary + live tail
    return assemble_with_tail(msgs, summary, tail, has_user_turn)