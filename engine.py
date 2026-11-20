"""Async Context Engine (Tasks 1, 5-7).

Background, platform-non-blocking context summarization.
The engine REPLACES the built-in ContextCompressor; job state is a file registry
(plugins/asinccontext/jobs/<session_id>.json), not in-memory: _retire_agent
rebuilds the engine inside a live session, and the fresh instance picks up the result
from the file.

Order in the host (agent_init):
  agent constructor → engine __deepcopy__ → bind_session_state(db, sid) →
  on_session_start(sid, ...) → per-turn should_compress()/compress() ->
  archive_and_compact (conversation_compression) — the host writes the DB itself,
  the plugin writes ONLY its own file registry of jobs.
"""

import copy
import sys

from agent.context_engine import ContextEngine  # noqa: E402

# RELATIVE imports: the plugin is loaded as hermes_plugins.asinccontext (plugins_loader),
# an absolute `import asinccontext.*` does not resolve in that role. bg_worker.py stays
# on absolute imports — it runs as a separate subprocess (python -m asinccontext.bg_worker).
from . import background as _bg  # noqa: E402  (dynamic JOBS_DIR for tests/overrides)
from .bg_cfg import load_bg_config  # noqa: E402
from .background import BackgroundJob  # noqa: E402
from .logsetup import trace  # noqa: E402

import logging
import os
import subprocess
import time

# Metrics: relative import (the engine is loaded as hermes_plugins.asinccontext),
# absolute as a fallback. A metrics failure/absence never affects the engine.
try:
    from .metrics import record as _mrecord  # noqa: E402
except Exception:  # noqa: BLE001
    try:
        from asinccontext.metrics import record as _mrecord  # noqa: E402
    except Exception:  # noqa: BLE001
        _mrecord = None


def _m(event, **f):
    """Record a metric; a metrics failure/absence never affects the work."""
    try:
        from .metrics import record as _r
    except Exception:  # noqa: BLE001
        try:
            from asinccontext.metrics import record as _r
        except Exception:  # noqa: BLE001
            return None
    try:
        return _r(event, **f)
    except Exception:  # noqa: BLE001
        return None


def _profile_cfg(cfg):
    """Add the profile id to a copy of the config for the background worker."""
    try:
        from hermes_constants import get_hermes_home  # noqa: PLC0415
        prof = os.path.basename(str(get_hermes_home()))
    except Exception:  # noqa: BLE001
        prof = None
    if not prof or not isinstance(cfg, dict):
        return cfg
    return dict(cfg, _profile=prof)

_log = logging.getLogger("asinccontext")


class AsyncContextEngine(ContextEngine):
    """Replacement for the built-in compressor: background summarization outside the host turn.

    After every return from compress() the host unconditionally runs
    archive_and_compact → stamp_db_persisted_markers → update_system_prompt
    (conversation_compression). So the plugin never writes the DB:
    it only decides WHAT the host gets and WHEN.
    """

    _threshold_tokens: int = 200_000  # default ONLY; the normal path is _resolved_threshold_tokens()

    _SYNC_FALLBACK_AFTER: int = 3  # after this many turns in a row over the threshold — sync fallback
    # Class defaults: __deepcopy__ below builds the object via cls.__new__, bypassing __init__,
    # so every field read before its first assignment must have a default here.
    _pending_since_over: int = 0
    _fallback_armed: bool = False
    _snapshot_fail: int = 0
    _last_bg_watermark = None
    _last_warm_spawn: float = 0.0
    _warm_cfg_ts: float = 0.0
    _warm_has_chain: bool = False
    _sync_fallback_disabled: bool = False
    threshold_tokens_cap = None  # optional absolute cap, set by the host's live config apply

    @staticmethod
    def _coerce_threshold_tokens_cap(value):
        """Same contract as ContextCompressor._coerce_threshold_tokens_cap: positive int or None."""
        try:
            iv = int(value) if value is not None else 0
        except (TypeError, ValueError):
            return None
        return iv if iv > 0 else None

    def __init__(self) -> None:
        super().__init__()
        self._session_id: str = ""
        self._session_db = None
        self._snapshot_dir = None
        self._last_threshold_source: str = "unset"
        self._last_bg_watermark = None  # watermark of the last background result
        self._last_warm_spawn: float = 0.0  # last warmer launch
        self._warm_cfg_ts: float = 0.0  # when free_chain was last read from config
        self._warm_has_chain: bool = False  # whether free_chain exists (config cache)
        self._pending_since_over: int = 0  # turns in a row over the threshold without a ready result
        self._fallback_armed: bool = False  # signal from should_compress: time for the sync fallback
        self._snapshot_fail: int = 0  # failed snapshots in a row (no job) — fallback trigger
        self._sync_fallback_disabled: bool = False  # true = forbid the blocking pre-API fallback

    def _resolved_threshold_tokens(self) -> int:
        """Trigger threshold — single source of truth.

        Priority:
        1) An explicit ``compression.threshold_tokens_override`` from the profile config (a test
           profile can set e.g. 200000 — a hard threshold to trigger quickly on live
           turns). This is an override, not the only source.
        2) Otherwise the host-computed ``self.threshold_tokens`` written by update_model()
           (context_engine: ctx * threshold_percent) — adapts to the real model window
           (1.3M → ~983k, 128k → ~96k). Another model → recomputed automatically.

        Earlier the engine read a hard ``_threshold_tokens=200_000`` and ignored the host field
        (a ~5x mismatch). Now the host value is the norm,
        and the constant remains only as the attribute default / override fallback.
        """
        try:
            cfg = load_bg_config()
            ov = cfg.get("threshold_tokens_override")
            if isinstance(ov, int) and ov > 0:
                self._last_threshold_source = "override"
                return ov
        except Exception as e:  # noqa: BLE001 — never fail: the threshold is needed even without config
            _log.warning("threshold override unavailable (%s) — fallback host", e)
        host = getattr(self, "threshold_tokens", 0) or 0
        cap = getattr(self, "threshold_tokens_cap", None)
        if host > 0:
            self._last_threshold_source = "host"
            return min(host, cap) if isinstance(cap, int) and cap > 0 else host
        # The host has not called update_model yet (the very first should_compress before model setup).
        # Return the default constant — the only case where it is used.
        self._last_threshold_source = "default"
        return self._threshold_tokens or type(self)._threshold_tokens  # host may null the cached value

    @property  # ABC: name is @property @abstractmethod; a plain method silently breaks engine selection
    def name(self) -> str:
        return "asinc-context"

    # ── Host lifecycle hooks ────────────────────────────────────

    def bind_session_state(self, session_db=None, session_id: str = "") -> None:
        """Called by the host when the method exists (agent_init; read-only for us).

        Gives read-only access to the transcript for the snapshot.
        """
        self._session_db = session_db
        if session_id:
            self._session_id = session_id or ""

    def on_session_start(self, session_id=None, **kwargs) -> None:
        """Every freshly built engine (including after _retire_agent) — pick-up via the registry."""
        if session_id is not None:
            self._session_id = session_id
        others = {k: v for k, v in kwargs.items() if k != "session_id"}
        super().on_session_start(self._session_id, **others)

    def on_session_reset(self) -> None:
        """Reset session_id-specific state."""
        self._clear_job()
        super().on_session_reset()

    # ── Predicate with a side effect ─────────────

    def on_turn_complete(self, messages, usage=None, **kwargs) -> None:
        """Called by the host ONCE per turn: increments turns_waited in the registry (turn boundary).

        should_compress() is called several times per turn (once per API call),
        so this is where "one more turn of waiting" is recorded. Never
        raises: the turn counter must not break the turn.
        """
        try:
            job = self._resolve_job()
            if job is not None and job.poll() in ("running", "resumed"):
                n = job.bump_turns()
                trace("JOB=turn-waited session=%s job=%s turns=%s" % (self._session_id, job.job_id, n))
            self._maybe_warm()
        except Exception:  # noqa: BLE001
            pass

    def _maybe_warm(self) -> None:
        """Start a background warm-up of the chain if needed and not started within 120 s.

        Cheap and silent: throttled by _last_warm_spawn, free_chain cached for 300 s,
        and the warmer process starts only when the health file is missing or
        stale (>200 s). Never raises — the turn must not fail because of warm-up.
        """
        try:
            t = time.time()
            if t - self._last_warm_spawn < 120:
                return
            if t - self._warm_cfg_ts >= 300:
                from .bg_cfg import load_bg_config  # noqa: PLC0415
                self._warm_has_chain = bool(load_bg_config().get("free_chain"))
                self._warm_cfg_ts = t
            if not self._warm_has_chain:
                return
            from .chain import health_path  # noqa: PLC0415
            hp = health_path()
            try:
                stale = (t - os.path.getmtime(hp)) > 200
            except OSError:
                stale = True
            if not stale:
                return
            cfg_path = None
            try:
                from .chain import stash_cfg, sweep_stale_cfg  # noqa: PLC0415
                d = os.path.dirname(hp) or "."
                sweep_stale_cfg(d)
                cfg_path = stash_cfg(load_bg_config(), d, "warmer.%d" % int(t))
            except Exception:
                cfg_path = None
            try:
                subprocess.Popen(
                    [*_bg.worker_cmd("asinccontext.warmer"), *(["--cfg", cfg_path] if cfg_path else [])],
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    start_new_session=True,
                    env=_bg.worker_env(),
                )
            except Exception:
                if cfg_path:
                    try:
                        os.remove(cfg_path)
                    except OSError:
                        pass
                raise
            self._last_warm_spawn = t
            trace("WARM=spawned session=%s" % (self._session_id,))
        except Exception:  # noqa: BLE001 — warm-up must not break the turn
            pass

    def should_compress(self, prompt_tokens: int = None) -> bool:
        # side-effect point: the predicate starts the background job (this comment is required).
        # Registry access by session_id (file jobs/<sid>.json), NOT via a live _bg:
        # a fresh engine after _retire_agent/restart sees the job from the file, not from memory.
        over = (prompt_tokens or 0) >= self._resolved_threshold_tokens()
        job = self._resolve_job()
        if job is not None:
            st = job.poll()
            if st == "done":
                self._pending_since_over = 0  # a ready result exists — nothing to wait for
                trace("JOB=ready session=%s job=%s" % (self._session_id, job.job_id))
                return True  # the host calls compress(), we hand over the ready result
            if st in ("running", "resumed"):
                if over:
                    n = job.turns_waited()  # count TURNS from the registry, not predicate calls
                    self._pending_since_over = n  # for tracing only
                    trace(
                        "JOB=pending session=%s job=%s pending=%s"
                        % (self._session_id, job.job_id, n)
                    )
                    if n > self._SYNC_FALLBACK_AFTER and self._sync_fallback_allowed():
                        _log.warning(
                            "sync fallback armed after %s pending turns (bg job %s still %s)",
                            n, job.job_id, st,
                        )
                        self._fallback_armed = True
                        return True  # the host calls compress() -> sync fallback branch
                else:
                    self._pending_since_over = 0
                return False  # do not duplicate, the job is alive (file exists)
            # dead: no result and the job failed — clear it and start again
            job.clear()
            self._pending_since_over = 0
            _log.info("dead bg job cleaned (no result): %s", job.job_id)
        threshold = self._resolved_threshold_tokens()
        if self._regrow_blocks(prompt_tokens or 0, over):
            return False  # [claude] 2026-10-28 anti-thrash: wait for growth after a fresh compaction
        if over:
            snap_path, watermark = self._snapshot()
            if snap_path is None:
                # Snapshot failed (no db/session/empty transcript): count failures in a row,
                # after _SYNC_FALLBACK_AFTER arm the sync fallback (otherwise the background job never starts).
                self._snapshot_fail += 1
                if self._snapshot_fail >= self._SYNC_FALLBACK_AFTER and self._sync_fallback_allowed():
                    self._fallback_armed = True
                    _log.warning("sync fallback armed after %s snapshot failures", self._snapshot_fail)
                    return True  # make the host call compress() -> sync fallback branch
            else:
                job = BackgroundJob.start(
                    self._session_id, snap_path, watermark, _profile_cfg(load_bg_config()),
                    jobs_dir=_bg.JOBS_DIR,
                )
                self._snapshot_fail = 0  # snapshot ok, job started
                trace(
                    "JOB=started session=%s job=%s watermark=%s"
                    % (self._session_id, job.job_id, watermark)
                )
                _m("job_start",
                   session=self._session_id,
                   prompt_tokens=prompt_tokens,
                   threshold=threshold,
                   threshold_source=getattr(self, "_last_threshold_source", "?"),
                   watermark=watermark)
                _log.info(
                    "should_compress BGFIRED session=%s threshold=%s threshold_source=%s prompt_tokens=%s job=%s out=%s",
                    self._session_id, threshold, getattr(self, "_last_threshold_source", "?"),
                    prompt_tokens, job.job_id, job.out_path,
                )
            return False  # this turn does not compress; the result comes on a later turn
        self._pending_since_over = 0  # threshold not exceeded — counter not needed
        return False

    # ── Transcript snapshot (read-only) ──────────────────────

    def _snapshot(self):
        """Take a read-only snapshot of the active messages + watermark.

        Returns (path to a temp file, watermark) or (None, 0) if db/session
        are unavailable. SessionDB is only READ — no writes.
        """
        db = getattr(self, "_session_db", None)
        sid = self._session_id
        if db is None or not sid:
            return None, 0
        try:
            watermark = int(db.get_active_message_watermark(sid))
            rows = db.get_messages(sid, include_inactive=False, include_compacted=False)
        except Exception as e:  # noqa: BLE001 — a read-only attempt must not break the turn
            _log.warning("snapshot read failed: %s", e)
            return None, 0
        if not rows:
            return None, 0
        # sanitize: wire fields only (role/content/tool_calls/name), no internal keys
        clean = []
        for m in rows:
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            if role not in ("user", "assistant", "tool", "system"):
                continue
            cm = {"role": role}
            if isinstance(m.get("content"), (str, list)):
                cm["content"] = m["content"]
            # tool_calls belong to role=="assistant" (tool invocations),
            # while role=="tool" carries tool_call_id (the answer to a call). Copy by role,
            # otherwise the snapshot loses calls OR mixes up fields.
            if role == "assistant" and isinstance(m.get("tool_calls"), list):
                cm["tool_calls"] = m["tool_calls"]
            elif role == "tool":
                tcid = m.get("tool_call_id")
                if tcid:
                    cm["tool_call_id"] = tcid
            clean.append(cm)
        if not clean:
            return None, 0
        snap_dir = self._snapshot_dir or (_bg.JOBS_DIR.parent / "snapshots")
        snap_dir.mkdir(parents=True, exist_ok=True)
        import json
        snap_path = snap_dir / f"{sid}.{__import__('time').time_ns()}.json"
        snap_path.write_text(json.dumps(clean, ensure_ascii=False))
        return str(snap_path), watermark

    # ── [claude] 2026-10-28 anti ping-pong ──────────────────────────────────
    # After compaction the context often stays above the threshold, and the next turn immediately
    # started a new job (10 cases in 3 days). Now: the first turn after
    # applying records base = prompt_tokens, a new job starts only when it
    # grows to base * _REGROW_FACTOR. Emergency exit: >= _REGROW_HARD_PCT of the window.
    # State lives in jobs/<sid>.floor.json — survives engine rebuilds.
    _REGROW_FACTOR = 1.25
    _REGROW_HARD_PCT = 0.8

    def _floor_path(self):
        return _bg.JOBS_DIR / f"{self._session_id}.floor.json"

    def _arm_regrow_floor(self) -> None:
        try:
            import json
            self._floor_path().write_text(json.dumps({"pending": True, "ts": time.time()}))
        except Exception:  # noqa: BLE001 — a file failure must not break compression
            pass

    def _regrow_blocks(self, pt: int, over: bool) -> bool:
        """True = do not start a new job now (the context has not grown since compaction)."""
        try:
            import json
            p = self._floor_path()
            if not self._session_id or not p.exists():
                return False
            d = json.loads(p.read_text() or "{}")
            if d.get("pending"):
                if pt <= 0:
                    return over  # tokens unknown yet — wait for the first measurement
                d = {"floor": int(pt * self._REGROW_FACTOR), "base": int(pt), "ts": time.time()}
                p.write_text(json.dumps(d))
                _m("regrow_floor", session=self._session_id, base=d["base"], floor=d["floor"])
            if not over:
                return False
            ctx = int(getattr(self, "context_length", 0) or 0)
            if ctx and pt >= ctx * self._REGROW_HARD_PCT:
                p.unlink(missing_ok=True)
                return False
            if pt < int(d.get("floor") or 0):
                return True
            p.unlink(missing_ok=True)
            return False
        except Exception:  # noqa: BLE001
            return False

    # ── Flag disabling the blocking pre-API fallback ───────────
    def _sync_fallback_allowed(self) -> bool:
        """True = the blocking sync fallback is allowed (default). False only
        when auxiliary.compression.sync_fallback_disabled: true — then the engine
        NEVER compresses synchronously/blocking: it waits for the background result on later
        turns. The flag is read from config on every call (cheap, load_bg_config cache)."""
        try:
            return not bool(load_bg_config().get("sync_fallback_disabled"))
        except Exception:  # noqa: BLE001 — a config failure must not break the engine
            return True

    # ── Registry access ─────────────────────────────────────────

    def _resolve_job(self):
        """BackgroundJob from the file registry by self._session_id; None if there is no file."""
        if not self._session_id:
            return None
        return BackgroundJob.retrieve(self._session_id, jobs_dir=_bg.JOBS_DIR)

    def _clear_job(self) -> None:
        """Remove the registry file (and an unconsumed out) — restart/reset."""
        if not self._session_id:
            return
        job = self._resolve_job()
        if job is not None:
            job.clear()
        else:
            try:
                (_bg.JOBS_DIR / f"{self._session_id}.json").unlink(missing_ok=True)
            except Exception:  # noqa: BLE001
                pass

    # ── Compaction quality check ───────────────────────────────────

    @staticmethod
    def _validate_compacted(compacted, original_messages) -> bool:
        """Cheap check of the compacted result before handing it to the host.

        (a) non-empty: a list with a user/assistant message with non-empty content;
        (b) the last message is not an assistant with dangling tool_calls.
        """
        if not isinstance(compacted, list) or not compacted:
            return False
        has_content = any(
            isinstance(m, dict)
            and m.get("role") in ("user", "assistant")
            and bool(m.get("content"))
            for m in compacted
        )
        if not has_content:
            return False
        last = compacted[-1]
        if isinstance(last, dict) and last.get("role") == "assistant" and last.get("tool_calls"):
            return False
        return True

    # ── Hand the ready result to the host ────────────────────────────

    def compress(self, messages, current_tokens=None, focus_topic=None,
                 force=False, memory_context=""):
        _in = len(messages or [])
        _pt = getattr(self, "last_prompt_tokens", None)
        _sid = self._session_id
        if force:
            # /compress — synchronous branch: a blocking call in the current turn.
            # THIS IS USER-INITIATED (/compress), NOT the pre-API auto-compression — the flag does not forbid it.
            from .force_compress import run_sync  # noqa: PLC0415
            _b0 = time.monotonic()
            _res = run_sync(messages, memory_context=memory_context)
            _block = round(time.monotonic() - _b0, 3)
            trace("PATH=force session=%s in=%s out=%s" % (_sid, _in, len(_res or messages)))
            _m("compress", path="force", session=_sid, msgs_in=_in,
               msgs_out=len(_res or messages), prompt_tokens=_pt, blocked_s=_block,
               appended=0, model=None, fallback_reason=None)
            return _res
        job = self._resolve_job()  # registry retrieve by self._session_id
        st = job.poll() if job is not None else None
        if job is None or st != "done":
            # No ready background result. If the threshold has been exceeded for several turns
            # in a row (should_compress armed the flag), stop waiting and compress synchronously.
            if self._fallback_armed and self._sync_fallback_allowed():
                _reason = (
                    "turns_waited" if self._pending_since_over > self._SYNC_FALLBACK_AFTER
                    else "snapshot_fail" if self._snapshot_fail >= self._SYNC_FALLBACK_AFTER
                    else "no_job"
                )
                from .force_compress import run_sync  # noqa: PLC0415
                _pending = self._pending_since_over
                _log.warning("sync fallback compress after %s pending turns", _pending)
                # Flag cleared BEFORE run_sync; the old snapshot is dropped so its result does not land
                # on top of an already compacted history.
                self._fallback_armed = False
                self._snapshot_fail = 0
                self._pending_since_over = 0
                self._clear_job()
                _b0 = time.monotonic()
                try:
                    compacted = run_sync(messages, memory_context=memory_context)
                except Exception as e:  # noqa: BLE001 — the fallback must not break the turn
                    _log.warning("sync fallback compress failed (%s) — keeping original", e)
                    _m("compress", path="sync-fallback-failed", session=_sid, msgs_in=_in, msgs_out=_in,
                       prompt_tokens=_pt, blocked_s=round(time.monotonic() - _b0, 3), appended=0,
                       model=None, fallback_reason=_reason, error=type(e).__name__)
                    return messages
                _block = round(time.monotonic() - _b0, 3)
                if not self._validate_compacted(compacted, messages):
                    _log.warning("compacted rejected by validator, keeping original")
                    _m("compress", path="sync-fallback-rejected", session=_sid, msgs_in=_in, msgs_out=_in,
                       prompt_tokens=_pt, blocked_s=_block, appended=0, model=None, fallback_reason=_reason)
                    return messages
                trace(
                    "PATH=sync-fallback session=%s pending=%s in=%s out=%s"
                    % (_sid, _pending, _in, len(compacted))
                )
                _m("compress", path="sync-fallback", session=_sid, msgs_in=_in,
                   msgs_out=len(compacted or messages), prompt_tokens=_pt, blocked_s=_block,
                   appended=0, model=None, fallback_reason=_reason)
                return compacted
            trace("PATH=noop session=%s" % (_sid,))
            _m("compress", path="noop", session=_sid, msgs_in=_in,
               msgs_out=_in, prompt_tokens=_pt, blocked_s=0.0,
               appended=0, model=None, fallback_reason=None)
            return messages  # no background result — leave untouched
        res = job.collect()  # marks consumed so it is not picked up twice
        self._clear_job()  # remove the registry file after consumption
        self._pending_since_over = 0  # ready result applied — counter reset
        compacted = res.get("compacted_messages")
        anchor = res.get("anchor")
        _model = (res.get("meta") or {}).get("model_used")
        _appended = 0
        if isinstance(anchor, dict) and anchor.get("fps"):
            from .summarize import find_anchor_end, merge_post_snapshot  # noqa: PLC0415

            end = find_anchor_end(messages, anchor)
            if end is None:
                _log.warning(
                    "anchor not found in live messages (session=%s job=%s) — stale compact rejected, keeping original",
                    _sid,
                    job.job_id,
                )
                trace("PATH=bg-rejected-anchor session=%s job=%s" % (_sid, job.job_id))
                _m("compress", path="bg-rejected-anchor", session=_sid, msgs_in=_in,
                   msgs_out=_in, prompt_tokens=_pt, blocked_s=0.0,
                   appended=0, model=_model, fallback_reason=None)
                return messages
            before = len(compacted or [])
            compacted = merge_post_snapshot(compacted or [], messages, end)
            _appended = len(compacted or []) - before
            _log.info("post-snapshot messages appended: %s", _appended)
        else:
            _log.info("compact has no anchor (legacy result) — applying as is")
        if not self._validate_compacted(compacted, messages):
            _log.warning("compacted rejected by validator, keeping original")
            trace("PATH=bg-rejected-validator session=%s job=%s" % (_sid, job.job_id))
            _m("compress", path="bg-rejected-validator", session=_sid, msgs_in=_in,
               msgs_out=_in, prompt_tokens=_pt, blocked_s=0.0,
               appended=0, model=_model, fallback_reason=None)
            return messages
        self._last_bg_watermark = res.get("watermark")  # keep for the host/diagnostics
        trace(
            "PATH=bg-applied session=%s job=%s anchor=%s appended=%s in=%s out=%s watermark=%s model=%s"
            % (
                _sid, job.job_id,
                "yes" if (isinstance(anchor, dict) and anchor.get("fps")) else "legacy",
                (len(compacted or []) - before) if (isinstance(anchor, dict) and anchor.get("fps")) else 0,
                _in, len(compacted or []), res.get("watermark"),
                _model,
            )
        )
        _log.info(
            "compress APPLIED session=%s job=%s msgs=%s->%s summary_chars=%s watermark=%s",
            _sid, job.job_id,
            _in, len(compacted or []),
            len((compacted or [{}])[0].get("content") or ""),
            self._last_bg_watermark,
        )
        self._arm_regrow_floor()  # [claude] 2026-10-28 anti-thrash
        _m("compress", path="bg-applied", session=_sid, msgs_in=_in,
           msgs_out=len(compacted or []), prompt_tokens=_pt, blocked_s=0.0,
           appended=_appended, model=_model, fallback_reason=None)
        # compacted goes to the host; the host does archive_and_compact + stamp itself. The plugin never writes the DB.
        return compacted or messages

    def update_from_response(self, usage) -> None:
        # ABC: abstractmethod — the engine must keep token state (run_agent
        # reads last_prompt_tokens / last_total_tokens directly). Mirrors
        # context_compressor (per-call tokens are always present).
        self.last_prompt_tokens = int(usage.get("prompt_tokens") or 0)
        self.last_completion_tokens = int(usage.get("completion_tokens") or 0)
        self.last_total_tokens = int(usage.get("total_tokens") or 0)

    def update_model(self, model: str, context_length: int,
                     base_url: str = "", api_key: str = "", provider: str = "", api_mode: str = "",
                     **kw):
        """The engine accepts ``auxiliary.compression.threshold_percent``.

        The host passes the percent to the built-in ContextCompressor via its constructor
        (agent_init), but NOT to an external engine (only update_model without a percent).
        So the engine resolves it from config itself BEFORE calling
        the base update_model (which computes threshold_tokens = ctx * percent).

        If auxiliary.compression sets threshold_percent (0 < p <= 1), it is used;
        otherwise the ContextEngine default (0.75) is inherited. Exact field name = ``threshold_percent``
        (context_engine), same kind as threshold_tokens_override — an explicit config
        parameter the engine owns; the host does not overwrite it.
        """
        try:
            tp = load_bg_config().get("threshold_percent")
            if isinstance(tp, (int, float)) and 0 < float(tp) <= 1:
                self.threshold_percent = float(tp)
                self._base_threshold_percent = float(tp)
        except Exception as e:  # noqa: BLE001 — never fail: default 0.75
            _log.warning("threshold_percent config unresolved (%s) — default 0.75", e)
        return super().update_model(
            model=model, context_length=context_length, base_url=base_url, api_key=api_key,
            provider=provider, api_mode=api_mode)

    def __deepcopy__(self, memo):
        """The host deep-copies the engine per agent (agent_init). Job state lives in the file
        registry, so a plain copy of every field is enough; the DB handle is not copyable and is
        re-bound by bind_session_state()."""
        cls = type(self)
        r = cls.__new__(cls)
        memo[id(self)] = r
        for k, v in self.__dict__.items():
            if k == "_session_db":
                r.__dict__[k] = None
                continue
            try:
                r.__dict__[k] = copy.deepcopy(v, memo)
            except Exception:  # noqa: BLE001 — uncopyable helper objects are shared, not copied
                r.__dict__[k] = v
        return r
