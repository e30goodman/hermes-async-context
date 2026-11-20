"""background.py — file registry of jobs and launching the background summarizer.

Why a file registry (not an in-memory dict): `_retire_agent` rebuilds the agent AND the engine
INSIDE a live session on a model/routing switch (cli_chat_turn_mixin and others) — on a bot
that is routine. A fresh engine must pick up the result by session_id from a file
(jobs/<session_id>.json), not from a live object. __deepcopy__ does NOT solve this (retire builds
a new engine, not a copy).

Single interface: the start() factory creates a job and writes the registry; __init__ is
storage only. No in-memory dict[session_id].
"""

import json
import logging
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

_log = logging.getLogger("asinccontext")

# Timeout of an "orphaned" background job without a result. Reuses the compression timeout;
# if unavailable, a local default (like conversation_compression context_timeout_seconds).
DEADLINE_S = float(os.environ.get("ASINCCONTEXT_JOB_TTL_SECONDS", 120))

PLUGIN_DIR = Path(__file__).resolve().parent
PLUGINS_ROOT = PLUGIN_DIR.parent  # parent of the package dir -> `python -m asinccontext.*` resolves
JOBS_DIR = PLUGIN_DIR / "jobs"
SNAPSHOT_DIR = PLUGIN_DIR / "snapshots"
SNAPSHOT_TTL_S = 6 * 3600       # snapshots hold full transcripts: keep them only while a job may need them
JOB_FILE_TTL_S = 7 * 86400      # stale registry / floor / unconsumed out files


def worker_cmd(module):
    """Interpreter for helper subprocesses: the host's own Python (override: ASINCCONTEXT_PYTHON)."""
    return [os.environ.get("ASINCCONTEXT_PYTHON") or sys.executable, "-m", module]


def worker_env():
    """Child env that sees the same packages as the host process (+ the plugins root)."""
    env = dict(os.environ)
    env.pop("PYTHONHOME", None)
    paths = [str(PLUGINS_ROOT)] + [p for p in sys.path if p] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
    env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(paths))
    return env


def sweep_housekeeping(jobs_dir=None, snapshot_dir=None, now=None):
    """Delete old snapshots and stale job files. Best effort, never raises."""
    now = now or time.time()
    for d, ttl in ((Path(snapshot_dir or SNAPSHOT_DIR), SNAPSHOT_TTL_S), (Path(jobs_dir or JOBS_DIR), JOB_FILE_TTL_S)):
        try:
            for p in d.iterdir():
                try:
                    if p.is_file() and now - p.stat().st_mtime > ttl:
                        p.unlink()
                except OSError:
                    pass
        except OSError:
            pass


_BG_WORKER = worker_cmd("asinccontext.bg_worker")


def _pid_starttime(pid) -> "int | None":
    """Process starttime from /proc/<pid>/stat (field 22, in clock ticks) — pid identity.

    comm (field 2) is in parentheses and may contain ')' — split at the LAST ')',
    after it counting starts at field 3, so starttime (field 22) = index 19
    (verified: CLK_TCK=100, the 19th element after ')' is the start time).
    Returns None if /proc is unavailable/damaged/has another format — the caller
    falls back to kill(pid, 0).
    """
    try:
        stat = Path(f"/proc/{int(pid)}/stat").read_bytes()
    except (OSError, ValueError, TypeError):
        return None
    try:
        return int(stat[stat.rindex(b")") + 1:].split()[19])
    except (IndexError, ValueError):
        return None


def _pid_alive(pid, starttime=None) -> bool:
    """Is the process alive, via kill(pid, 0) (no signal is sent). ESRCH means dead.

    poll() on the retrieve path (proc=None) cannot read the process exit code —
    the only reliable sign that an orphaned worker
    is still running (and out will appear) is the liveness of its pid. A dead pid with
    no out = the result will never appear (out.json removed externally before
    consumed OR a crash before the atomic write) → dead at once, no "resumed" until the deadline.

    If the worker died and its PID was reused by another
    process, kill(pid,0) lies "alive" → poll() is "resumed" forever and the job is never
    restarted. Identity is checked against the pid_starttime recorded at start:
    if the current process has a different starttime it is ANOTHER
    process and the worker is treated as dead (starttime=None in an old record or
    /proc unavailable → fall back to kill(pid,0), no stricter check).
    """
    if not pid:
        return False
    if starttime is not None:
        cur = _pid_starttime(pid)
        if cur is not None:
            # pid alive but starttime differs → pid reused → dead.
            return cur == int(starttime)
        # /proc unavailable / other format → fall back to kill(pid,0)
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # another user's process — alive
    except OSError:
        return False
    return True


class BackgroundJob:
    """Single interface: the start() factory creates a job and writes the registry;
    __init__ is storage only. No in-memory dict[session_id]."""

    def __init__(self, session_id, job_id, proc, out_path, jobs_dir=JOBS_DIR):
        self.session_id = session_id
        self.job_id = job_id
        self.proc = proc
        self.out_path = out_path
        self.jobs_dir = jobs_dir

    def _rec(self):
        return json.loads((self.jobs_dir / f"{self.session_id}.json").read_text())

    @classmethod
    def retrieve(cls, session_id, jobs_dir=JOBS_DIR):
        # Registry file: {"job_id","out_path","started_monotonic","pid","consumed","deadline_s"}
        try:
            rec = json.loads((jobs_dir / f"{session_id}.json").read_text())
        except FileNotFoundError:
            return None
        if not rec:
            return None
        return cls(session_id, rec["job_id"], None, rec["out_path"], jobs_dir)

    def turns_waited(self) -> int:
        """How many TURNS (not should_compress calls) we have waited for this job's result.

        The host calls should_compress() several times per turn (once per API call),
        so an in-memory counter does not count turns. Only on_turn_complete() knows the turn
        boundary; it increments this field in the registry. The registry survives
        a process change (hermes -z: every turn is a new process).
        """
        try:
            return int(self._rec().get("turns_waited", 0))
        except Exception:  # noqa: BLE001 — registry missing/damaged: count 0
            return 0

    def bump_turns(self) -> int:
        """+1 to turns_waited with an ATOMIC registry write; returns the new value.

        tmp file next to it + os.replace: a concurrent reader (another process of the same sid)
        never sees half a JSON. On any error the current value is returned,
        nothing is raised: the turn counter must not break the turn.
        """
        try:
            path = self.jobs_dir / f"{self.session_id}.json"
            rec = json.loads(path.read_text())
            n = int(rec.get("turns_waited", 0)) + 1
            rec["turns_waited"] = n
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(rec))
            os.replace(tmp, path)
            return n
        except Exception:  # noqa: BLE001 — see docstring
            return self.turns_waited()

    @classmethod
    def start(cls, session_id, messages_snapshot_path, watermark, cfg, *, jobs_dir=JOBS_DIR):
        job_id = uuid.uuid4().hex[:12]
        out_path = jobs_dir / f"{session_id}.{job_id}.out.json"
        (jobs_dir / session_id).parent.mkdir(parents=True, exist_ok=True)
        # PYTHONPATH: we run the module; bg_worker itself does sys.path.insert(0, plugins).
        # Secrets are resolved here, inside the profile scope; the worker gets the ready config as a private 0600 file.
        cfg_path = None
        try:
            try:
                from .chain import stash_cfg, sweep_stale_cfg
            except ImportError:
                from asinccontext.chain import stash_cfg, sweep_stale_cfg
            sweep_stale_cfg(jobs_dir)
            sweep_housekeeping(jobs_dir)
            cfg_path = stash_cfg(cfg, jobs_dir, f"{session_id}.{job_id}")
        except Exception:
            cfg_path = None  # the worker resolves the config itself (legacy path)
        try:
            # --home/--session-id are only used by optional notifications.
            # session_id is the first argument of start(); home comes from the process HERMES_HOME.
            # If home is not resolved, the launch is NOT broken: both flags are simply skipped
            # (the rest of the flow is unchanged).
            _home = os.environ.get("HERMES_HOME")
            _notify_flags = (["--home", _home, "--session-id", session_id]
                             if _home else [])
        except Exception:  # noqa: BLE001
            _notify_flags = []
        try:
            proc = subprocess.Popen(
                [*_BG_WORKER,
                 "--msgs", str(messages_snapshot_path), "--watermark", str(watermark),
                 "--out", str(out_path), *(["--cfg", cfg_path] if cfg_path else []),
                 *_notify_flags],
                env=worker_env(),
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            if cfg_path:
                try:
                    os.remove(cfg_path)
                except OSError:
                    pass
            raise
        (jobs_dir / f"{session_id}.json").write_text(json.dumps({
            "job_id": job_id, "out_path": str(out_path),
            "started_monotonic": time.monotonic(), "pid": proc.pid, "consumed": False,
            # pid identity anchor: starttime from /proc.
            # An orphaned worker is judged alive only when it matches this field; if the pid
            # was reused, the starttime differs → the worker is treated as dead.
            "pid_starttime": _pid_starttime(proc.pid),
            # deadline = job_deadline (a METRIC, not a killer; null in config):
            # keeps "HTTP timeout" and "job lifetime" as two separate settings.
            "deadline_s": (cfg or {}).get("job_deadline"),
            "stall_warning_s": (lambda s: float(s) if s and float(s) > 0 else None)((cfg or {}).get("stall_warning")),
            "stall_warned": False}))
        return cls(session_id, job_id, proc, out_path, jobs_dir)

    def poll(self):
        """Live Popen: judged by its exit code."""
        if self.proc is not None and self.proc.poll() is None:
            return "running"
        # The result has arrived — done, whether or not we hold proc (retrieve path after _retire_agent).
        if Path(self.out_path).exists():
            return "done"
        if self.proc is not None:
            # Live Popen exited and there is no file → a real failure.
            return "dead"
        # No self.proc (retrieve after an agent rebuild) and no file: the exit code cannot be read,
        # so judge by the worker being ALIVE (pid from the registry): an orphaned Popen
        # may still be running and write out later (resumed). pid dead → out will NOT appear
        # (out.json removed externally before consumed OR a crash before the atomic write):
        # dead at once, so the next should_compress clears and RESTARTS the job instead of
        # sticking in "resumed" until the deadline (a silent non-restart).
        try:
            rec = self._rec()
        except Exception:
            # registry missing/damaged — do not stall forever, give a restart a chance
            return "dead"
        if _pid_alive(rec.get("pid"), rec.get("pid_starttime")):
            # Worker ALIVE (or pid reused): do NOT kill it for "running long".
            # Force-killing a live worker by deadline caused a refire loop
            # (a job was declared dead at ~150 s while it finished and wrote out
            # 17.9K chars). A live worker either finishes (→ out → done) or dies by itself
            # (read-timeout ~ http_timeout) → pid dead → self-heal below.
            elapsed = time.monotonic() - rec.get("started_monotonic", 0)
            sw = rec.get("stall_warning_s")
            if sw and elapsed > sw and not rec.get("stall_warned"):
                # stall-warning: LOG/alert only, never a kill.
                rec["stall_warned"] = True
                _log.warning(
                    "job %s running >%ss (stall, pid=%s, elapsed=%ss) — ALERT only, NOT firing dead",
                    rec.get("job_id"), int(sw), rec.get("pid"), int(elapsed),
                )
                try:
                    (self.jobs_dir / f"{self.session_id}.json").write_text(json.dumps(rec))
                except Exception:  # noqa: BLE001 — the alert flag is not critical
                    pass
            return "resumed"
        _log.warning(
            "job %s out missing and worker pid %s not alive — treating as dead (tamper/manual delete or crash)",
            rec.get("job_id"), rec.get("pid"),
        )
        return "dead"

    def collect(self):
        """CONSUME: mark consumed and remove the registry so it is not picked up twice."""
        rec = self._rec()
        rec["consumed"] = True
        (self.jobs_dir / f"{self.session_id}.json").write_text(json.dumps(rec))
        # out is written read-only (0o444) — restore write permission before reading
        # so the later clear() and editors do not trip over it.
        try:
            os.chmod(self.out_path, 0o644)
        except OSError:
            pass
        # result from out_path (contains {"compacted_messages":..., "watermark":...})
        return json.loads(Path(self.out_path).read_text())

    def clear(self):
        """Remove the registry file and an unconsumed out (restart after dead)."""
        (self.jobs_dir / f"{self.session_id}.json").unlink(missing_ok=True)
        try:
            Path(self.out_path).unlink(missing_ok=True)
        except Exception:
            pass