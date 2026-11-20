# hermes-async-context

**Background (non-blocking) context compression for [Hermes Agent](https://github.com/NousResearch/hermes-agent).**

The built-in `ContextCompressor` summarizes the conversation *inside* the turn: when the
prompt crosses the threshold, the agent stops, waits for the summary model, and only then
answers. On long tool-heavy sessions (and on slow or free summary models) that is a
visible 30–90 s freeze, every time.

This plugin is a drop-in `ContextEngine` that does the same job **in the background**:

1. When the prompt crosses the threshold, the engine takes a read-only snapshot of the
   session transcript and starts a separate worker process. The current turn continues
   normally.
2. The worker summarizes the older part of the conversation (with a chain of models and
   automatic fallback) and writes the result atomically to a small file registry.
3. On a later turn the host asks the engine to compress; the engine hands over the ready
   result. Messages that arrived while the worker was running are appended verbatim
   (anchor match on the snapshot tail), so nothing is lost.
4. The last two user turns (with their tool calls and results) stay verbatim; only older
   history is summarized.

In daily use on a small ARM server (a phone running postmarketOS) with ~10 agent profiles
and several Telegram bots, compression became practically invisible: over two months 263
compactions were applied from the background without holding a turn. 63 times the
background result was late and the blocking fallback ran instead, mostly during a period
when the free summary models were failing; after the model chain was fixed that became rare.

## Features

- **Non-blocking.** The summary is produced by a subprocess; turns are never held.
- **Survives agent rebuilds and restarts.** Job state lives in a file registry keyed by
  session id, not in memory: a fresh engine (model switch, `_retire_agent`, gateway restart)
  picks up the pending result. Worker liveness is checked by PID plus `/proc` start time,
  so a recycled PID is not mistaken for a live worker.
- **Safe apply.** A result is applied only if the snapshot anchor is found in the live
  transcript; stale results are rejected. The compacted list is validated (non-empty, no
  dangling `tool_calls`) before it reaches the host.
- **Model chain with warm-up.** `free_chain` entries are tried in order; a background
  warmer probes them so dead endpoints are skipped quickly (cool-downs 60 s / 5 min / 30 min),
  with an optional paid primary as the last resort.
- **Anti-thrash.** After a compaction, a new job starts only when the prompt grows by 25 %
  over the first post-compaction size (or reaches 80 % of the window).
- **Sync fallback.** If the background result does not arrive for several turns while over
  the threshold, the engine falls back to a normal blocking compression (can be disabled
  with `sync_fallback_disabled: true`). `/compress` stays synchronous, as the user expects.
- **Housekeeping.** Snapshots (full transcript copies) are deleted as soon as the worker
  reads them; leftovers older than 6 h and stale registry files older than 7 days are swept.
- **Metrics.** JSONL metrics (`metrics/compress.jsonl`): job start, model used, latency,
  anchor retention, apply path.

## Install

```bash
git clone https://github.com/e30goodman/hermes-async-context ~/.hermes/plugins/asinccontext
```

Then select the engine in the profile's `config.yaml`:

```yaml
context:
  engine: asinc-context

auxiliary:
  compression:
    threshold_percent: 0.75        # optional, default 0.75 of the context window
    free_chain:                    # tried in order; any OpenAI-compatible endpoint
      - name: or-nemotron
        base_url: https://openrouter.ai/api/v1
        model: nvidia/nemotron-3-ultra-550b-a55b:free
        key_env: OPENROUTER_API_KEY
      - name: local-proxy
        base_url: http://127.0.0.1:9656/v1
        model: deepseek-chat
        key_env: MY_PROXY_KEY
    # provider/model/key_env of a paid primary may be set as the last resort
    sync_fallback_disabled: false
```

Restart the gateway (`hermes gateway restart`). The directory name must be `asinccontext`
(the worker runs as `python -m asinccontext.bg_worker`).

Tuning via environment: `ASINCCONTEXT_TAIL_TURNS` (default 2), `ASINCCONTEXT_TAIL_MAX_CHARS`
(default 40000), `ASINCCONTEXT_PYTHON` (interpreter for the worker; default: the host's own).

## What it does on your machine (disclosure)

- **Background processes:** one short-lived worker per compression job and an occasional
  warm-up probe process. Both are children of the Hermes process and exit on their own.
- **Network calls:** only to the summary endpoints you configure in `free_chain` / primary.
  The transcript text being summarized is sent there (the same data the built-in compressor
  sends to `auxiliary.compression`). Secrets matching common key patterns are redacted
  before sending.
- **Files written:** only inside the plugin directory (`jobs/`, `snapshots/`, `metrics/`,
  `health.json`) and `$HERMES_HOME/logs/asinccontext.log`. Snapshots are deleted right after
  use. The worker receives its resolved config through a 0600 file that it deletes on read.
- **Credentials:** API keys are resolved through Hermes's profile-scoped secret lookup
  (`agent.secret_scope.get_secret`); the plugin does not read `.env` files or other
  credential stores.
- **No telemetry**, no self-updating code.

## Limits

- Linux is the tested platform. PID liveness uses `/proc` when available and falls back to
  `os.kill(pid, 0)`; macOS should work, Windows is untested.
- The engine imports a few internal Hermes helpers (`agent.auxiliary_client` resolution of
  the main provider); a major Hermes refactor may need a small update here.

## Tests

```bash
HERMES_AGENT_ROOT=/path/to/hermes-agent python tests/test_tail_compat.py
```

## License

MIT
