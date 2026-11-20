"""bg_cfg.py — how the background job finds its endpoint.

Resolution mirrors the built-in engine:
- config: auxiliary.compression.* (e.g. provider=openrouter, model=<summary model>)
- key: NOT os.environ directly but _scoped_key_env (agent/secret_scope: fail-closed,
  falls back to os.environ) — exactly like _try_openrouter in auxiliary_client.
  The background job is a subprocess of the Hermes process, so it inherits the parent environ where the key
  already lives ("inherit", not "re-read .env").
- provider=main: resolved NOT via an env var but via the main runtime
  (auxiliary_client _read_main_provider). Hardcoding OPENROUTER_API_KEY for
  main would fail auth — the path goes through _resolve_task_provider_model.
"""

import os
import sys


def _secret(name):
    """Credential by env-var name through the Hermes profile secret scope (agent.secret_scope)."""
    try:
        from agent.secret_scope import get_secret
    except ImportError:  # older Hermes
        from agent.auxiliary_client import _scoped_key_env
        return _scoped_key_env(name) or ""
    try:
        return get_secret(name) or ""
    except Exception:  # noqa: BLE001 - no scope under multiplexing: fail closed, never another profile
        return ""

# Inside the host the agent package is importable already; helper subprocesses get the host
# sys.path via background.worker_env(). HERMES_AGENT_ROOT is only a manual override.
if os.environ.get("HERMES_AGENT_ROOT"):
    sys.path.insert(0, os.environ["HERMES_AGENT_ROOT"])


def _resolve_main_runtime_credentials():
    """(api_key, base_url) for provider=main — credentials of the profile's MAIN model.

    Same path as the built-in engine: provider/model come from
    _read_main_model/_read_main_provider (auxiliary_client.py:2452-2453),
    the KEY comes from _secret("MAIN_PROVIDER_API_KEY") like _try_openrouter
    (auxiliary_client), NOT from _resolve_task_provider_model: that one returns
    only an explicit cfg_api_key and does not see the env scope (verified: key_len=0 even
    after init → auth failure). If the provider is unknown, the last
    fallback is OPENROUTER_API_KEY. Returns (None, None) only if the runtime is
    unavailable (emergency path, the fallback chain decides).
    """
    try:
        from agent.auxiliary_client import (
            _read_main_api_key,
            _read_main_base_url,
            _read_main_provider,
        )

        provider = (_read_main_provider() or "").strip().lower()
        base_url = _read_main_base_url() or None
        # _read_main_api_key reads model.api_key from config (usually empty — the key is in .env / the scope)
        api_key = _read_main_api_key() or ""
        if not api_key and provider:
            # The main provider key lives in the profile scope, like the built-in
            # _try_openrouter → _secret("OPENROUTER_API_KEY").
            api_key = _secret(f"{provider.upper()}_API_KEY") or ""
        if not api_key:
            api_key = _secret("OPENROUTER_API_KEY") or ""
        return api_key or None, base_url
    except Exception:
        return None, None


def load_bg_config() -> dict:
    """Full resolution of auxiliary.compression + free_chain + fallback_chain.

    Returns {provider, model, base_url, api_key, http_timeout, job_deadline,
    stall_warning, threshold_tokens_override, threshold_percent, context_length,
    fallback_chain, free_chain}.
    auxiliary.compression is the same section the built-in compressor uses.
    Key order (provider != "main"): aux.api_key → aux.key_env (env variable name)
    → OPENROUTER_API_KEY.

    free_chain — an optional list of free chain entries in
    auxiliary.compression. Format example (never store keys in plain text —
    only via key_env):

        auxiliary:
          compression:
            provider: openrouter
            model: deepseek-v4-flash-0731
            free_chain:
              - name: free-a
                model: some-free-model
                base_url: https://host-a.example/v1
                key_env: HOST_A_API_KEY
                timeout: 120
              - name: free-b
                model: some-free-model-b
                base_url: https://host-b.example/v1
                api_key: literal-key-here

    Fields: name (optional, otherwise assigned by chain.py), model, base_url,
    key_env OR api_key, timeout. Entries without model or base_url are skipped.
    A provider field in an entry is ignored.
    """
    from agent.auxiliary_client import _get_auxiliary_task_config

    aux = _get_auxiliary_task_config("compression") or {}
    provider = (aux.get("provider") or "").strip().lower()

    if provider == "main":
        api_key, base_url = _resolve_main_runtime_credentials()
    else:
        # Key/base: like the built-in engine (auxiliary_client):
        # from the profile scope with a fallback to os.environ.
        # Key priority: aux.api_key → aux.key_env → OPENROUTER_API_KEY.
        api_key = (aux.get("api_key") or "").strip() or None
        if not api_key and aux.get("key_env"):
            api_key = _secret(str(aux.get("key_env")).strip()) or None
        if not api_key:
            api_key = _secret("OPENROUTER_API_KEY") or None
        base_url = aux.get("base_url") or None

    return {
        "provider": provider,
        "model": aux.get("model"),
        "base_url": base_url,
        "api_key": api_key,
        # HTTP timeout of the model call — separate from the job lifetime.
        "http_timeout": aux.get("http_timeout", aux.get("timeout", 600)),
        "job_deadline": aux.get("job_deadline"),   # metric, not a killer (null in config)
        "stall_warning": aux.get("stall_warning"),  # "running long" alert threshold, not a kill
        # Explicit hard trigger threshold (for tests), otherwise None → the engine
        # uses the host-computed threshold_tokens (adapts to the model window).
        "threshold_tokens_override": aux.get("threshold_tokens_override"),
        # Percent trigger threshold (adaptive): if set in auxiliary.compression,
        # the engine sets it as threshold_percent (the host update_model computes ctx*percent);
        # if None/0.75 the engine keeps the ContextEngine default (0.75).
        "threshold_percent": aux.get("threshold_percent"),
        # sync_fallback_disabled: true = NEVER compress synchronously/blocking (pre-API).
        # When the background result is not ready / the snapshot fails, the turn is not blocked; the result is awaited on
        # the next turns. Default False = previous behaviour (blocking fallback).
        "sync_fallback_disabled": bool(aux.get("sync_fallback_disabled")),
        "context_length": aux.get("context_length"),
        "fallback_chain": aux.get("fallback_chain", []),
        "free_chain": _resolve_free_chain(aux.get("free_chain"), _secret),
    }


def _resolve_free_chain(raw, key_resolver) -> list:
    """Build free_chain from auxiliary.compression.free_chain.

    Each entry: name (opt.), model, base_url, key_env OR api_key, timeout.
    Entries without model or base_url are skipped. The key comes from api_key,
    otherwise from key_env via key_resolver (in production — _scoped_key_env).
    Returns a list of dicts {name, model, base_url, api_key, timeout}.
    """
    out = []
    for e in (raw or []):
        if not isinstance(e, dict):
            continue
        model = (e.get("model") or "").strip()
        base_url = (e.get("base_url") or "").strip()
        if not model or not base_url:
            continue
        api_key = (e.get("api_key") or "").strip() or None
        if not api_key and e.get("key_env"):
            api_key = key_resolver(str(e["key_env"]).strip()) or None
        out.append({
            "name": e.get("name"),
            "model": model,
            "base_url": base_url,
            "api_key": api_key,
            "timeout": e.get("timeout"),
            "reasoning_effort": e.get("reasoning_effort"),
            "headers": e.get("headers") if isinstance(e.get("headers"), dict) else None,
            "unique_session": bool(e.get("unique_session")),
        })
    return out