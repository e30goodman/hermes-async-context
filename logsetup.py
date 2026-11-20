"""Observability logging setup for the asinccontext plugin.

Adds a dedicated rotating file handler to the plugin logger so that
compression-path decisions are visible on disk.  Never raises: logging
problems must not take the agent down.
"""

import logging
import logging.handlers
import os
from datetime import datetime


def setup_logging() -> None:
    """Attach (idempotently) a rotating file handler to the 'asinccontext' logger."""
    try:
        home = os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes")
        logs = os.path.join(home, "logs")
        os.makedirs(logs, exist_ok=True)

        lg = logging.getLogger("asinccontext")
        lg.setLevel(logging.INFO)

        for h in lg.handlers:
            if getattr(h, "_asinc_handler", False):
                return

        h = logging.handlers.RotatingFileHandler(
            os.path.join(logs, "asinccontext.log"),
            maxBytes=2_000_000,
            backupCount=3,
            encoding="utf-8",
        )
        h._asinc_handler = True
        h.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s [pid %(process)d] %(message)s"
        ))
        lg.addHandler(h)
    except Exception:
        pass


def trace(msg: str) -> None:
    """Append one line to the asinccontext log, bypassing the logging module.

    Needed because the host calls logging.disable(logging.CRITICAL) in oneshot
    mode, which silences every logger and handler.  Never raises.
    """
    try:
        home = os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes")
        logs = os.path.join(home, "logs")
        os.makedirs(logs, exist_ok=True)
        path = os.path.join(logs, "asinccontext.log")
        line = f"{datetime.now().isoformat(timespec='seconds')} TRACE [pid {os.getpid()}] {msg}\n"
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass

