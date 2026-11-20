"""anchors/retention/depth_of for the compressor metrics.

Pure functions without IO — split out so they can be tested independently and
reused from metrics.py (which only re-exports them).
"""

import os
import re

_PATH_RE = re.compile(r"/\S+")
_VERSION_RE = re.compile(r"\bv?\d+\.\d+(\.\d+)+\b")
_URL_RE = re.compile(r"https?://\S+")
_BTICK_RE = re.compile(r"`([^`]{3,80})`")
_HEX_RE = re.compile(r"\b[0-9a-f]{8,}\b")
_NUM_RE = re.compile(r"\b\d{4,}\b")

_PATH_MIN_LEN = 6
_YEAR_MIN, _YEAR_MAX = 1990, 2100


def _load_prefix():
    """Summary message prefix from summarize.py (relative + fallback)."""
    try:
        from .summarize import _SUMMARY_PREFIX
        return _SUMMARY_PREFIX
    except Exception:
        try:
            from asinccontext.summarize import _SUMMARY_PREFIX
            return _SUMMARY_PREFIX
        except Exception:
            return None


def _strip_trailing_punct(s):
    return s.rstrip(".,;:!?)]}\"'")


def anchors(text):
    """Set of checkable details: paths, versions, URLs, ``ids``, hex, numbers."""
    if not isinstance(text, str) or not text:
        return set()
    out = set()
    for m in _PATH_RE.finditer(text):
        p = _strip_trailing_punct(m.group(0))
        if len(p) >= _PATH_MIN_LEN and p.count("/") >= 2:
            out.add(p)
    for m in _VERSION_RE.finditer(text):
        out.add(m.group(0))
    for m in _URL_RE.finditer(text):
        out.add(_strip_trailing_punct(m.group(0)).lower())
    for m in _BTICK_RE.finditer(text):
        out.add(m.group(1))
    for m in _HEX_RE.finditer(text):
        out.add(m.group(0).lower())
    for m in _NUM_RE.finditer(text):
        num = m.group(0)
        if len(num) == 4 and _YEAR_MIN <= int(num) <= _YEAR_MAX:
            continue
        out.add(num)
    return out


def retention(source_text, summary_text):
    """(total, kept): how many source details survived in the summary. total==0 -> (0,0)."""
    src = anchors(source_text)
    if not src:
        return (0, 0)
    s = summary_text if isinstance(summary_text, str) else ""
    kept = 0
    for a in src:
        if a in s or (a.startswith("/") and os.path.basename(a) in s):
            kept += 1
    return (len(src), kept)


def depth_of(messages):
    """Number of messages whose content starts with the summary prefix."""
    prefix = _load_prefix()
    if prefix is None or not isinstance(messages, (list, tuple)):
        return 0
    n = 0
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, str) and content.startswith(prefix):
            n += 1
    return n