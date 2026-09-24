"""The switch for decision-model routing, read here and nowhere else.

``XO_INTELLIGENCE_MODE``:

- ``off`` (the default): no decision is made and nothing is sent anywhere.
  Chat runs as it always has; a request may still choose its own profile.
- ``shadow``: every new session is classified by Levanto Sage and the
  decision is logged, but the session runs as if it were ``off``.
- ``on``: the decision is applied.

Anything else is treated as ``off``, with one warning, so a typo can never
turn network calls on. ``XO_INTELLIGENCE_TIMEOUT_S`` bounds each Sage call
(default 8 s, clamped to 1–30).

``XO_INTELLIGENCE_CONTEXT`` switches what XO hands the agent each turn
(``context.py``), independently of the routing switch:

- ``off`` (the default): nothing is added to any turn.
- ``note``: a fixed, harmless note, to prove the channel end to end (plan
  step 5). What is actually relevant comes later (step 6).
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)

ENV_MODE = "XO_INTELLIGENCE_MODE"
ENV_TIMEOUT = "XO_INTELLIGENCE_TIMEOUT_S"
ENV_CONTEXT = "XO_INTELLIGENCE_CONTEXT"

OFF = "off"
SHADOW = "shadow"
ON = "on"
MODES = (OFF, SHADOW, ON)

CONTEXT_NOTE = "note"
CONTEXT_MODES = (OFF, CONTEXT_NOTE)

DEFAULT_TIMEOUT_S = 8.0

_warned: set[str] = set()


def mode() -> str:
    raw = (os.getenv(ENV_MODE, "") or "").strip().lower()
    if raw in MODES:
        return raw
    if raw and raw not in _warned:
        _warned.add(raw)
        log.warning("%s=%r is not one of %s; intelligence stays off", ENV_MODE, raw, ", ".join(MODES))
    return OFF


def sage_timeout_s() -> float:
    raw = (os.getenv(ENV_TIMEOUT, "") or "").strip()
    try:
        value = float(raw) if raw else DEFAULT_TIMEOUT_S
    except ValueError:
        value = DEFAULT_TIMEOUT_S
    return min(30.0, max(1.0, value))


def context_mode() -> str:
    raw = (os.getenv(ENV_CONTEXT, "") or "").strip().lower()
    if raw in CONTEXT_MODES:
        return raw
    if raw and f"context:{raw}" not in _warned:
        _warned.add(f"context:{raw}")
        log.warning("%s=%r is not one of %s; no context is added", ENV_CONTEXT, raw, ", ".join(CONTEXT_MODES))
    return OFF
