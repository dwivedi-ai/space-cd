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
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)

ENV_MODE = "XO_INTELLIGENCE_MODE"
ENV_TIMEOUT = "XO_INTELLIGENCE_TIMEOUT_S"

OFF = "off"
SHADOW = "shadow"
ON = "on"
MODES = (OFF, SHADOW, ON)

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
