"""The switch for decision-model routing, read here and nowhere else.

``XO_INTELLIGENCE_MODE``:

- ``off`` (the default): no decision is made and nothing is sent anywhere.
  Chat runs as it always has; a request may still choose its own profile.
- ``shadow``: every new session is classified by Levanto Sage and the
  decision is logged, but the session runs as if it were ``off``.
- ``on``: the decision is applied.

Anything else is treated as ``off``, with one warning, so a typo can never
turn network calls on. ``XO_INTELLIGENCE_TIMEOUT_S`` bounds each Sage call
(default 8 s, clamped to 1–30). ``XO_INTELLIGENCE_WAIT_S`` sets how long a new
session's first turn waits for the decision in ``on`` (default 2 s, at most the
Sage timeout).

``XO_INTELLIGENCE_CONTEXT`` switches what XO hands the agent each turn
(``context.py``), independently of the routing switch:

- ``off`` (the default): nothing is added to any turn.
- ``note``: a fixed, harmless note, to prove the channel end to end (plan
  step 5). What is actually relevant comes later (step 6).

``XO_INTELLIGENCE_RECALIBRATE`` switches recalibration from the past record
(``recalibrate.py``, plan step 4b), on top of a decision:

- ``off`` (the default): nothing is looked up.
- ``shadow``: the one-tier correction the past record would make is logged on
  the decision line; the setup is not changed.
- ``on`` (with ``XO_INTELLIGENCE_MODE=on``): the correction is applied (step 4b-3): the
  new session starts one tier up or down, and the decision line says ``applied: true``.
  With routing in ``shadow`` it only logs, as ``shadow`` does.
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)

ENV_MODE = "XO_INTELLIGENCE_MODE"
ENV_TIMEOUT = "XO_INTELLIGENCE_TIMEOUT_S"
ENV_CONTEXT = "XO_INTELLIGENCE_CONTEXT"
ENV_RECALIBRATE = "XO_INTELLIGENCE_RECALIBRATE"
ENV_WAIT = "XO_INTELLIGENCE_WAIT_S"

OFF = "off"
SHADOW = "shadow"
ON = "on"
MODES = (OFF, SHADOW, ON)

CONTEXT_NOTE = "note"
CONTEXT_MAP = "map"
CONTEXT_MODES = (OFF, CONTEXT_NOTE, CONTEXT_MAP)

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


def first_turn_wait_s(default: float) -> float:
    """How long a new session's first turn waits for the decision in ``on``:
    ``XO_INTELLIGENCE_WAIT_S``, else ``default``; at most the Sage timeout."""
    raw = (os.getenv(ENV_WAIT, "") or "").strip()
    try:
        value = float(raw) if raw else default
    except ValueError:
        value = default
    return min(sage_timeout_s(), max(0.1, value))


def context_mode() -> str:
    raw = (os.getenv(ENV_CONTEXT, "") or "").strip().lower()
    if raw in CONTEXT_MODES:
        return raw
    if raw and f"context:{raw}" not in _warned:
        _warned.add(f"context:{raw}")
        log.warning("%s=%r is not one of %s; no context is added", ENV_CONTEXT, raw, ", ".join(CONTEXT_MODES))
    return OFF


def recalibrate_mode() -> str:
    raw = (os.getenv(ENV_RECALIBRATE, "") or "").strip().lower()
    if raw in MODES:
        return raw
    if raw and f"recalibrate:{raw}" not in _warned:
        _warned.add(f"recalibrate:{raw}")
        log.warning("%s=%r is not one of %s; recalibration stays off", ENV_RECALIBRATE, raw, ", ".join(MODES))
    return OFF
