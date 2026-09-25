"""Recalibrate a new session's decision from the past record (plan step 4b).

When routing keeps getting one kind of request wrong, the next request of that
kind should route differently. Here, when a new session is decided:

1. Its **key** says what kind of request it is: its primary area (the
   confident area with the highest probability, once requests are tagged with
   the project's areas), else the three ``needs_*`` facts Sage tagged it with,
   e.g. ``code=1,search=0,outside=0`` (``?`` where Sage was unsure).
2. The **evidence** is the earlier finished sessions in the same log (the
   project's, or the no-project log) with the same key that ran on the same
   setup as the one decided now: the picked profile, or the default. Each is
   labelled by ``labels.py``, exactly as the report labels it, from the last
   :data:`READ_LIMIT` lines of every log.
3. The **rule**: with at least :data:`MIN_EVIDENCE` such sessions, where at
   least :data:`MIN_RATE` of them were under-powered, move one tier up the
   agent's ``tiers`` ladder; where as many were over-powered, one tier down.
   The default sits above the ladder's top; a profile off the ladder is never
   moved. The strongest setup is never labelled under, so it never goes up.

Switched by ``XO_INTELLIGENCE_RECALIBRATE`` (``mode.py``). Step 4b-2 only
logs: the decision line gets a ``correction`` (``applied: false``) when the
record would move the setup, else a small ``recalibrate`` saying why not. An
explicit profile, model or effort in the request is never recalibrated.
Nothing here ever raises into a chat.
"""

from __future__ import annotations

import logging
from typing import Any

from services.cowork_agent.intelligence import labels, mode, profiles, report, selection

log = logging.getLogger(__name__)

#: Lines read from the end of each decision log.
READ_LIMIT = 2000
#: Past sessions of the same key and setup needed before anything moves.
MIN_EVIDENCE = 5
#: Share of them labelled one way needed to move one tier that way.
MIN_RATE = 0.6

UP = "up"
DOWN = "down"

# Short names for the request's own facts (``classify.TAGS``), in key order.
_KEY_TAGS = (("code", "needs_code_writing"), ("search", "needs_search"), ("outside", "needs_external_lookup"))


def _flag(value: object) -> str:
    return "1" if value is True else "0" if value is False else "?"


def key(row: dict[str, Any]) -> str | None:
    """What kind of request a session was, for matching it to past ones.

    ``row`` is shaped like a report row: ``tags`` maps each fact to Sage's
    ``applies``, ``areas`` (optional) maps an area to ``[p, applies]``.
    """
    areas = row.get("areas")
    if isinstance(areas, dict):
        confident = [
            (value[0], area) for area, value in areas.items()
            if isinstance(value, (list, tuple)) and len(value) >= 2 and value[1] is True
            and isinstance(value[0], (int, float))
        ]
        if confident:
            return f"area={max(confident)[1]}"
    tags = row.get("tags") or {}
    if not any(name in tags for _short, name in _KEY_TAGS):
        return None
    return ",".join(f"{short}={_flag(tags.get(name))}" for short, name in _KEY_TAGS)


_key_of = key  # ``rule`` takes a ``key`` argument, which hides the function


def neighbour(config: profiles.IntelligenceConfig, profile: str | None, direction: str) -> str | None:
    """The profile one tier ``up`` or ``down`` from ``profile`` (``None``: the
    default), or ``None`` when there is nowhere to go."""
    tiers = config.tiers
    if not tiers:
        return None
    if profile is None:
        return tiers[-1] if direction == DOWN else None
    if profile not in tiers:
        return None
    index = tiers.index(profile) + (1 if direction == UP else -1)
    return tiers[index] if 0 <= index < len(tiers) else None


def rule(
    config: profiles.IntelligenceConfig,
    *,
    profile: str | None,
    key: str | None,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """What the record says about running ``profile`` (``None``: the default)
    for a request with ``key``. A move has ``to``; otherwise ``result`` says why not."""
    if key is None:
        return {"result": "no_key"}
    setup = profile or labels.DEFAULT_KEY
    matching = [
        r for r in rows
        if r.get("label") and labels.setup_key(r.get("applied")) == setup and _key_of(r) == key
    ]
    evidence = {"key": key, "evidence": len(matching)}
    if not config.tiers or (profile is not None and profile not in config.tiers):
        return {**evidence, "result": "untiered"}
    if len(matching) < MIN_EVIDENCE:
        return {**evidence, "result": "insufficient"}
    for direction, verdict in ((UP, labels.UNDER), (DOWN, labels.OVER)):
        rate = sum(1 for r in matching if r["label"] == verdict) / len(matching)
        target = neighbour(config, profile, direction)
        if rate >= MIN_RATE and target:
            return {"from": setup, "to": target, **evidence, "rate": round(rate, 2), "direction": direction}
    return {**evidence, "result": "none"}


def evaluate(
    config: profiles.IntelligenceConfig,
    *,
    project: str | None,
    session_id: str | None,
    profile: str | None,
    sage: dict[str, Any],
    request: selection.RequestChoice | None = None,
    areas: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The fields recalibration adds to a decision line: ``{}`` when it is
    off, else ``correction`` or ``recalibrate``. Blocking I/O; never raises."""
    current = mode.recalibrate_mode()
    if current == mode.OFF:
        return {}
    try:
        if request is not None and not request.empty:
            return {"recalibrate": {"mode": current, "result": "request"}}
        tags = {k: v.get("applies") for k, v in ((sage or {}).get("tags") or {}).items() if isinstance(v, dict)}
        request_key = key({"tags": tags, "areas": areas})
        past = []
        if request_key is not None:
            past = [
                r for r in report.session_rows(limit=READ_LIMIT)
                if r.get("project") == (project or None) and r.get("session_id") != session_id
            ]
        result = rule(config, profile=profile, key=request_key, rows=past)
    except Exception:  # noqa: BLE001 - a missing correction must never cost a decision
        log.exception("intelligence: recalibrating session %s failed", session_id)
        return {"recalibrate": {"mode": current, "result": "error"}}
    if "to" in result:
        return {"correction": {"mode": current, **result, "applied": False}}
    return {"recalibrate": {"mode": current, **result}}
