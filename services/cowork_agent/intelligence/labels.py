"""Was a session's setup right-sized? Labels from its outcome lines (plan step 4b).

A label is computed when the log is read, never stored, so the rules can
change without rewriting history:

- ``under``: the setup was too weak (never the top tier: there is nothing
  stronger to move to, and a long session there is just hard work). The
  session had failed turns; or a later
  turn's request raised the effort (the user asked for more); or, once a setup
  has at least :data:`MIN_GROUP` sessions, it took more agent turns or cost
  more than :data:`PERCENTILE`% of that setup's sessions.
- ``over``: the setup was stronger than needed. It ran on the top tier (the
  default, today's strongest setup, or the highest-effort profile), for one
  agent turn, on a request Sage said writes no code.
- ``right``: neither. Sessions without any finished turn get no label.

These labels are what recalibration (4b-2) counts per area.
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from typing import Any

UNDER = "under"
OVER = "over"
RIGHT = "right"

#: Sessions a setup needs before its percentiles mean anything.
MIN_GROUP = 5
PERCENTILE = 80

DEFAULT_KEY = "(default)"


def setup_key(applied: dict[str, Any] | None) -> str:
    """Which setup a session ran on, for grouping: its profile, else the default."""
    applied = applied or {}
    return applied.get("profile") or DEFAULT_KEY


def _percentile(values: list[float]) -> float:
    return statistics.quantiles(values, n=100, method="inclusive")[PERCENTILE - 1]


def thresholds(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per setup with enough finished sessions: the turns and cost above which
    a session counts as too long for that setup."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("turn_lines"):
            groups[setup_key(row.get("applied"))].append(row)
    out: dict[str, dict[str, Any]] = {}
    for key, group in groups.items():
        turns = [r["agent_turns"] for r in group if isinstance(r.get("agent_turns"), (int, float))]
        costs = [r["cost_usd"] for r in group if isinstance(r.get("cost_usd"), (int, float))]
        if len(group) < MIN_GROUP:
            continue
        out[key] = {
            "sessions": len(group),
            "turns": round(_percentile(turns), 2) if len(turns) >= MIN_GROUP else None,
            "cost_usd": round(_percentile(costs), 6) if len(costs) >= MIN_GROUP else None,
        }
    return out


def label(row: dict[str, Any], limits: dict[str, dict[str, Any]]) -> tuple[str | None, list[str]]:
    """``(label, reasons)`` for one report row."""
    if not row.get("turn_lines"):
        return None, []
    key = setup_key(row.get("applied"))
    tags = row.get("tags") or {}
    if row.get("top_tier"):
        if (row.get("agent_turns") or 0) <= 1 and tags.get("needs_code_writing") is False:
            return OVER, ["top tier for one agent turn that wrote no code"]
        return RIGHT, []
    reasons: list[str] = []
    if row.get("failed_turns"):
        reasons.append("failed turns")
    if row.get("raised_by_request"):
        reasons.append("a later request raised the effort")
    limit = limits.get(key) or {}
    if limit.get("turns") is not None and (row.get("agent_turns") or 0) > limit["turns"]:
        reasons.append(f"more agent turns than {PERCENTILE}% of {key} sessions")
    if limit.get("cost_usd") is not None and (row.get("cost_usd") or 0) > limit["cost_usd"]:
        reasons.append(f"costlier than {PERCENTILE}% of {key} sessions")
    if reasons:
        return UNDER, reasons
    return RIGHT, []


def apply(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Label every row in place (``label``, ``label_reasons``); returns the thresholds used."""
    limits = thresholds(rows)
    for row in rows:
        row["label"], row["label_reasons"] = label(row, limits)
    return limits


def counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Per setup: how many sessions got each label."""
    out: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        if row.get("label"):
            out[setup_key(row.get("applied"))][row["label"]] += 1
    return {key: dict(c) for key, c in sorted(out.items())}
