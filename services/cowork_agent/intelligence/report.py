"""Read the decision logs back: one row per session, decision joined to outcome.

Read-only. Walks every decision log under the state root (a project's
``projects/<key>/intelligence/decisions.jsonl`` and the no-project
``sessions/intelligence/decisions.jsonl``), groups the lines by
``session_id``, and adds up each session's turn lines. Files edited come from
the watcher's per-session stats beside the log, mapped from XO's session id
to the agent's own through the session index. ``scripts/intelligence_report.py``
prints it; step 4 of the plan reads it to judge which profiles fit which work.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator

from services.cowork_agent.engine import sessions_io
from services.cowork_agent.intelligence import decision_log, labels, profiles
from services.storage.layout import projects_dir, sessions_dir


def log_files() -> list[Path]:
    files = sorted(projects_dir().glob(f"*/{decision_log.SUBDIR}/{decision_log.FILENAME}"))
    root = sessions_dir() / decision_log.SUBDIR / decision_log.FILENAME
    return files + ([root] if root.is_file() else [])


def _read(path: Path) -> Iterator[dict]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return
    for raw in text.splitlines():
        try:
            line = json.loads(raw)
        except ValueError:
            continue  # a torn or hand-edited line is skipped, not fatal
        if isinstance(line, dict) and line.get("session_id"):
            yield line


def _files_edited(path: Path) -> dict[str, int]:
    """XO session id -> distinct files edited, from the stats beside a project's log."""
    runtime = path.parent.parent
    if runtime.parent != projects_dir():
        return {}
    try:
        by_session = json.loads((runtime / "stats.json").read_text(encoding="utf-8")).get("by_session") or {}
    except (OSError, ValueError, AttributeError):
        return {}
    native_to_ours = {
        row.get("nativeSessionId"): row.get("sessionId")
        for row in sessions_io.read_session_index_at(runtime).values()
        if isinstance(row, dict) and row.get("nativeSessionId")
    }
    return {
        native_to_ours[native]: len(stats.get("files") or [])
        for native, stats in by_session.items()
        if native in native_to_ours and isinstance(stats, dict)
    }


def _sum(values: Iterable[Any]) -> float | None:
    numbers = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return round(sum(numbers), 6) if numbers else None


def _effort_rank(runtime: str | None):
    """A function ranking an effort level by the agent's ``efforts`` order."""
    config = profiles.load(runtime or "")
    efforts = list(config.efforts) if config else []
    top = max((efforts.index(p.effort) for p in config.profiles if p.effort in efforts), default=None) if config else None

    def rank(effort: str | None) -> int | None:
        return efforts.index(effort) if effort in efforts else None
    return rank, top


def session_rows(since: str | None = None) -> list[dict[str, Any]]:
    """One row per session with a decision or a turn line, oldest first,
    each labelled right-sized, under- or over-powered (``labels.py``)."""
    rows: list[dict[str, Any]] = []
    for path in log_files():
        files = _files_edited(path)
        by_session: dict[str, dict[str, Any]] = defaultdict(lambda: {"decision": None, "turns": []})
        for line in _read(path):
            if line.get("type") == decision_log.TYPE:
                by_session[line["session_id"]]["decision"] = line
            elif line.get("type") == decision_log.TURN_TYPE:
                by_session[line["session_id"]]["turns"].append(line)
        for session_id, parts in by_session.items():
            decision, turns = parts["decision"], parts["turns"]
            first = decision or turns[0]
            if since and (first.get("ts") or "") < since:
                continue
            sage = (decision or {}).get("sage") or {}
            choice = sage.get("choice") or {}
            outcomes = [t.get("outcome") or {} for t in turns]
            tokens = [o.get("tokens") or {} for o in outcomes]
            applied = (decision or turns[0]).get("applied") or {}
            rank, top = _effort_rank(first.get("runtime"))
            first_rank = rank(applied.get("effort"))
            later_request_ranks = [
                rank((t.get("applied") or {}).get("effort")) for t in turns
                if (t.get("applied") or {}).get("source") == "request" and not t.get("new_session")]
            raised = first_rank is not None and any(r is not None and r > first_rank for r in later_request_ranks)
            # The default passes no flags: today's own setup, the strongest one.
            no_flags = not applied.get("model") and not applied.get("effort")
            top_tier = no_flags or (top is not None and first_rank is not None and first_rank >= top)
            rows.append({
                "session_id": session_id,
                "project": first.get("project_id"),
                "started": first.get("ts"),
                "mode": first.get("mode"),
                "sage_chosen": choice.get("chosen"),
                "sage_probability": choice.get("probability"),
                "reason": ((decision or {}).get("decision") or {}).get("reason"),
                "decided_profile": ((decision or {}).get("decision") or {}).get("profile"),
                "tags": {k: v.get("applies") for k, v in (sage.get("tags") or {}).items() if isinstance(v, dict)},
                "sage_units": sage.get("units"),
                "sage_errors": [e.get("kind") for e in sage.get("errors") or []],
                "applied": applied,
                "turn_lines": len(turns),
                "agent_turns": _sum(o.get("turns") for o in outcomes),
                "cost_usd": _sum(o.get("cost_usd") for o in outcomes),
                "duration_ms": _sum(o.get("duration_ms") for o in outcomes),
                "tokens_in": _sum(t.get("input") for t in tokens),
                "tokens_out": _sum(t.get("output") for t in tokens),
                "tokens_cache_read": _sum(t.get("cache_read") for t in tokens),
                "failed_turns": sum(1 for t, o in zip(turns, outcomes) if t.get("agent_error") or o.get("is_error")),
                "files_edited": files.get(session_id),
                "top_tier": top_tier,
                "raised_by_request": raised,
            })
    rows.sort(key=lambda r: r["started"] or "")
    labels.apply(rows)
    return rows


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Counts for the whole window, and per decided profile the typical outcome."""
    decided = [r for r in rows if r["reason"] is not None]
    per_profile: dict[str, dict[str, Any]] = {}
    for profile, group in _group(r for r in decided if r["turn_lines"]):
        per_profile[profile] = {
            "sessions": len(group),
            "median_cost_usd": _median(r["cost_usd"] for r in group),
            "median_agent_turns": _median(r["agent_turns"] for r in group),
            "median_duration_s": _median((r["duration_ms"] or 0) / 1000 for r in group if r["duration_ms"]),
        }
    return {
        "sessions": len(rows),
        "with_decision": len(decided),
        "decisions_with_outcome": sum(1 for r in decided if r["turn_lines"]),
        "reasons": dict(Counter(r["reason"] for r in decided)),
        "decided_profiles": dict(Counter(r["decided_profile"] or "(default)" for r in decided)),
        "sage_units": _sum(r["sage_units"] for r in decided),
        "by_decided_profile": per_profile,
        "labels_by_setup": labels.counts(rows),
        "label_thresholds": labels.thresholds(rows),
    }


def _group(rows: Iterable[dict]) -> list[tuple[str, list[dict]]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["decided_profile"] or "(default)"].append(row)
    return sorted(groups.items())


def _median(values: Iterable[Any]) -> float | None:
    numbers = [v for v in values if isinstance(v, (int, float))]
    return round(statistics.median(numbers), 4) if numbers else None
