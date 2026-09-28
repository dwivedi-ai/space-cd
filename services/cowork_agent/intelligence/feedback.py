"""A person's answer to "did this chat still complete your task?" (plan step 4b).

Asked in the Space UI on any session in Agents > Sessions: routed by XO or
started elsewhere (a terminal, another agent). Every answer is one line in
``~/.quirq/sessions/intelligence/feedback.jsonl``: machine-local, never
synced, with

- the session (as the UI names it: the agent's own id), its agent, and the
  model it ran on;
- for a session XO routed: its XO session id, project, and the layer it ran
  on (the decision line's ``applied``);
- a preview and hash of its first prompt (never the whole text), so answers
  on sessions nobody routed still say which kind of prompt a model handled;
- ``answer``: ``yes`` or ``no``. The latest answer for a session wins.

The report joins a routed session's answer to its decision
(:func:`latest_answers`), and ``labels.py`` ranks it above its own guesses.
Nothing here calls the decision model.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from services.cowork_agent.engine import sessions_io
from services.cowork_agent.intelligence import decision_log, profiles
from services.errors import ServiceError
from services.storage.atomic_write import append_jsonl
from services.storage.layout import sessions_dir
from services.storage.reader import read_jsonl_tail_reverse
from services.timestamps import now_iso

log = logging.getLogger(__name__)

TYPE = "intelligence.feedback"
SCHEMA = 1
FILENAME = "feedback.jsonl"
ANSWERS = ("yes", "no")
#: How far back a lookup reads the log.
READ_LIMIT = 5000

_AGENT = re.compile(r"[A-Za-z0-9_-]{1,64}")
_SESSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")


def log_path() -> Path:
    return sessions_dir() / decision_log.SUBDIR / FILENAME


def _check(agent: str, session_id: str) -> None:
    if not isinstance(agent, str) or not _AGENT.fullmatch(agent):
        raise ServiceError("invalid_agent", f"{agent!r} is not an agent name.")
    if not isinstance(session_id, str) or not _SESSION.fullmatch(session_id):
        raise ServiceError("invalid_session", "That is not a session id.")


def _xo_session(session_id: str) -> tuple[str | None, str] | None:
    """``(project, XO session id)`` for a session XO started, found by either
    id; ``None`` for a session XO did not start."""
    for project, _folder, index in sessions_io.iter_project_session_indexes():
        for row in index.values():
            if isinstance(row, dict) and session_id in (row.get("nativeSessionId"), row.get("sessionId")):
                xo_id = row.get("sessionId")
                if isinstance(xo_id, str) and xo_id:
                    return project or None, xo_id
    return None


def _routing(session_id: str) -> dict[str, Any]:
    """What XO knows of how a session was routed: nothing, or its ids and layer."""
    found = _xo_session(session_id)
    if found is None:
        return {"routed": False, "xo_session_id": None, "project_id": None, "applied": None}
    project, xo_id = found
    decision = decision_log.find(project, xo_id)
    applied = decision.get("applied") if isinstance(decision, dict) else None
    return {"routed": bool(applied), "xo_session_id": xo_id, "project_id": project,
            "applied": applied if isinstance(applied, dict) else None}


def _first_prompt(agent: str, session_id: str) -> str | None:
    """The session's first prompt, through the agent's ``session_prompts``
    capability (the one the session view reads). ``None`` when there is none."""
    from services.cowork_agent.adapters.loader import try_load_capability

    try:
        module = try_load_capability("session_prompts", agent=agent)
        collector = getattr(module, "collect_session_prompts", None) if module else None
        if not callable(collector):
            return None
        prompts = (collector(session_id) or {}).get("prompts") or []
    except Exception:  # noqa: BLE001 - an unreadable transcript only costs the preview
        log.info("intelligence: no first prompt for %s session %s", agent, session_id)
        return None
    first = prompts[0] if prompts else None
    if isinstance(first, dict) and first.get("turn") == 1 and isinstance(first.get("text"), str):
        return first["text"]
    return None


def _latest(match) -> dict[str, Any] | None:
    path = log_path()
    if not path.is_file():
        return None
    for line in read_jsonl_tail_reverse(path, limit=READ_LIMIT, types=frozenset({TYPE})):
        if match(line):
            return line
    return None


def status(agent: str, session_id: str) -> dict[str, Any]:
    """How a session was routed, and its latest answer. Blocking I/O."""
    _check(agent, session_id)
    routing = _routing(session_id)
    ids = {session_id, routing["xo_session_id"]} - {None}
    line = _latest(lambda l: l.get("agent") == agent and (l.get("session_id") in ids or l.get("xo_session_id") in ids))
    return {"session_id": session_id, "agent": agent, "routed": routing["routed"], "applied": routing["applied"],
            "answer": (line or {}).get("answer"), "answered_at": (line or {}).get("ts")}


def record(agent: str, session_id: str, answer: str, *, model: str | None) -> dict[str, Any]:
    """Store one answer; returns the session's :func:`status`. Blocking I/O.

    ``model`` is what the UI shows the session ran on; a routed session's
    layer names its own."""
    _check(agent, session_id)
    if answer not in ANSWERS:
        raise ServiceError("invalid_answer", "The answer must be yes or no.")
    if model is not None and not profiles.is_valid_model(model):
        raise ServiceError("invalid_model", f"{model!r} is not a model id.")
    routing = _routing(session_id)
    applied = routing["applied"]
    prompt = _first_prompt(agent, session_id)
    line = {
        "ts": now_iso(), "type": TYPE, "schema": SCHEMA,
        "session_id": session_id, "agent": agent, **routing,
        "model": (applied or {}).get("model") or model,
        "answer": answer,
        "prompt": decision_log.request_record(prompt) if prompt else None,
    }
    path = log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    append_jsonl(path, [line])
    return status(agent, session_id)


def latest_answers() -> dict[str, str]:
    """XO session id -> its latest answer, for the sessions XO routed."""
    path = log_path()
    if not path.is_file():
        return {}
    answers: dict[str, str] = {}
    for line in read_jsonl_tail_reverse(path, limit=READ_LIMIT, types=frozenset({TYPE})):
        xo_id = line.get("xo_session_id")
        if isinstance(xo_id, str) and xo_id not in answers and line.get("answer") in ANSWERS:
            answers[xo_id] = line["answer"]
    return answers
