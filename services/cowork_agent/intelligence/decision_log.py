"""The decision log: one JSON line per new session, machine-local.

- a project's chats: ``~/.quirq/projects/<key>/intelligence/decisions.jsonl``
- chats with no project: ``~/.quirq/sessions/intelligence/decisions.jsonl``

Never in the project folder and never synced. A line records what was asked
(a short preview and a hash, not the whole text), what Sage answered, the
profile that would be chosen and why, and what the session actually ran with.
``session_id`` is XO's session id; the session index maps it to the agent's
own id, which is how outcomes are joined to decisions later.

Writing never raises into a chat: a failure is logged and the line dropped.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from services.cowork_agent import project_layout
from services.storage.atomic_write import append_jsonl
from services.storage.layout import sessions_dir
from services.timestamps import now_iso

log = logging.getLogger(__name__)

TYPE = "intelligence.decision"
SCHEMA = 1
SUBDIR = "intelligence"
FILENAME = "decisions.jsonl"
PREVIEW_CHARS = 160
#: ``append_jsonl`` keeps a line whole only below the stream buffer (~8 KiB).
MAX_LINE_BYTES = 7000


def log_path(project: str | None) -> tuple[Path, dict[str, str]] | None:
    """Where a session's decision goes, and the identity fields its line carries.

    ``None`` when a project's runtime home cannot be resolved safely.
    """
    if not project:
        return sessions_dir() / SUBDIR / FILENAME, {}
    identity: dict[str, str] = {}
    meta = project_layout.load_project(project) or {}
    if meta.get("pid") and not meta.get("_template"):
        identity["pid"] = str(meta["pid"])
    identity["project_id"] = project
    root = project_layout.runtime_dir_for_project(project, create=True)
    if root is None:
        # The project folder is not there yet (the adapter creates it when it
        # starts the agent). Use the home keyed by its name, which
        # project_layout folds into the pid-keyed one once the pid exists.
        try:
            root = project_layout.runtime_dir(project_layout.runtime_key(project))
        except ValueError:
            log.warning("intelligence: no safe runtime home for project %r; decision not logged", project)
            return None
    return root / SUBDIR / FILENAME, identity


def request_record(content: str) -> dict[str, Any]:
    return {
        "preview": content[:PREVIEW_CHARS],
        "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "chars": len(content),
    }


def build_line(
    *,
    identity: dict[str, str],
    session_id: str | None,
    runtime: str,
    mode: str,
    request: dict[str, Any],
    profiles_sha256: str,
    sage: dict[str, Any],
    decision: dict[str, Any],
    applied: dict[str, Any],
    latency_ms: float,
) -> dict[str, Any]:
    line: dict[str, Any] = {"ts": now_iso(), "type": TYPE, "schema": SCHEMA, **identity}
    line.update({
        "session_id": session_id,
        "runtime": runtime,
        "mode": mode,
        "request": request,
        "profiles": {"sha256": profiles_sha256},
        "sage": sage,
        "decision": decision,
        "applied": applied,
        "latency_ms": latency_ms,
    })
    if len(json.dumps(line).encode("utf-8")) > MAX_LINE_BYTES:
        # Only a very long profile list gets here; its per-option
        # probabilities are what can be dropped.
        choice = (line.get("sage") or {}).get("choice")
        if isinstance(choice, dict):
            choice.pop("options", None)
    return line


def record(project: str | None, **fields: Any) -> Path | None:
    """Write one session's decision. Returns the file written, or ``None``.

    ``fields`` are :func:`build_line`'s, less ``identity``, which comes from
    the project. Blocking file I/O: call it from a worker thread.
    """
    try:
        target = log_path(project)
        if target is None:
            return None
        path, identity = target
        line = build_line(identity=identity, **fields)
        path.parent.mkdir(parents=True, exist_ok=True)
        append_jsonl(path, [line])
        return path
    except Exception:  # noqa: BLE001 - a lost log line must never cost a reply
        log.exception("intelligence: could not write the decision for session %s", fields.get("session_id"))
        return None
