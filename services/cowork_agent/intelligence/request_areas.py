"""Which of a project's areas a new request would change (plan step 6c).

In a project that is indexed (a category list, and a file map built from that
list), each new session's request gets a third Sage call beside the routing
``choice`` and ``tags``: a ``tags`` call over the project's areas, asking which
ones the task would need to change. It is sent at the same moment as the
other two and costs one more decision unit.

The answer is logged on the decision line (``areas``, as ``{id: [p, applies]}``),
recalibration groups a request by its primary area, and "where to look"
(``where_to_look.py``) shortlists files by the :func:`confident` areas. A
project that is not indexed gets no call.
"""

from __future__ import annotations

import logging
from typing import Any

from services.cowork_agent.intelligence import categories, classify, file_map
from services.levanto import client

log = logging.getLogger(__name__)

QUESTION_ID = "request_areas"
INSTRUCTIONS = (
    "The content is a task given to an AI coding agent working in this repository. "
    "Tag the functional areas of the code the task would need to change. "
    "Do not tag areas it only mentions or would only need to read."
)
#: Sage's probability at or above which an area it says applies is confident.
CONFIDENT_P = 0.8


def indexed(project: str | None) -> dict[str, Any] | None:
    """The project's category list when a file map with files was built from
    it; ``None`` otherwise. Blocking I/O."""
    if not project:
        return None
    document = categories.load(project)
    target = file_map.path_for(project)
    if not document or target is None or not target.is_file():
        return None
    mapped = file_map.load(target)
    if not mapped["files"] or mapped.get("categories_ts") != document.get("ts"):
        return None
    return document


def question(document: dict[str, Any]) -> dict:
    return {
        "id": QUESTION_ID, "kind": "tags", "instructions": INSTRUCTIONS,
        "tags": [{"id": a["id"], "name": f"{a['id']}: {a['description']}"} for a in document["categories"]],
    }


async def ask(content: str, document: dict[str, Any], *, timeout: float) -> dict[str, Any]:
    """``{areas, units, latency_ms}``; ``areas`` is ``None`` and ``error`` names
    the failure when Sage gave no answer. Never raises."""
    try:
        result = await client.decide(content, question(document), timeout=timeout)
    except Exception:  # noqa: BLE001 - a missing answer only costs the hand-over
        log.exception("intelligence: request areas failed")
        return {"areas": None, "units": 0, "latency_ms": None, "error": "exception"}
    if not result.ok:
        return {"areas": None, "units": 0, "latency_ms": result.latency_ms, "error": result.kind}
    ids = {a["id"] for a in document["categories"]}
    items = (result.data.get("result") or {}).get("tags") or []
    areas = {t["id"]: [t.get("probability"), t.get("applies")]
             for t in items if isinstance(t, dict) and t.get("id") in ids}
    return {"areas": areas, "units": classify._units(result.data), "latency_ms": result.latency_ms}


def confident(areas: dict[str, Any] | None) -> list[str]:
    """The areas Sage says apply with at least :data:`CONFIDENT_P`, strongest first."""
    picked = [(value[0], area) for area, value in (areas or {}).items()
              if isinstance(value, (list, tuple)) and len(value) >= 2 and value[1] is True
              and isinstance(value[0], (int, float)) and value[0] >= CONFIDENT_P]
    return [area for _p, area in sorted(picked, reverse=True)]
