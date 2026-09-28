"""Where to look: a short, confident list of the files a new request most
likely needs (plan step 6d).

From the project's file map (6b) and the request's areas (6c):

1. **Shortlist** the files tagged with one of the request's confident areas;
   a file Sage was unsure about for that area stays in rather than being
   wrongly excluded.
2. **Rank** them by how well the request's words match each file's path and
   outline (BM25, word statistics from the whole map), plus up to
   :data:`PAST_CAP` for past sessions in the same areas that edited the file
   (the watcher's per-session stats, joined through the session index).
3. **Only when sure:** hand over at most :data:`MAX_FILES` files, and only
   when some file matches and, among more than :data:`MAX_FILES` matches, the
   top file scores at least :data:`MIN_LEAD` times the next-best outside the
   list. Otherwise nothing is sent and :attr:`Handover.record` says why.

The text stays under :data:`TEXT_MAX` characters and says it is a starting
point, not a constraint. It never includes file contents.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

from services.cowork_agent.engine import sessions_io
from services.cowork_agent.intelligence import decision_log, file_map, request_areas
from services.storage.reader import read_jsonl_tail_reverse

log = logging.getLogger(__name__)

KIND = "map"
MAX_FILES = 5
MIN_LEAD = 1.5
PAST_WEIGHT = 1.0
PAST_CAP = 3
TEXT_MAX = 600
#: How far back the decision log is read for past sessions' areas.
PAST_LOG_LIMIT = 2000

_K1, _B = 1.2, 0.75
_WORD = re.compile(r"[A-Za-z][a-z]+|[A-Z]+(?![a-z])|\d+")
_STOP = frozenset("""
    the and for with that this from into when what which where why how are was were not but all any its
    new use using should would could need needs please can you your our file files code make add fix
    change update does don have has will just also some more than then there their them they
""".split())


@dataclass
class Handover:
    """``text`` for the agent (``None``: nothing to hand over) and what the turn line logs."""

    text: str | None
    record: dict[str, Any]


def _words(text: str) -> list[str]:
    """Lowercase words of 3+ letters, split at ``_``, ``/``, ``.``, and camelCase."""
    return [w for w in (m.lower() for m in _WORD.findall(text or "")) if len(w) >= 3 and w not in _STOP]


def _document(path: str, entry: dict[str, Any]) -> list[str]:
    return _words(path) + _words(entry.get("outline") or "")


def _in_areas(entry: dict[str, Any], areas: Iterable[str]) -> bool:
    tags = entry.get("tags") or {}
    return any(isinstance(tags.get(a), list) and len(tags[a]) >= 2 and tags[a][1] in (True, None) for a in areas)


def rank(request: str, areas: list[str], files: dict[str, dict[str, Any]], past: Counter) -> list[tuple[str, float]]:
    """The shortlist, best first: ``[(path, score)]``."""
    documents = {path: _document(path, entry) for path, entry in files.items()}
    total = len(documents) or 1
    average = sum(len(d) for d in documents.values()) / total or 1.0
    frequency = Counter(word for d in documents.values() for word in set(d))
    query = set(_words(request))
    ranked: list[tuple[str, float]] = []
    for path, entry in files.items():
        if not _in_areas(entry, areas):
            continue
        counts = Counter(documents[path])
        norm = _K1 * (1 - _B + _B * len(documents[path]) / average)
        score = sum(
            math.log(1 + (total - frequency[w] + 0.5) / (frequency[w] + 0.5)) * counts[w] * (_K1 + 1) / (counts[w] + norm)
            for w in query if counts[w]
        )
        score += PAST_WEIGHT * min(past.get(path, 0), PAST_CAP)
        ranked.append((path, round(score, 4)))
    ranked.sort(key=lambda item: (-item[1], item[0]))
    return ranked


def _session_areas(project: str) -> dict[str, set[str]]:
    """XO session id -> its request's confident areas, from the decision log."""
    path = decision_log.existing_path(project)
    if path is None or not path.is_file():
        return {}
    found: dict[str, set[str]] = {}
    for line in read_jsonl_tail_reverse(path, limit=PAST_LOG_LIMIT, types=frozenset({decision_log.TYPE})):
        session_id = line.get("session_id")
        if session_id and session_id not in found:
            found[session_id] = set(request_areas.confident(line.get("areas")))
    return found


def past_edits(project: str, areas: set[str]) -> Counter:
    """How many past sessions in these areas edited each file. Blocking I/O."""
    edits: Counter = Counter()
    log_path = decision_log.existing_path(project)
    if log_path is None:
        return edits
    runtime = log_path.parent.parent
    try:
        by_session = json.loads((runtime / "stats.json").read_text(encoding="utf-8")).get("by_session") or {}
    except (OSError, ValueError, AttributeError):
        return edits
    native_to_ours = {
        row.get("nativeSessionId"): row.get("sessionId")
        for row in sessions_io.read_session_index_at(runtime).values()
        if isinstance(row, dict) and row.get("nativeSessionId")
    }
    session_areas = _session_areas(project)
    for native, stats in by_session.items():
        ours = native_to_ours.get(native)
        if ours and session_areas.get(ours, set()) & areas and isinstance(stats, dict):
            edits.update(set(f for f in stats.get("files") or [] if isinstance(f, str)))
    return edits


def _text(areas: list[str], top: list[str], also: list[str]) -> str:
    labels = [a.replace("_", " ") for a in areas]
    names = labels[0] if len(labels) == 1 else ", ".join(labels[:-1]) + " and " + labels[-1]

    def compose(paths: list[str], extra: list[str]) -> str:
        text = (f"XO Space, from this project's map (a starting point, not a constraint): the request looks like "
                f"work in {names}. Files most likely to matter: " + ", ".join(f"`{p}`" for p in paths) + ".")
        if extra:
            text += " Similar past work in these areas also edited " + ", ".join(f"`{p}`" for p in extra) + "."
        return text

    paths, extra = list(top), list(also)
    text = compose(paths, extra)
    while len(text) > TEXT_MAX and (extra or len(paths) > 1):
        if extra:
            extra.pop()
        else:
            paths.pop()
        text = compose(paths, extra)
    return text[:TEXT_MAX]


def _withheld(reason: str, areas: list[str] | None = None) -> Handover:
    record: dict[str, Any] = {"kind": KIND, "withheld": reason}
    if areas:
        record["areas"] = areas
    return Handover(text=None, record=record)


def build(project: str | None, request: str, areas: dict[str, Any] | None) -> Handover:
    """What to hand the agent for a new request, or why nothing. Blocking I/O; never raises."""
    try:
        confident = request_areas.confident(areas)
        if not confident:
            return _withheld("no confident area")
        target = file_map.path_for(project) if project else None
        files = file_map.load(target)["files"] if target is not None and target.is_file() else {}
        if not files:
            return _withheld("project not indexed", confident)
        past = past_edits(project, set(confident))
        positive = [(p, s) for p, s in rank(request, confident, files, past) if s > 0]
        if not positive:
            return _withheld("no file matches the request", confident)
        if len(positive) > MAX_FILES and positive[0][1] < MIN_LEAD * positive[MAX_FILES][1]:
            return _withheld("no file stands out", confident)
        top = [p for p, _s in positive[:MAX_FILES]]
        also = [p for p, _n in past.most_common() if p not in top and p in files][:2]
        text = _text(confident, top, also)
        shown = [p for p in top if f"`{p}`" in text]
        return Handover(text=text, record={"kind": KIND, "areas": confident, "files": shown,
                                           "also_edited": [p for p in also if f"`{p}`" in text]})
    except Exception:  # noqa: BLE001 - the reply matters more than the hand-over
        log.exception("intelligence: where to look failed for project %s", project)
        return _withheld("error")
