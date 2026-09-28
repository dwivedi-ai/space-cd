"""The file map: each code file's areas, tagged once (plan step 6b).

A file one area's path rules place (``categories.area_for_path``) is tagged
with that area for free (``source: "path"``). Every other tracked code file
(tests excluded, as in the pilot) gets one Sage ``tags`` call over the
project's categories (``source: "sage"``), given the file's path and an
outline: its docstring's first line and the names of its functions and
classes, at most 320 characters. Never the file's contents. Each file keeps
Sage's probability and verdict per area, with "not sure" (``null``) kept, so
an unsure file stays in the pool instead of being wrongly excluded.

Incremental: a file is re-tagged only when its git blob changes, deleted files
are dropped, and a new category list re-tags everything. Indexing is capped at
``XO_CONTEXT_INDEX_MAX_FILES`` files per project (default 2,000), stops at the
first ``402`` (balance), and saves as it goes, so a stopped run resumes where
it left off. Kept at ``~/.quirq/projects/<key>/intelligence/file_map.json``.

:func:`check` asks Sage about a sample of the rule-tagged files and counts how
often it agrees, to show whether the rules can be trusted. It never changes
the map.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from services.cowork_agent.intelligence import categories as categories_mod
from services.cowork_agent.intelligence import mode
from services.levanto import client
from services.storage.atomic_write import write_json_atomic
from services.timestamps import now_iso
from utils.commands import run

log = logging.getLogger(__name__)

FILENAME = "file_map.json"
SCHEMA = 1
ENV_MAX_FILES = "XO_CONTEXT_INDEX_MAX_FILES"
DEFAULT_MAX_FILES = 2000
CONCURRENCY = 6
SAVE_EVERY = 25
_READ_MAX_BYTES = 20_000
_OUTLINE_MAX = 320

QUESTION_ID = "file_areas"
INSTRUCTIONS = (
    "Tag the functional areas this source file belongs to. A file usually belongs to one to three "
    "areas. Judge from its path and outline; do not tag areas it only imports or mentions."
)

_TEST_PATH = re.compile(r"(^|/)(tests?|__tests__|spec)/|(^|/)test_[^/]*\.py$|_test\.(py|go)$|\.(test|spec)\.[mc]?[jt]sx?$")


def max_files() -> int:
    raw = (os.getenv(ENV_MAX_FILES, "") or "").strip()
    try:
        return max(1, int(raw)) if raw else DEFAULT_MAX_FILES
    except ValueError:
        return DEFAULT_MAX_FILES


def is_indexed(path: str) -> bool:
    return categories_mod.is_code_file(path) and not _TEST_PATH.search(path)


def outline(text: str, path: str) -> str:
    """Docstring's first line plus function/class names (the pilot's outline)."""
    parts: list[str] = []
    doc = re.search(r'^\s*(?:"""|\'\'\')\s*([^\n]+)', text)
    if doc:
        parts.append(doc.group(1).strip().rstrip('"\'').strip()[:120])
    if path.endswith(".py"):
        parts += [a or b for a, b in re.findall(r"^(?:async )?def (\w+)|^class (\w+)", text, re.M)]
    elif path.endswith((".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx")):
        comment = re.search(r"^\s*/[/*]\**\s*([^\n]+)", text)
        if comment and not parts:
            parts.append(comment.group(1)[:120])
        parts += re.findall(r"(?:export\s+)?(?:async\s+)?(?:function|class)\s+(\w+)", text)[:15]
    else:
        comment = re.search(r"(?:/\*|<!--|#)\s*([^\n]{8,})", text)
        if comment:
            parts.append(comment.group(1)[:120])
    return "; ".join(dict.fromkeys(p for p in parts if p))[:_OUTLINE_MAX]


def tags_question(areas: list[dict[str, str]]) -> dict:
    return {
        "id": QUESTION_ID, "kind": "tags", "instructions": INSTRUCTIONS,
        "tags": [{"id": a["id"], "name": f"{a['id']}: {a['description']}"} for a in areas],
    }


async def tracked_blobs(repo: Path) -> dict[str, str] | None:
    """``{path: blob sha}`` for the repo's indexable files, or ``None`` if not a git repo."""
    if not (repo / ".git").exists():
        return None
    result = await run(["git", "-C", str(repo), "ls-files", "-s", "-z"], timeout=30,
                       separate_stderr=True, log_label="intelligence: file map")
    if not result.ok:
        return None
    blobs: dict[str, str] = {}
    for entry in result.stdout.split("\0"):
        meta, _, path = entry.partition("\t")
        fields = meta.split()
        if path and len(fields) >= 2 and is_indexed(path):
            blobs[path] = fields[1]
    return blobs


def path_for(project: str) -> Path | None:
    """Where a project's file map lives, beside its category list."""
    categories_path = categories_mod.path_for(project)
    return categories_path.parent / FILENAME if categories_path else None


def load(target: Path) -> dict[str, Any]:
    try:
        document = json.loads(target.read_text(encoding="utf-8"))
        if isinstance(document, dict) and document.get("schema") == SCHEMA and isinstance(document.get("files"), dict):
            return document
    except (OSError, ValueError):
        pass
    return {"schema": SCHEMA, "categories_ts": None, "files": {}}


@dataclass
class IndexRun:
    """What one indexing run did."""

    tagged: int = 0
    by_path: int = 0  # of ``tagged``: placed by a path rule, no Sage call
    unchanged: int = 0
    dropped: int = 0
    failed: int = 0
    skipped_over_cap: int = 0
    units: int = 0
    stopped: str | None = None  # why it stopped early: "balance", "not_configured", "auth"
    errors: list[str] = field(default_factory=list)


def _content_for(repo: Path, path: str) -> str:
    try:
        with open(repo / path, "rb") as f:
            text = f.read(_READ_MAX_BYTES).decode("utf-8", errors="replace")
    except OSError:
        text = ""
    return f"File: {path}\nOutline: {outline(text, path)}"


async def index(repo: Path, areas_doc: dict[str, Any], target: Path, *, limit: int | None = None) -> IndexRun:
    """Tag what is new or changed; save as it goes. Never raises for a Sage failure."""
    report = IndexRun()
    blobs = await tracked_blobs(repo)
    if blobs is None:
        report.stopped = "not a git repository"
        return report
    document = load(target)
    if document.get("categories_ts") != areas_doc.get("ts"):
        document = {"schema": SCHEMA, "categories_ts": areas_doc.get("ts"), "files": {}}  # new list: re-tag all
    files = document["files"]
    for gone in [p for p in files if p not in blobs]:
        del files[gone]
        report.dropped += 1
    todo = [p for p in sorted(blobs) if (files.get(p) or {}).get("blob") != blobs[p]]
    report.unchanged = len(blobs) - len(todo)
    cap = max_files() - report.unchanged
    if limit is not None:
        cap = min(cap, limit)
    if len(todo) > max(cap, 0):
        report.skipped_over_cap = len(todo) - max(cap, 0)
        todo = todo[:max(cap, 0)]

    area_ids = {a["id"] for a in areas_doc["categories"]}
    for path in todo:
        ruled = categories_mod.area_for_path(path, areas_doc["categories"])
        if ruled is not None:
            files[path] = {"blob": blobs[path], "tags": {ruled: [1.0, True]}, "source": "path"}
            report.tagged += 1
            report.by_path += 1
    todo = [p for p in todo if (files.get(p) or {}).get("blob") != blobs[p]]

    question = tags_question(areas_doc["categories"])
    timeout = mode.sage_timeout_s()
    semaphore = asyncio.Semaphore(CONCURRENCY)
    stop = asyncio.Event()

    def save() -> None:
        document["updated"] = now_iso()
        target.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(target, document)

    async def tag(path: str) -> None:
        async with semaphore:
            if stop.is_set():
                return
            result = await client.decide(_content_for(repo, path), question, timeout=timeout)
        if not result.ok:
            if result.kind in ("balance", "not_configured", "auth"):
                report.stopped = report.stopped or result.kind
                stop.set()
            else:
                report.failed += 1
                if len(report.errors) < 5:
                    report.errors.append(f"{path}: {result.kind} {result.detail}"[:200])
            return
        items = (result.data.get("result") or {}).get("tags") or []
        files[path] = {
            "blob": blobs[path],
            "tags": {t["id"]: [t.get("probability"), t.get("applies")]
                     for t in items if isinstance(t, dict) and t.get("id") in area_ids},
            "source": "sage",
        }
        report.tagged += 1
        report.units += 1
        if report.tagged % SAVE_EVERY == 0:
            save()

    await asyncio.gather(*(tag(p) for p in todo))
    save()
    return report


#: Sage's probability for an area at or above which it counts as agreeing with a rule.
AGREE_P = 0.5


@dataclass
class CheckRun:
    """How often Sage agrees with the path rules on a sample of rule-tagged files."""

    checked: int = 0
    agreed: int = 0
    units: int = 0
    stopped: str | None = None
    disagreements: list[dict[str, Any]] = field(default_factory=list)


async def check(repo: Path, areas_doc: dict[str, Any], target: Path, *, sample: int = 30, seed: int = 0) -> CheckRun:
    """Ask Sage about up to ``sample`` rule-tagged files (1 unit each). Sage
    agrees when it says the rule's area applies, or gives it at least
    :data:`AGREE_P`. Read-only: the map is not changed."""
    report = CheckRun()
    files = load(target)["files"]
    ruled = sorted(p for p, entry in files.items() if entry.get("source") == "path")
    picked = random.Random(seed).sample(ruled, min(sample, len(ruled)))
    question = tags_question(areas_doc["categories"])
    area_ids = {a["id"] for a in areas_doc["categories"]}
    timeout = mode.sage_timeout_s()
    semaphore = asyncio.Semaphore(CONCURRENCY)
    stop = asyncio.Event()

    async def ask(path: str) -> None:
        async with semaphore:
            if stop.is_set():
                return
            result = await client.decide(_content_for(repo, path), question, timeout=timeout)
        if not result.ok:
            if result.kind in ("balance", "not_configured", "auth"):
                report.stopped = report.stopped or result.kind
                stop.set()
            return
        rule = next(iter(files[path]["tags"]))
        items = [t for t in (result.data.get("result") or {}).get("tags") or [] if isinstance(t, dict)]
        said = next((t for t in items if t.get("id") == rule), {})
        report.checked += 1
        report.units += 1
        if said.get("applies") is True or (said.get("probability") or 0) >= AGREE_P:
            report.agreed += 1
        else:
            report.disagreements.append({"path": path, "rule": rule,
                                         "sage": [t["id"] for t in items if t.get("applies") is True
                                                  and t.get("id") in area_ids]})

    await asyncio.gather(*(ask(p) for p in picked))
    report.disagreements.sort(key=lambda d: d["path"])
    return report
