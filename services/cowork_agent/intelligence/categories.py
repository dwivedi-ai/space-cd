"""A project's category list: 15–40 functional areas of its code (plan step 6a).

The foundation of the category cache. Every code file is tagged with its areas
once (step 6b), each new request is tagged too (6c), and both "where to look"
(6d) and recalibration (4b) work per area.

Only git repos with at least ``XO_CONTEXT_MIN_FILES`` code files (default
150) get a list; below that an agent finds files faster than a map helps.
The list is drafted once by a one-off, tool-free agent run (the agent's
``oneshot`` capability) from the repo's file list and README, then kept at
``~/.quirq/projects/<key>/intelligence/categories.json``, where a person may
edit it. It is never redrafted unless asked. Only paths and the README
excerpt go into the draft, never other file contents.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

from services.cowork_agent import project_layout
from services.cowork_agent.adapters.loader import try_load_capability
from services.cowork_agent.intelligence import decision_log
from services.storage.atomic_write import write_json_atomic
from services.timestamps import now_iso
from utils.commands import run

log = logging.getLogger(__name__)

ENV_MIN_FILES = "XO_CONTEXT_MIN_FILES"
DEFAULT_MIN_FILES = 150
FILENAME = "categories.json"
SCHEMA = 1

MIN_CATEGORIES = 12
MAX_CATEGORIES = 40
_ID = re.compile(r"[a-z][a-z0-9_]{1,39}")
_DESCRIPTION_MAX = 240
_PROMPT_MAX_FILES = 1500
_README_MAX_CHARS = 3000

CODE_EXTENSIONS = frozenset({
    ".py", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx", ".vue", ".svelte", ".go", ".rs",
    ".java", ".kt", ".kts", ".scala", ".rb", ".php", ".c", ".h", ".cc", ".cpp", ".hpp",
    ".cs", ".swift", ".m", ".sh", ".bash", ".sql", ".html", ".css", ".scss",
})


class CategoryError(ValueError):
    """A drafted category list is not usable."""


def min_files() -> int:
    raw = (os.getenv(ENV_MIN_FILES, "") or "").strip()
    try:
        return max(1, int(raw)) if raw else DEFAULT_MIN_FILES
    except ValueError:
        return DEFAULT_MIN_FILES


def is_code_file(path: str) -> bool:
    return Path(path).suffix.lower() in CODE_EXTENSIONS


async def code_files(repo: Path) -> list[str] | None:
    """The repo's tracked code files, or ``None`` when it is not a git repo."""
    if not (repo / ".git").exists():
        return None
    result = await run(["git", "-C", str(repo), "ls-files", "-z"], timeout=30, separate_stderr=True,
                       log_label="intelligence: categories")
    if not result.ok:
        return None
    return sorted(p for p in result.stdout.split("\0") if p and is_code_file(p))


async def eligible(repo: Path) -> tuple[bool, str, list[str]]:
    """``(eligible, reason, code files)`` for a category list."""
    files = await code_files(repo)
    if files is None:
        return False, "not a git repository", []
    needed = min_files()
    if len(files) < needed:
        return False, f"{len(files)} code files, fewer than {needed}", files
    return True, f"{len(files)} code files", files


def _readme(repo: Path) -> str:
    for name in ("README.md", "README.rst", "README.txt", "README"):
        path = repo / name
        if path.is_file():
            try:
                return path.read_text(encoding="utf-8", errors="replace")[:_README_MAX_CHARS]
            except OSError:
                return ""
    return ""


def draft_prompt(repo: Path, files: list[str]) -> str:
    folders = Counter(str(Path(f).parent) for f in files)
    folder_lines = "\n".join(f"  {folder or '.'}/  ({n})" for folder, n in sorted(folders.items()))
    listed = files[:_PROMPT_MAX_FILES]
    more = f"\n  … and {len(files) - len(listed)} more" if len(files) > len(listed) else ""
    readme = _readme(repo)
    return (
        "You are mapping a code repository into functional areas, so that each file can later be "
        "tagged with the areas it belongs to and each coding task with the areas it touches.\n\n"
        f"Repository: {repo.name} ({len(files)} tracked code files)\n\n"
        + (f"README (excerpt):\n{readme}\n\n" if readme else "")
        + f"Folders (code files per folder):\n{folder_lines}\n\nCode files:\n  "
        + "\n  ".join(listed) + more + "\n\n"
        f"Return {MIN_CATEGORIES} to {MAX_CATEGORIES} areas. An area is a function of the code "
        "(what it does for the product), not a file type or a folder name. Every file should "
        "belong to at least one area. Give each area a lowercase snake_case id and a one-line "
        "description saying what belongs in it and what does not, for example: "
        '"the chat prompt API and streaming replies; not per-agent adapter details".\n\n'
        'Reply with JSON only, no prose: {"categories": [{"id": "...", "description": "..."}]}'
    )


def parse(reply: str) -> list[dict[str, str]]:
    """The categories in a drafted reply. Raises :class:`CategoryError`."""
    start, end = reply.find("{"), reply.rfind("}")
    if start < 0 or end <= start:
        raise CategoryError("the reply holds no JSON object")
    try:
        document = json.loads(reply[start:end + 1])
    except ValueError as exc:
        raise CategoryError(f"the reply is not valid JSON: {exc}") from None
    items = document.get("categories") if isinstance(document, dict) else None
    if not isinstance(items, list):
        raise CategoryError("the reply has no `categories` list")
    return validate(items)


def validate(items: list) -> list[dict[str, str]]:
    if not MIN_CATEGORIES <= len(items) <= MAX_CATEGORIES:
        raise CategoryError(f"{len(items)} categories; expected {MIN_CATEGORIES}–{MAX_CATEGORIES}")
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in items:
        cid = item.get("id") if isinstance(item, dict) else None
        description = item.get("description") if isinstance(item, dict) else None
        if not isinstance(cid, str) or not _ID.fullmatch(cid):
            raise CategoryError(f"category id {cid!r} is not lowercase snake_case (2–40 chars)")
        if cid in seen:
            raise CategoryError(f"category id {cid!r} is used twice")
        if not isinstance(description, str) or not description.strip():
            raise CategoryError(f"category {cid!r} has no description")
        seen.add(cid)
        out.append({"id": cid, "description": description.strip()[:_DESCRIPTION_MAX]})
    return out


def path_for(project: str) -> Path | None:
    """Where a project's list lives, or ``None`` without a safe runtime home."""
    target = decision_log.existing_path(project)
    return target.parent / FILENAME if target and project else None


def load(project: str) -> dict[str, Any] | None:
    path = path_for(project)
    if path is None or not path.is_file():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        document["categories"] = validate(document.get("categories") or [])
        return document
    except (OSError, ValueError, AttributeError) as exc:
        log.warning("intelligence: %s is not usable: %s", path, exc)
        return None


async def draft(repo: Path, files: list[str], agent_name: str) -> list[dict[str, str]]:
    """Draft a category list with the agent's ``oneshot`` capability.
    Raises :class:`CategoryError` when there is no capability or no usable reply."""
    oneshot = try_load_capability("oneshot", agent=agent_name)
    if oneshot is None:
        raise CategoryError(f"{agent_name} cannot draft (no oneshot capability)")
    reply = await oneshot.complete(draft_prompt(repo, files))
    if reply is None:
        raise CategoryError("the drafting run failed; see the server log")
    return parse(reply)


def save(target: Path, categories: list[dict[str, str]], *, code_file_count: int, source: str) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(target, {
        "schema": SCHEMA,
        "source": source,
        "ts": now_iso(),
        "code_files": code_file_count,
        "categories": categories,
    })
    return target


async def ensure(project: str, agent_name: str, *, force: bool = False) -> tuple[dict[str, Any] | None, str]:
    """The project's list, drafting it if it is eligible and has none.
    Returns ``(document or None, what happened)``. Never raises."""
    try:
        existing = None if force else load(project)
        if existing is not None:
            return existing, "kept the existing list"
        repo = project_layout.project_dir(project)
        ok, reason, files = await eligible(repo)
        if not ok:
            return None, f"not eligible: {reason}"
        target = path_for(project)
        if target is None:
            return None, "no safe runtime home for this project"
        categories = await draft(repo, files, agent_name)
        save(target, categories, code_file_count=len(files), source="drafted")
        return load(project), f"drafted {len(categories)} areas from {len(files)} code files"
    except CategoryError as exc:
        return None, f"draft failed: {exc}"
    except Exception as exc:  # noqa: BLE001 - background work must not raise
        log.exception("intelligence: category list for %s failed", project)
        return None, f"failed: {type(exc).__name__}"
