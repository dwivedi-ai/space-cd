#!/usr/bin/env python3
"""Build a project's category cache for "where to look" (plan step 6).

    venv/bin/python scripts/intelligence_index.py draft <project>            # draft categories.json once
    venv/bin/python scripts/intelligence_index.py draft <project> --force    # redraft
    venv/bin/python scripts/intelligence_index.py draft --repo PATH --out FILE   # try on any git repo
    venv/bin/python scripts/intelligence_index.py show <project>
    venv/bin/python scripts/intelligence_index.py index <project> [--limit N]   # tag files: path rules free, the rest 1 unit each
    venv/bin/python scripts/intelligence_index.py check <project> [--sample N]  # Sage vs the path rules on N files (1 unit each)
    venv/bin/python scripts/intelligence_index.py index --repo PATH --categories FILE --out FILE [--limit N]

<project> is a folder name under the XO projects root. Only git repos with at
least XO_CONTEXT_MIN_FILES code files (default 150) get a list. Drafting runs
the active agent once, without tools; only file paths and the README excerpt
go into it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.cowork_agent.intelligence import categories, file_map  # noqa: E402
from services.xo_manifest import resolve_agent_name  # noqa: E402


def _print_list(document: dict) -> None:
    print(f"{len(document['categories'])} areas ({document.get('source')}, {document.get('code_files')} code files, {document.get('ts')}):")
    for item in document["categories"]:
        print(f"  {item['id']:28} {item['description']}")
        if item.get("paths"):
            print(f"  {'':28} paths: {', '.join(item['paths'])}")


async def _draft(args) -> int:
    if args.repo:
        repo = Path(args.repo).expanduser().resolve()
        ok, reason, files = await categories.eligible(repo)
        print(f"{repo}: {reason}")
        if not ok:
            return 1
        items = await categories.draft(repo, files, resolve_agent_name())
        out = categories.save(Path(args.out).expanduser(), items, code_file_count=len(files), source="drafted")
        _print_list(json.loads(out.read_text()))
        print(f"written to {out}")
        return 0
    document, what = await categories.ensure(args.project, resolve_agent_name(), force=args.force)
    print(f"{args.project}: {what}")
    if document:
        _print_list(document)
        print(f"file: {categories.path_for(args.project)}")
    return 0 if document else 1


def _target(args):
    """``(repo, areas, file map path)`` for a project or a --repo run, or ``None``."""
    if getattr(args, "repo", None):
        return (Path(args.repo).expanduser().resolve(), json.loads(Path(args.categories).expanduser().read_text()),
                Path(args.out).expanduser())
    areas = categories.load(args.project)
    if not areas:
        print(f"{args.project}: no category list; run `draft {args.project}` first")
        return None
    from services.cowork_agent import project_layout
    return project_layout.project_dir(args.project), areas, file_map.path_for(args.project)


async def _check(args) -> int:
    found = _target(args)
    if found is None:
        return 1
    repo, areas, target = found
    done = await file_map.check(repo, areas, target, sample=args.sample)
    rate = f"{done.agreed / done.checked:.0%}" if done.checked else "n/a"
    print(f"checked {done.checked} rule-tagged files (units {done.units}): Sage agrees on {done.agreed} ({rate})"
          + (f", STOPPED: {done.stopped}" if done.stopped else ""))
    for d in done.disagreements:
        print(f"  {d['path']}: rule says {d['rule']}, Sage says {', '.join(d['sage']) or 'no area'}")
    return 0 if not done.stopped else 1


async def _index(args) -> int:
    if args.repo:
        repo = Path(args.repo).expanduser().resolve()
        areas = json.loads(Path(args.categories).expanduser().read_text())
        target = Path(args.out).expanduser()
    else:
        repo_areas = categories.load(args.project)
        if not repo_areas:
            print(f"{args.project}: no category list; run `draft {args.project}` first")
            return 1
        from services.cowork_agent import project_layout
        repo, areas, target = project_layout.project_dir(args.project), repo_areas, file_map.path_for(args.project)
    done = await file_map.index(repo, areas, target, limit=args.limit)
    print(f"tagged {done.tagged}: {done.by_path} by path rules, {done.tagged - done.by_path} by Sage (units {done.units}); "
          f"unchanged {done.unchanged}, dropped {done.dropped}, "
          f"failed {done.failed}, not reached (cap/limit) {done.skipped_over_cap}"
          + (f", STOPPED: {done.stopped}" if done.stopped else ""))
    for error in done.errors:
        print(f"  {error}")
    print(f"file: {target}")
    return 0 if not done.stopped else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    draft = sub.add_parser("draft", help="draft the project's category list")
    draft.add_argument("project", nargs="?")
    draft.add_argument("--force", action="store_true", help="redraft even if a list exists")
    draft.add_argument("--repo", help="any git repo path, instead of a project")
    draft.add_argument("--out", help="with --repo: where to write the list")
    show = sub.add_parser("show", help="print the project's category list")
    show.add_argument("project")
    index = sub.add_parser("index", help="tag the project's files with its areas (path rules free, the rest 1 Levanto unit each)")
    index.add_argument("project", nargs="?")
    index.add_argument("--limit", type=int, help="tag at most N new or changed files this run")
    index.add_argument("--repo", help="any git repo path, instead of a project")
    index.add_argument("--categories", help="with --repo: the category list to tag against")
    index.add_argument("--out", help="with --repo: where to write the file map")
    check = sub.add_parser("check", help="ask Sage about a sample of rule-tagged files (1 unit each)")
    check.add_argument("project")
    check.add_argument("--sample", type=int, default=30, help="how many files to ask about (default 30)")
    args = parser.parse_args()

    if args.command == "check":
        return asyncio.run(_check(args))

    if args.command == "index":
        if bool(args.repo) == bool(args.project) or (args.repo and not (args.categories and args.out)):
            parser.error("index needs a <project>, or --repo PATH with --categories FILE and --out FILE")
        return asyncio.run(_index(args))

    if args.command == "show":
        document = categories.load(args.project)
        if not document:
            print(f"{args.project}: no category list")
            return 1
        _print_list(document)
        return 0
    if bool(args.repo) == bool(args.project) or (args.repo and not args.out):
        parser.error("draft needs a <project>, or --repo PATH with --out FILE")
    try:
        return asyncio.run(_draft(args))
    except categories.CategoryError as exc:
        print(f"draft failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
