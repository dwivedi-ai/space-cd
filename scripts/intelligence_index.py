#!/usr/bin/env python3
"""Build a project's category cache for "where to look" (plan step 6).

    venv/bin/python scripts/intelligence_index.py draft <project>            # draft categories.json once
    venv/bin/python scripts/intelligence_index.py draft <project> --force    # redraft
    venv/bin/python scripts/intelligence_index.py draft --repo PATH --out FILE   # try on any git repo
    venv/bin/python scripts/intelligence_index.py show <project>

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

from services.cowork_agent.intelligence import categories  # noqa: E402
from services.xo_manifest import resolve_agent_name  # noqa: E402


def _print_list(document: dict) -> None:
    print(f"{len(document['categories'])} areas ({document.get('source')}, {document.get('code_files')} code files, {document.get('ts')}):")
    for item in document["categories"]:
        print(f"  {item['id']:28} {item['description']}")


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
    args = parser.parse_args()

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
