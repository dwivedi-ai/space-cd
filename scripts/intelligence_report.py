#!/usr/bin/env python3
"""Print the intelligence decision logs: one row per session, decision joined to outcome.

Read-only. Uses the same state root as the server (``QUIRQ_STATE_ROOT``, else
``~/.quirq``)::

    venv/bin/python scripts/intelligence_report.py                 # table + summary
    venv/bin/python scripts/intelligence_report.py --since 2026-09-24
    venv/bin/python scripts/intelligence_report.py --json > week.json
    venv/bin/python scripts/intelligence_report.py --csv > week.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.cowork_agent.intelligence import report  # noqa: E402

COLUMNS = [
    ("started", 20), ("project", 14), ("sage_chosen", 11), ("sage_probability", 5),
    ("reason", 13), ("applied", 12), ("turn_lines", 5), ("agent_turns", 6),
    ("cost_usd", 8), ("duration_ms", 9), ("files_edited", 5), ("tags", 30),
]


def _cell(row: dict, name: str) -> str:
    value = row.get(name)
    if name == "applied":
        value = (value or {}).get("profile") or (value or {}).get("effort") or "-"
    elif name == "tags":
        value = " ".join(k.replace("needs_", "") for k, v in (value or {}).items() if v) or "-"
    elif name == "sage_probability" and isinstance(value, float):
        value = f"{value:.2f}"
    elif name == "duration_ms" and isinstance(value, (int, float)):
        value = f"{value / 1000:.0f}s"
    elif name == "cost_usd" and isinstance(value, (int, float)):
        value = f"{value:.3f}"
    return "-" if value is None else str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--since", help="only sessions started at or after this ISO date/time (UTC)")
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--json", action="store_true", help="rows and summary as JSON")
    output.add_argument("--csv", action="store_true", help="rows as CSV")
    args = parser.parse_args()

    rows = report.session_rows(since=args.since)
    if args.json:
        json.dump({"summary": report.summary(rows), "sessions": rows}, sys.stdout, indent=2)
        print()
        return 0
    if args.csv:
        writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]) if rows else ["session_id"])
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()})
        return 0

    print("  ".join(name[:width].ljust(width) for name, width in COLUMNS))
    for row in rows:
        print("  ".join(_cell(row, name)[:width].ljust(width) for name, width in COLUMNS))
    print()
    print(json.dumps(report.summary(rows), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
