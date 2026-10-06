"""Requirement -> test traceability check (constitution P5).

Reads every REQ-ID from specs/01-requirements.md and looks for it in tests/.
    uv run python scripts/trace_matrix.py            # report
    uv run python scripts/trace_matrix.py --strict   # exit 1 if any MUST requirement is untested
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROW = re.compile(r"^\|\s*(REQ-[A-Z]+-\d+)\s*\|\s*([MSC])\s*\|")
REF = re.compile(r"REQ-[A-Z]+-\d+")


def load_requirements() -> dict[str, str]:
    reqs: dict[str, str] = {}
    for line in (ROOT / "specs" / "01-requirements.md").read_text().splitlines():
        if m := ROW.match(line):
            reqs[m.group(1)] = m.group(2)
    return reqs


def load_test_refs() -> dict[str, set[str]]:
    refs: dict[str, set[str]] = defaultdict(set)
    for path in (ROOT / "tests").rglob("*.py"):
        for req in REF.findall(path.read_text()):
            refs[req].add(path.relative_to(ROOT).as_posix())
    return refs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    reqs, refs = load_requirements(), load_test_refs()
    unknown = sorted(set(refs) - set(reqs))
    by_area: dict[str, list[str]] = defaultdict(list)
    for req in reqs:
        by_area[req.split("-")[1]].append(req)

    print(f"{'Area':<6} {'Tested':>7} {'Total':>6}  Untested (priority)")
    missing_must = []
    for area, ids in by_area.items():
        untested = [r for r in ids if r not in refs]
        missing_must += [r for r in untested if reqs[r] == "M"]
        listing = ", ".join(f"{r.split('-', 1)[1]}({reqs[r]})" for r in untested)
        print(f"{area:<6} {len(ids) - len(untested):>7} {len(ids):>6}  {listing}")
    tested = sum(1 for r in reqs if r in refs)
    print(f"\n{tested}/{len(reqs)} requirements have at least one test.")
    if unknown:
        print(f"Tests cite unknown IDs (typo or removed requirement): {', '.join(unknown)}")
    if args.strict and (missing_must or unknown):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
