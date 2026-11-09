"""Tier-0 consistency check for project docs (S-20, INF-10). Stdlib only, Python >= 3.11.

Checks that every referenced ID (G05.3, GAP-07, Q-02, D-004, INF-5, S-12) is defined
somewhere, and that every relative Markdown link points to an existing file.

    python tools/check_docs.py        # exit code 0 = pass, 1 = fail
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REF = re.compile(r"\b(G\d\d\.\d+|GAP-\d\d|Q-\d\d|D-\d{3}|INF-\d+|S-\d\d)\b")
DEFS = [
    re.compile(r"^### (G\d\d\.\d+)", re.M),                      # goal headings
    re.compile(r"^\| (GAP-\d\d|Q-\d\d|D-\d{3}|INF-\d+) ", re.M),  # table rows in the record
    re.compile(r"\*\*(S-\d\d)", re.M),                            # bold standard labels
]
LINK = re.compile(r"\]\(([^)#\s]+)(?:#[^)]*)?\)")


def md_files() -> list[Path]:
    return [p for p in ROOT.rglob("*.md")
            if "third_party" not in p.parts and ".venv" not in p.parts]


def main() -> int:
    files = md_files()
    text = {p: p.read_text(encoding="utf-8") for p in files}
    defined = {m for t in text.values() for rx in DEFS for m in rx.findall(t)}
    errors = []
    for p, t in text.items():
        rel = p.relative_to(ROOT)
        for ref in sorted(set(REF.findall(t)) - defined):
            errors.append(f"{rel}: undefined ID {ref}")
        for target in LINK.findall(t):
            if "://" in target or target.startswith("mailto:"):
                continue
            if not (p.parent / target).exists():
                errors.append(f"{rel}: broken link {target}")
    for e in errors:
        print("FAIL", e)
    print(f"check_docs: {len(files)} files, {len(defined)} IDs defined, {len(errors)} problem(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
