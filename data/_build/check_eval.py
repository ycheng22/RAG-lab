"""Check that the eval set matches the corpus.

    uv run python data/_build/check_eval.py

For every question, each `evidence` string must appear (case-insensitive, whitespace-normalised)
in at least one of its `source_documents`. Also checks category counts and the 5-item quick subset.
Prints OK or the list of problems.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from bs4 import BeautifulSoup
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data" / "corpus"
QUESTIONS = ROOT / "data" / "eval" / "questions.json"

EXPECTED = {
    "factual": 10,
    "comparative": 5,
    "multi_hop": 6,
    "metadata_filter": 5,
    "unanswerable": 5,
    "conflicting_versions": 4,
    "no_search_needed": 5,
}
QUICK_CATEGORIES = {"factual", "multi_hop", "metadata_filter", "unanswerable", "conflicting_versions"}


def normalise(text: str) -> str:
    return " ".join(text.split()).lower()


def extract(path: Path) -> str:
    if path.suffix == ".pdf":
        return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    raw = path.read_text(encoding="utf-8")
    if path.suffix == ".html":
        return BeautifulSoup(raw, "html.parser").get_text(" ")
    return raw


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    texts = {p.name: normalise(extract(p)) for p in CORPUS.iterdir() if p.is_file()}
    items = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    problems: list[str] = []

    print(f"{'id':>3}  {'category':<22}{'evidence':>9}  status")
    for item in items:
        missing = []
        for ev in item["evidence"]:
            if not any(normalise(ev) in texts.get(src, "") for src in item["source_documents"]):
                missing.append(ev)
        for src in item["source_documents"]:
            if src not in texts:
                missing.append(f"<unknown source {src}>")
        status = "ok" if not missing else "MISSING: " + " | ".join(missing)
        print(f"{item['id']:>3}  {item['category']:<22}{len(item['evidence']):>9}  {status}")
        if missing:
            problems.append(f"#{item['id']}: {status}")

    counts = Counter(i["category"] for i in items)
    if dict(counts) != EXPECTED:
        problems.append(f"category counts {dict(counts)} != {EXPECTED}")
    quick = [i for i in items if i["quick"]]
    if len(quick) != 5 or {i["category"] for i in quick} != QUICK_CATEGORIES:
        problems.append(f"quick subset must be 5 items, one of each of {sorted(QUICK_CATEGORIES)}")
    if len({i["id"] for i in items}) != len(items):
        problems.append("duplicate ids")
    if "reykjavík" not in texts.get("hotel_profiles.pdf", ""):
        problems.append("'Reykjavík' did not survive PDF text extraction")

    print("\nOK" if not problems else "\nPROBLEMS:\n" + "\n".join(problems))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
