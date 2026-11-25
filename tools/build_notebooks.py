"""Build notebooks from their jupytext sources, and (by default) execute them.

    uv run python tools/build_notebooks.py                      # build + execute all
    uv run python tools/build_notebooks.py 01_rag_from_scratch  # just one (name without .py)
    uv run python tools/build_notebooks.py --no-execute         # convert only

Sources live in notebooks/_src/NN_name.py (percent format); outputs go to notebooks/NN_name.ipynb.
Execution runs with the notebooks/ folder as the working directory, like a learner would.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import jupytext
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "notebooks" / "_src"
OUT = ROOT / "notebooks"

KERNELSPEC = {"display_name": "Python 3 (raglab)", "language": "python", "name": "python3"}


def build(name: str, execute: bool, timeout: int) -> bool:
    src = SRC / f"{name}.py"
    nb = jupytext.read(src, fmt="py:percent")
    nb.metadata["kernelspec"] = KERNELSPEC
    nb.metadata["language_info"] = {"name": "python"}
    nb.metadata.pop("jupytext", None)
    out = OUT / f"{name}.ipynb"
    ok = True
    if execute:
        started = time.time()
        client = NotebookClient(nb, timeout=timeout, kernel_name="python3", resources={"metadata": {"path": str(OUT)}})
        try:
            client.execute()
        except CellExecutionError as err:
            ok = False
            print(f"✗ {name}: cell failed\n{str(err)[-3000:]}")
        print(f"{'✓' if ok else '✗'} {name} executed in {time.time() - started:.0f}s")
    nbformat.write(nb, out)
    return ok


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*", help="notebook names without extension (default: all)")
    parser.add_argument("--no-execute", action="store_true")
    parser.add_argument("--timeout", type=int, default=900, help="per-cell timeout in seconds")
    args = parser.parse_args()

    names = args.names or sorted(p.stem for p in SRC.glob("[0-9][0-9]_*.py"))
    if not names:
        print(f"No sources found in {SRC}")
        return 1
    failures = [n for n in names if not build(n, not args.no_execute, args.timeout)]
    if failures:
        print("Failed:", ", ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
