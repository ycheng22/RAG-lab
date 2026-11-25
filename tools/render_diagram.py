"""Render an Archify diagram: JSON source -> interactive HTML -> static SVG -> PNG.

    uv run python tools/render_diagram.py series_map            # one diagram
    uv run python tools/render_diagram.py --all                 # every diagrams/*.json

Source: diagrams/<name>.json (Archify typed JSON; its meta.output must be "diagrams/<name>.html").
Outputs: diagrams/<name>.html (interactive), diagrams/<name>.svg and diagrams/<name>.png (static).
Archify is located at $ARCHIFY_HOME or ~/.claude/skills/archify (install it as a Claude Code skill).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIAGRAMS = ROOT / "diagrams"
RECEIPTS = ROOT / ".cache" / "archify"


def archify_home() -> Path:
    home = Path(os.environ.get("ARCHIFY_HOME", Path.home() / ".claude" / "skills" / "archify"))
    if not (home / "bin" / "archify.mjs").exists():
        sys.exit(f"Archify not found at {home}. Install the archify skill or set ARCHIFY_HOME.")
    return home


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    env = {**os.environ, "ARCHIFY_UPDATE_CHECK_DISABLED": "1"}
    return subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8")


def svg_to_png(svg: Path, png: Path, width: int) -> None:
    """Rasterise the static SVG at `width` CSS px (2x device pixels) on a white page."""
    from playwright.sync_api import sync_playwright

    markup = svg.read_text(encoding="utf-8")
    page_html = (
        "<!doctype html><html><body style='margin:0;background:#fff'>"
        f"<div id='d' style='width:{width}px'>{markup}</div>"
        "<style>#d svg{width:100% !important;height:auto !important;display:block}</style></body></html>"
    )
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": width, "height": 800}, device_scale_factor=2)
        page.set_content(page_html, wait_until="load")
        page.locator("#d svg").first.screenshot(path=str(png))
        browser.close()


def render(name: str, width: int, theme: str) -> bool:
    home = archify_home()
    src = DIAGRAMS / f"{name}.json"
    spec = json.loads(src.read_text(encoding="utf-8"))
    kind = spec["diagram_type"]
    quality = spec.get("meta", {}).get("quality_profile", "showcase")
    html = DIAGRAMS / f"{name}.html"
    expected_output = f"diagrams/{name}.html"
    if spec.get("meta", {}).get("output") != expected_output:
        print(f"✗ {name}: meta.output must be {expected_output!r}")
        return False

    out_dir = RECEIPTS / name
    shutil.rmtree(out_dir, ignore_errors=True)  # archify will not overwrite old evidence
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run([
        "node", str(home / "bin" / "archify.mjs"), "finalize", kind, str(src.relative_to(ROOT).as_posix()),
        expected_output, "--quality", quality, "--json", "--out-dir", str(out_dir.relative_to(ROOT).as_posix()),
    ])
    if result.returncode != 0:
        print(f"✗ {name}: archify finalize failed\n{result.stdout[-4000:]}\n{result.stderr[-2000:]}")
        return False

    svg = DIAGRAMS / f"{name}.svg"
    svg.unlink(missing_ok=True)  # export-svg refuses to overwrite
    result = run(["node", str(home / "bin" / "export-svg.mjs"), str(html), str(svg), theme])
    if result.returncode != 0:
        print(f"✗ {name}: svg export failed\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}")
        return False

    svg_to_png(svg, DIAGRAMS / f"{name}.png", width)
    print(f"✓ {name}: {html.name}, {svg.name}, {name}.png")
    return True


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--width", type=int, default=1400, help="PNG width in CSS pixels (rendered at 2x)")
    parser.add_argument("--theme", default="light", choices=["auto", "light", "dark"])
    args = parser.parse_args()
    names = sorted(p.stem for p in DIAGRAMS.glob("*.json")) if args.all else args.names
    if not names:
        parser.error("give diagram names or --all")
    failed = [n for n in names if not render(n, args.width, args.theme)]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
