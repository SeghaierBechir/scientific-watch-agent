"""show_topology.py — Generate and open a PNG topology diagram of the agent pipeline.

Uses LangGraph's built-in Mermaid renderer (calls mermaid.ink public API,
no extra dependencies required).

Usage:
    python show_topology.py              # saves + opens data/topology.png
    python show_topology.py --no-open    # saves only, do not open viewer
    python show_topology.py --out my.png # custom output path
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("show_topology")

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_OUT = PROJECT_ROOT / "data" / "topology.png"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate agent topology PNG")
    p.add_argument(
        "--out", type=Path, default=DEFAULT_OUT,
        help=f"Output PNG path (default: {DEFAULT_OUT})",
    )
    p.add_argument(
        "--no-open", action="store_true",
        help="Do not open the PNG after saving",
    )
    return p.parse_args()


def _open_file(path: Path) -> None:
    """Open a file with the system default viewer (Windows / macOS / Linux)."""
    try:
        if sys.platform == "win32":
            subprocess.Popen(["start", "", str(path)], shell=True)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except Exception as exc:
        logger.warning("Could not open file automatically: %s", exc)
        print(f"  → Open manually: {path}")


def main() -> None:
    args = _parse_args()

    print()
    print("=" * 58)
    print("  Scientific Watch Agent — Pipeline Topology")
    print("=" * 58)

    # ── Build graph ───────────────────────────────────────────
    print("\n[1/3] Building LangGraph pipeline...")
    try:
        from src.agents.graph import build_graph
        graph = build_graph()
        print("      Graph compiled successfully.")
    except Exception as exc:
        print(f"\n[ERROR] Could not build graph: {exc}")
        sys.exit(1)

    # ── Render PNG via Mermaid.ink API ────────────────────────
    print("[2/3] Rendering PNG via Mermaid.ink API...")
    try:
        from langchain_core.runnables.graph import MermaidDrawMethod  # type: ignore
        png_bytes: bytes = graph.get_graph().draw_mermaid_png(
            draw_method=MermaidDrawMethod.API,
            background_color="white",
            padding=20,
        )
        print(f"      Rendered: {len(png_bytes):,} bytes")
    except Exception as exc:
        print(f"\n[ERROR] Rendering failed: {exc}")
        print("  Possible cause: no internet connection (Mermaid.ink is a public API)")
        print("  Fallback: printing Mermaid source instead...\n")
        try:
            mermaid_src = graph.get_graph().draw_mermaid()
            print(mermaid_src)
        except Exception:
            pass
        sys.exit(1)

    # ── Save ──────────────────────────────────────────────────
    print(f"[3/3] Saving to {args.out}...")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(png_bytes)
    print(f"      Saved: {args.out}")

    if not args.no_open:
        print("\n  Opening in default viewer...")
        _open_file(args.out)

    print()
    print("=" * 58)
    print(f"  Done → {args.out}")
    print("=" * 58)
    print()


if __name__ == "__main__":
    main()
