"""Scientific Watch Agent — CLI entry point.

Runs the full 7-agent pipeline and exports the result as a PDF report.
Both structured (6-field cards) and narrative (prose) summary modes are supported.

Usage:
    python run_pipeline.py "graph neural networks"
    python run_pipeline.py "fake news detection" --top-n 15 --from-year 2018
    python run_pipeline.py "federated learning" --narrative
    python run_pipeline.py "federated learning" --narrative --output my_report.pdf
    python run_pipeline.py "federated learning" --email you@example.com

The PDF is saved to output/<slug>_<timestamp>.pdf by default.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_pipeline")


# ── helpers ────────────────────────────────────────────────────────────────────

def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:60]


def _fmt_cost(state: dict) -> str:
    total = sum(getattr(log, "cost_usd", 0) or 0 for log in state.get("logs", []))
    return f"${total:.4f}"


def _fmt_tokens(state: dict) -> str:
    total = sum(getattr(log, "tokens_used", 0) or 0 for log in state.get("logs", []))
    return f"{total:,}"


def _print_summary(state: dict, elapsed: float) -> None:
    synthesis = state.get("synthesis")
    trends = state.get("trend_analysis")

    print("\n" + "=" * 60)
    print("SCIENTIFIC WATCH REPORT")
    print("=" * 60)

    if synthesis:
        print(f"\n{synthesis.overview or ''}\n")

    if trends:
        if trends.trends:
            print("EMERGING TRENDS")
            for t in trends.trends[:3]:
                print(f"  · {getattr(t, 'name', t)}")
        if trends.gaps:
            print("\nRESEARCH GAPS")
            for g in trends.gaps[:3]:
                print(f"  · {getattr(g, 'name', g)}")
        if trends.future_perspectives:
            print("\nFUTURE DIRECTIONS")
            for p in trends.future_perspectives[:3]:
                print(f"  · {p}")

    print(f"\n{'─' * 60}")
    print(
        f"Papers: {len(state.get('raw_articles', []))} fetched → "
        f"{len(state.get('top_articles', []))} selected  |  "
        f"Reflexion: {state.get('synthesis_iteration', 0)} iter  |  "
        f"Cost: {_fmt_cost(state)}  |  "
        f"Time: {elapsed:.1f}s"
    )
    errors = state.get("errors", [])
    if errors:
        print(f"Warnings: {len(errors)} (see report for details)")
    print()


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scientific Watch Agent — automated literature monitoring"
    )
    parser.add_argument("topic", help="Research topic (e.g. 'graph neural networks')")
    parser.add_argument("--n-raw",    type=int, default=30,   help="Papers to fetch (default: 30)")
    parser.add_argument("--top-n",   type=int, default=10,   help="Papers to keep after scoring (default: 10)")
    parser.add_argument("--from-year", type=int, default=None, help="Earliest publication year")
    parser.add_argument("--narrative", action="store_true",
                        help="Prose paragraph summaries instead of structured 6-field cards")
    parser.add_argument("--output", type=str, default=None,
                        help="Output PDF path (default: output/<slug>_<timestamp>.pdf)")
    parser.add_argument("--email", type=str, default=None, metavar="ADDRESS",
                        help="Send the PDF to this email address after generation")
    parser.add_argument("--no-save", action="store_true",
                        help="Print terminal summary only, do not save PDF")
    args = parser.parse_args()

    # Import here so errors surface cleanly before the pipeline starts
    try:
        from src.agents.graph import run_pipeline
        from src.config import DEFAULT_FROM_YEAR
    except ImportError as e:
        print(f"[ERROR] Cannot import pipeline: {e}")
        print("Make sure you're in the project root and have run: pip install -r requirements.txt")
        sys.exit(1)

    from_year = args.from_year or DEFAULT_FROM_YEAR
    mode = "narrative" if args.narrative else "structured"

    print(f"\nTopic       : {args.topic}")
    print(f"Papers      : fetch {args.n_raw} → keep top {args.top_n}")
    print(f"From year   : {from_year}")
    print(f"Mode        : {mode} summaries")
    print(f"Running pipeline...\n")

    t0 = time.time()
    state = run_pipeline(
        args.topic,
        n_raw=args.n_raw,
        top_n=args.top_n,
        from_year=from_year,
        narrative_mode=args.narrative,
    )
    elapsed = time.time() - t0

    _print_summary(state, elapsed)

    if not args.no_save:
        # Resolve output path
        if args.output:
            out_path = args.output
        else:
            Path("output").mkdir(exist_ok=True)
            slug = _slug(args.topic)
            ts = datetime.now().strftime("%Y%m%d_%H%M")
            out_path = f"output/{slug}_{ts}.pdf"

        # Generate PDF via export_to_pdf
        try:
            from export_to_pdf import export_to_pdf, send_report_by_email
        except ImportError as e:
            print(f"[ERROR] Cannot import export_to_pdf: {e}")
            print("Make sure reportlab is installed: pip install reportlab")
            sys.exit(1)

        print("Generating PDF report...")
        pdf_path = export_to_pdf(state, output_path=out_path)
        print(f"Report saved → {pdf_path}")

        # Optional email delivery
        recipient = args.email
        if recipient:
            print(f"Sending report to {recipient}...")
            try:
                send_report_by_email(
                    pdf_path=pdf_path,
                    topic=args.topic,
                    recipient=recipient,
                    final=state,
                )
                print(f"Report sent to {recipient}")
            except Exception as exc:
                print(f"[Email failed] {exc}")


if __name__ == "__main__":
    main()
