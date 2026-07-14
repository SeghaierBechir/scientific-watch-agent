"""demo_groq.py — Test the full Scientific Watch pipeline with Groq Llama.

Runs the complete 7-agent pipeline (QueryExpander → Searcher → QualityCritic
→ Summarizer → Synthesizer → Critic → TrendAnalyst) using Groq for all
LLM-powered agents, then prints a structured console report.

Why Groq?
    - Ultra-fast inference (~10x faster than OpenAI)
    - Very cheap ($0.05/$0.08 per 1M tokens for Llama 8B)
    - Good for testing and benchmarking

Usage:
    python demo_groq.py "fake news detection"
    python demo_groq.py "medical image segmentation" --model llama-3.3-70b-versatile
    python demo_groq.py "quantum computing" --n-raw 20 --top-n 5
    python demo_groq.py "NLP low resource" --no-reflexion

Available Groq models:
    llama-3.1-8b-instant     (default — fastest, cheapest)
    llama-3.3-70b-versatile  (best quality, still cheap)
    llama3-8b-8192           (8K context window)
    mixtral-8x7b-32768       (32K context, good for synthesis)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("demo_groq")

# Suppress noisy sub-loggers during demo
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


# ============================================================
# CLI
# ============================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the Scientific Watch pipeline with Groq Llama"
    )
    p.add_argument(
        "topic",
        help='Research topic to investigate (e.g. "fake news detection")',
    )
    p.add_argument(
        "--model", "-m",
        default="llama-3.1-8b-instant",
        help="Groq model ID (default: llama-3.1-8b-instant)",
    )
    p.add_argument(
        "--n-raw", type=int, default=20,
        help="Articles to fetch from OpenAlex (default: 20)",
    )
    p.add_argument(
        "--top-n", type=int, default=5,
        help="Articles to keep after quality filtering (default: 5)",
    )
    p.add_argument(
        "--no-reflexion", action="store_true",
        help="Disable Reflexion loop (single-shot synthesis)",
    )
    p.add_argument(
        "--from-year", type=int, default=2022,
        help="Earliest publication year (default: 2022)",
    )
    return p.parse_args()


# ============================================================
# Display helpers
# ============================================================

def _divider(char: str = "─", width: int = 64) -> None:
    print(char * width)


def _header(title: str) -> None:
    print()
    _divider("═")
    print(f"  {title}")
    _divider("═")


def _section(title: str) -> None:
    print()
    _divider()
    print(f"  {title}")
    _divider()


def _cost_summary(logs: list) -> None:
    if not logs:
        return
    total_cost = sum(getattr(log, "cost_usd", 0.0) or 0.0 for log in logs)
    total_tokens = sum(
        (getattr(log, "input_tokens", 0) or 0) + (getattr(log, "output_tokens", 0) or 0)
        for log in logs
    )
    print(f"  Total tokens : {total_tokens:,}")
    print(f"  Total cost   : ${total_cost:.4f}")


# ============================================================
# Groq-patched pipeline
# ============================================================

def _build_groq_overrides(model: str) -> dict:
    """Return a TASK_MODELS-compatible dict with all tasks mapped to Groq."""
    return {
        "query_expansion": ("groq", model),
        "summarize":       ("groq", model),
        "synthesize":      ("groq", model),
        "trend_analysis":  ("groq", model),
        "critic":          ("groq", model),
        "judge":           ("groq", model),
    }


def run_with_groq(
    topic: str,
    model: str,
    n_raw: int,
    top_n: int,
    no_reflexion: bool,
    from_year: int,
) -> dict:
    """Patch TASK_MODELS with Groq, run the pipeline, restore TASK_MODELS."""
    import src.config as cfg
    from src.agents.graph import run_pipeline

    original = dict(cfg.TASK_MODELS)

    try:
        # Temporarily override all tasks to use Groq
        cfg.TASK_MODELS.clear()
        cfg.TASK_MODELS.update(_build_groq_overrides(model))

        if no_reflexion:
            original_max = cfg.MAX_REFLEXION_ITERATIONS
            cfg.MAX_REFLEXION_ITERATIONS = 0

        final = run_pipeline(
            topic=topic,
            n_raw=n_raw,
            top_n=top_n,
            from_year=from_year,
        )

    finally:
        cfg.TASK_MODELS.clear()
        cfg.TASK_MODELS.update(original)
        if no_reflexion:
            cfg.MAX_REFLEXION_ITERATIONS = original_max  # type: ignore[possibly-undefined]

    return final


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = _parse_args()

    _header(f"Scientific Watch Agent — Groq/{args.model}")
    print(f"  Topic      : {args.topic}")
    print(f"  Model      : {args.model}")
    print(f"  Articles   : fetch {args.n_raw} → keep top {args.top_n}")
    print(f"  Reflexion  : {'OFF' if args.no_reflexion else 'ON (max 3 iterations)'}")
    print(f"  From year  : {args.from_year}")

    # ── Sanity-check: Groq API key present ───────────────────
    from src.config import GROQ_API_KEY
    if not GROQ_API_KEY:
        print("\n[ERROR] GROQ_API_KEY not found in .env")
        print("  1. Go to https://console.groq.com → API Keys → Create key")
        print("  2. Add GROQ_API_KEY=gsk_... to your .env file")
        sys.exit(1)
    print(f"  Groq key   : {GROQ_API_KEY[:8]}***")

    # ── Run pipeline ─────────────────────────────────────────
    _section("Running pipeline...")
    t0 = time.time()

    try:
        final = run_with_groq(
            topic=args.topic,
            model=args.model,
            n_raw=args.n_raw,
            top_n=args.top_n,
            no_reflexion=args.no_reflexion,
            from_year=args.from_year,
        )
    except Exception as exc:
        print(f"\n[ERROR] Pipeline failed: {exc}")
        logger.exception("Pipeline error")
        sys.exit(1)

    duration = time.time() - t0

    # ── 1. Query expansion ────────────────────────────────────
    _section("1. Query Expansion (ReAct)")
    queries = final.get("expanded_queries") or []
    if queries:
        for i, q in enumerate(queries, 1):
            print(f"  [{i}] {q}")
    else:
        print("  (no queries generated)")

    # ── 2. Top articles ───────────────────────────────────────
    _section("2. Top Articles (after quality scoring)")
    articles = final.get("top_articles") or []
    scores   = final.get("top_scores") or []
    score_map = {s.article_id: s for s in scores} if scores else {}

    if articles:
        for art in articles:
            sc = score_map.get(art.id)
            score_str = f"  score={sc.final_score:.2f}" if sc else ""
            year = getattr(art, "publication_year", "?")
            cit  = getattr(art, "cited_by_count", 0)
            print(f"  • [{year}] {art.title[:70]}...")
            print(f"    citations={cit}{score_str}")
    else:
        print("  (no articles found)")

    # ── 3. Summaries ──────────────────────────────────────────
    _section("3. Article Summaries (Groq Llama)")
    summaries = final.get("summaries") or []
    if summaries:
        for i, s in enumerate(summaries, 1):
            print(f"\n  [{i}] {s.article_id}")
            print(f"    Problem : {(s.problem or 'n/a')[:120]}")
            print(f"    Method  : {(s.method  or 'n/a')[:120]}")
            print(f"    Results : {(s.results or 'n/a')[:120]}")
            if s.key_contributions:
                print(f"    Key contrib: {s.key_contributions[0][:100]}")
    else:
        print("  (no summaries generated)")

    # ── 4. Reflexion history ──────────────────────────────────
    feedbacks = final.get("critic_feedbacks") or []
    n_iter    = final.get("synthesis_iteration", 0)
    if feedbacks and not args.no_reflexion:
        _section(f"4. Reflexion Loop ({n_iter} iteration(s))")
        for i, fb in enumerate(feedbacks, 1):
            needs = "→ revised" if fb.needs_revision else "→ accepted"
            print(f"  Iter {i}: quality={fb.overall_quality}  {needs}")
            if fb.strengths:
                print(f"    + {fb.strengths[0][:100]}")
            if fb.weaknesses:
                print(f"    - {fb.weaknesses[0][:100]}")

    # ── 5. Synthesis ──────────────────────────────────────────
    _section("5. Global Synthesis")
    synth = final.get("synthesis")
    if synth:
        print(f"  Overview:\n  {(synth.overview or '')[:400]}")
        if synth.dominant_approaches:
            print(f"\n  Dominant approaches:")
            for a in synth.dominant_approaches[:3]:
                print(f"    • {a[:100]}")
        if synth.key_findings:
            print(f"\n  Key findings:")
            for f in synth.key_findings[:3]:
                print(f"    • {f[:100]}")
    else:
        print("  (synthesis not generated)")

    # ── 6. Trend analysis ────────────────────────────────────
    _section("6. Trends, Gaps & Perspectives")
    ta = final.get("trend_analysis")
    if ta:
        if ta.trends:
            print("  Trends:")
            for tr in ta.trends[:4]:
                tag = f"[{tr.maturity.upper()}]" if tr.maturity else ""
                print(f"    {tag} {tr.name}")
                if tr.description:
                    print(f"        {tr.description[:120]}")
        if ta.research_gaps:
            print("\n  Research gaps:")
            for g in ta.research_gaps[:3]:
                print(f"    • {g[:120]}")
        if ta.future_perspectives:
            print("\n  Future perspectives:")
            for fp in ta.future_perspectives[:3]:
                print(f"    • {fp[:120]}")
    else:
        print("  (trend analysis not generated)")

    # ── 7. Summary stats ─────────────────────────────────────
    _section("Run Summary")
    errors = final.get("errors") or []
    logs   = final.get("logs") or []
    print(f"  Duration   : {duration:.1f}s")
    print(f"  Articles   : {len(articles)} selected / {args.n_raw} fetched")
    print(f"  Summaries  : {len(summaries)}")
    print(f"  Reflexion  : {n_iter} iteration(s)")
    print(f"  Errors     : {len(errors)}")
    _cost_summary(logs)

    if errors:
        print("\n  Errors encountered:")
        for e in errors[:5]:
            print(f"    ✗ {e[:120]}")

    print()
    _divider("═")
    print(f"  Done in {duration:.1f}s  |  model: {args.model}")
    _divider("═")
    print()


if __name__ == "__main__":
    main()
