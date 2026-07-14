"""End-to-end demo of Phase 2: search, score, and filter to top-N.

This demo simulates the full pipeline up through quality filtering.
The LLM steps (summarize, synthesize) come in Phase 4.

Run with:
    python demo_phase2.py "fake news detection"
    python demo_phase2.py "fake news detection" 30 10   # 30 raw -> top 10
"""

from __future__ import annotations

import logging
import sys

from src.scoring.quality_scorer import (
    filter_top_n,
    score_articles,
)
from src.sources.openalex import OpenAlexClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def main(topic: str = "fake news detection",
         n_raw: int = 30,
         top_n: int = 10):
    print(f"\n{'='*70}")
    print(f"Phase 2 demo: end-to-end search + score + filter")
    print(f"  Topic:    '{topic}'")
    print(f"  Fetching: {n_raw} raw articles, keeping top {top_n}")
    print(f"{'='*70}\n")

    # === Step 1: Fetch from OpenAlex ===
    client = OpenAlexClient()
    raw_articles = client.search(
        query=topic,
        n_results=n_raw,
        from_year=2020,
        require_abstract=True,
    )
    print(f"\nGot {len(raw_articles)} articles with abstracts.\n")

    if not raw_articles:
        print("No articles retrieved - cannot proceed.")
        return

    # === Step 2: Score them all ===
    print(f"Scoring all articles using default weights...")
    scores = score_articles(raw_articles, topic)

    # === Step 3: Filter to top-N ===
    top_articles, top_scores = filter_top_n(raw_articles, scores, n=top_n)
    print(f"Filtered to top {len(top_articles)} articles.\n")

    # === Step 4: Pretty-print ===
    print(f"{'='*70}")
    print(f"TOP {len(top_articles)} ARTICLES (sorted by quality score)")
    print(f"{'='*70}\n")

    for rank, (art, sc) in enumerate(zip(top_articles, top_scores), 1):
        venue = art.journal_name or "[no venue]"
        if art.is_preprint:
            venue += " (preprint)"

        print(f"#{rank} [score={sc.final_score:.3f}] {art.title}")
        print(f"     {venue} - {art.year} - cited {art.citation_count}x")
        print(f"     Sub-scores: venue={sc.venue_score:.2f}  "
              f"authors={sc.authors_score:.2f}  "
              f"impact={sc.impact_score:.2f}  "
              f"relevance={sc.relevance_score:.2f}")
        if art.doi:
            print(f"     DOI: {art.doi}")
        print()

    # === Step 5: Diagnostics ===
    print(f"{'='*70}")
    print("Score diagnostics across the kept articles:")

    def stats(values: list[float]) -> str:
        if not values:
            return "n/a"
        return (f"min={min(values):.2f}  "
                f"avg={sum(values) / len(values):.2f}  "
                f"max={max(values):.2f}")

    print(f"  Final:     {stats([s.final_score for s in top_scores])}")
    print(f"  Venue:     {stats([s.venue_score for s in top_scores])}")
    print(f"  Authors:   {stats([s.authors_score for s in top_scores])}")
    print(f"  Impact:    {stats([s.impact_score for s in top_scores])}")
    print(f"  Relevance: {stats([s.relevance_score for s in top_scores])}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    topic_arg = sys.argv[1] if len(sys.argv) > 1 else "fake news detection"
    n_raw_arg = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    top_n_arg = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    main(topic_arg, n_raw_arg, top_n_arg)
