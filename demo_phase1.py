"""End-to-end demo of Phase 1: search OpenAlex for a topic and print results.

Run with:
    python demo_phase1.py "fake news detection"
or just:
    python demo_phase1.py
"""

from __future__ import annotations

import logging
import sys

from src.sources.openalex import OpenAlexClient

# ============================================================
# Setup logging so we can see what's happening
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def main(topic: str = "fake news detection", n: int = 15):
    print(f"\n{'='*60}")
    print(f"Searching OpenAlex for: '{topic}' (top {n} results)")
    print(f"{'='*60}\n")

    client = OpenAlexClient()
    articles = client.search(
        query=topic,
        n_results=n,
        from_year=2020,  # last 6 years
        require_abstract=True,
    )

    print(f"\nGot {len(articles)} articles with abstracts.\n")

    # Pretty-print a summary of each
    for i, art in enumerate(articles, 1):
        authors_str = ", ".join(a.name for a in art.authors[:3])
        if len(art.authors) > 3:
            authors_str += f" et al. ({len(art.authors)} authors total)"
        venue = art.journal_name or "[unknown venue]"
        if art.is_preprint:
            venue += " (preprint)"

        print(f"[{i}] {art.title}")
        print(f"    {authors_str}")
        print(f"    {venue} — {art.year} — cited {art.citation_count}×")
        if art.doi:
            print(f"    DOI: {art.doi}")
        if art.abstract:
            preview = art.abstract[:200] + ("..." if len(art.abstract) > 200 else "")
            print(f"    Abstract: {preview}")
        print()

    # Quick stats
    print(f"{'='*60}")
    print("Quick stats:")
    print(f"  Articles with DOI: {sum(1 for a in articles if a.doi)}/{len(articles)}")
    print(f"  Open access:       {sum(1 for a in articles if a.open_access)}/{len(articles)}")
    print(f"  Preprints:         {sum(1 for a in articles if a.is_preprint)}/{len(articles)}")
    if articles:
        avg_cites = sum(a.citation_count for a in articles) / len(articles)
        print(f"  Avg citations:     {avg_cites:.1f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    topic_arg = sys.argv[1] if len(sys.argv) > 1 else "fake news detection"
    n_arg = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    main(topic_arg, n_arg)
