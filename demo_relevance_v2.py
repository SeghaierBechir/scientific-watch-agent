"""Demo: compare V1 (keyword) vs V2 (semantic) relevance scoring.

Usage:
    python demo_relevance_v2.py
    python demo_relevance_v2.py "misinformation detection" --top 15

What this demonstrates:
    - V2 catches synonyms and related concepts that V1 misses
    - V2 ranks genuinely related papers higher (e.g., "computational propaganda"
      for topic "fake news detection")
    - Ranking differences between V1 and V2 are shown side by side
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

from src.features.relevance_score import relevance_score as v1_score
from src.features.relevance_score_v2 import relevance_score_v2 as v2_score
from src.schemas import Article, SourceType


# ============================================================
# Synthetic test articles (no network needed)
# ============================================================

def _make_articles(topic: str) -> list[Article]:
    """Build articles that illustrate V1 vs V2 differences."""
    return [
        Article(
            id="a1",
            title="Detecting fake news using BERT-based classifiers",
            abstract=(
                "We propose a transformer-based approach to fake news detection. "
                "Our BERT fine-tuned model achieves 94.1% accuracy on the FakeNewsNet "
                "benchmark, outperforming prior keyword and TF-IDF baselines by 8%."
            ),
            year=2023, source=SourceType.OPENALEX, url="u1",
            concepts=["fake news", "BERT", "text classification"],
        ),
        Article(
            id="a2",
            title="Misinformation spread on social media: a computational approach",
            abstract=(
                "This paper studies the propagation of misinformation on Twitter. "
                "We build a cascade model to predict which false stories go viral, "
                "achieving 87% F1 on a 50,000-tweet dataset."
            ),
            year=2023, source=SourceType.OPENALEX, url="u2",
            concepts=["misinformation", "social media", "virality"],
        ),
        Article(
            id="a3",
            title="Computational propaganda: bots and disinformation campaigns",
            abstract=(
                "We analyze automated accounts (bots) used to spread disinformation "
                "during the 2020 elections. Network analysis reveals coordinated "
                "inauthentic behavior affecting 12% of political tweets."
            ),
            year=2022, source=SourceType.OPENALEX, url="u3",
            concepts=["computational propaganda", "bots", "disinformation"],
        ),
        Article(
            id="a4",
            title="Fact-checking with neural evidence retrieval",
            abstract=(
                "Automated fact-checking requires retrieving relevant evidence. "
                "We introduce a bi-encoder model that retrieves supporting or "
                "refuting documents with 91% recall@5 on the FEVER benchmark."
            ),
            year=2024, source=SourceType.OPENALEX, url="u4",
            concepts=["fact-checking", "evidence retrieval", "NLP"],
        ),
        Article(
            id="a5",
            title="Attention mechanisms in neural machine translation",
            abstract=(
                "We revisit the attention mechanism for sequence-to-sequence models. "
                "Our multi-head attention variant improves BLEU score by 2.1 points "
                "on WMT14 En-De without additional parameters."
            ),
            year=2023, source=SourceType.OPENALEX, url="u5",
            concepts=["attention", "machine translation", "transformer"],
        ),
        Article(
            id="a6",
            title="Deep learning for medical image segmentation",
            abstract=(
                "Convolutional neural networks achieve state-of-the-art results on "
                "CT scan segmentation. Our U-Net variant reaches 0.92 Dice score on "
                "the BraTS 2023 brain tumor dataset."
            ),
            year=2023, source=SourceType.OPENALEX, url="u6",
            concepts=["medical imaging", "segmentation", "deep learning"],
        ),
        Article(
            id="a7",
            title="Credibility assessment of online news articles",
            abstract=(
                "We introduce a credibility scoring model that combines linguistic "
                "cues, source reputation, and social signals to rate news articles. "
                "Evaluated on 8,000 articles with 89% agreement with expert labels."
            ),
            year=2024, source=SourceType.OPENALEX, url="u7",
            concepts=["credibility", "online news", "misinformation"],
        ),
        Article(
            id="a8",
            title="Reinforcement learning from human feedback for LLMs",
            abstract=(
                "RLHF aligns large language models with human preferences. "
                "We scale reward model training to 175B parameters and show "
                "significant improvements in helpfulness and harmlessness."
            ),
            year=2023, source=SourceType.OPENALEX, url="u8",
            concepts=["RLHF", "alignment", "language models"],
        ),
        Article(
            id="a9",
            title="Cross-lingual fake news detection in low-resource languages",
            abstract=(
                "We transfer a fake news detector trained on English to Arabic and "
                "Hindi using multilingual embeddings. Zero-shot transfer achieves "
                "79% accuracy despite no target-language training data."
            ),
            year=2024, source=SourceType.OPENALEX, url="u9",
            concepts=["fake news", "cross-lingual", "low-resource NLP"],
        ),
        Article(
            id="a10",
            title="Graph neural networks for molecular property prediction",
            abstract=(
                "Graph-level representations via message-passing GNNs predict "
                "molecular properties with high accuracy. We achieve state-of-the-art "
                "results on 8 datasets from the MoleculeNet benchmark."
            ),
            year=2023, source=SourceType.OPENALEX, url="u10",
            concepts=["graph neural networks", "chemistry", "molecular biology"],
        ),
    ]


# ============================================================
# Comparison logic
# ============================================================

def _run_comparison(topic: str, top_n: int) -> None:
    articles = _make_articles(topic)

    print(f"\nTopic: '{topic}'")
    print("=" * 70)

    # --- V1 scores ---
    t0 = time.perf_counter()
    v1_scores = {a.id: v1_score(a, topic) for a in articles}
    t_v1 = (time.perf_counter() - t0) * 1000

    # --- V2 scores (first call loads the model) ---
    print("Loading sentence-transformers model (first call, ~80 MB) ...")
    t0 = time.perf_counter()
    v2_scores = {a.id: v2_score(a, topic) for a in articles}
    t_v2 = (time.perf_counter() - t0) * 1000

    # --- Rankings ---
    v1_ranked = sorted(articles, key=lambda a: v1_scores[a.id], reverse=True)
    v2_ranked = sorted(articles, key=lambda a: v2_scores[a.id], reverse=True)

    v1_rank = {a.id: i + 1 for i, a in enumerate(v1_ranked)}
    v2_rank = {a.id: i + 1 for i, a in enumerate(v2_ranked)}

    # --- Print table sorted by V2 rank ---
    header = f"{'#':>3}  {'Article (truncated)':40}  {'V1 kw':>8}  {'V2 sem':>8}  {'Drank':>6}"
    sep = "-" * len(header)
    print(f"\n{header}")
    print(sep)

    for art in v2_ranked[:top_n]:
        title_short = art.title[:38] + ".." if len(art.title) > 40 else art.title
        delta = v1_rank[art.id] - v2_rank[art.id]  # positive = V2 promoted
        arrow = f"(+{delta})" if delta > 0 else (f"({delta})" if delta < 0 else "  =")
        print(
            f"{v2_rank[art.id]:>3}.  {title_short:40}  "
            f"{v1_scores[art.id]:>8.3f}  {v2_scores[art.id]:>8.3f}  {arrow:>6}"
        )

    print(sep)
    print(f"\nTiming:  V1 keywords = {t_v1:.1f} ms   V2 semantic = {t_v2:.1f} ms")
    print(f"(V2 subsequent calls will be ~{t_v2 * 0.05:.1f} ms due to embedding cache)\n")

    # --- Key differences commentary ---
    print("Key differences (articles promoted or demoted by V2):")
    for art in articles:
        delta = v1_rank[art.id] - v2_rank[art.id]
        if abs(delta) >= 2:
            direction = "PROMOTED (+)" if delta > 0 else "DEMOTED  (-)"
            print(
                f"  {direction} rank {v1_rank[art.id]:>2} -> {v2_rank[art.id]:>2} | "
                f"V1={v1_scores[art.id]:.3f}  V2={v2_scores[art.id]:.3f} | "
                f"{art.title[:50]}"
            )


# ============================================================
# Entry point
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare relevance scoring V1 (keywords) vs V2 (semantic embeddings)"
    )
    parser.add_argument(
        "topic",
        nargs="?",
        default="fake news detection",
        help="Search topic (default: 'fake news detection')",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of articles to show (default: 10)",
    )
    args = parser.parse_args()

    try:
        _run_comparison(args.topic, args.top)
    except ImportError:
        print(
            "\n[ERROR] sentence-transformers is not installed.\n"
            "Run:  pip install sentence-transformers\n"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
