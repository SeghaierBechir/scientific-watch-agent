"""Quality scorer - orchestrates the 4 feature scores into a final QualityScore.

This is the V1 manual scorer. In Phase 3, AutoML will learn optimal weights
via Optuna, but the API will stay the same.

Public functions:
    - score_article(article, topic, weights) -> QualityScore
    - score_articles(articles, topic, weights) -> list[QualityScore]
    - filter_top_n(articles, scores, n) -> list[Article]
"""

from __future__ import annotations

import logging
from typing import Optional

from src.config import DEFAULT_TOP_N, DEFAULT_WEIGHTS, USE_SEMANTIC_RELEVANCE
from src.features.authors_score import authors_score
from src.features.impact_score import impact_score
from src.features.recency_score import recency_score
from src.features.relevance_score import relevance_score as relevance_score_v1
from src.features.velocity_score import velocity_score
from src.features.venue_score import venue_score
from src.schemas import Article, QualityScore

logger = logging.getLogger(__name__)

# Try to import semantic V2; fall back gracefully if not installed.
try:
    from src.features.relevance_score_v2 import relevance_score_v2
    _SEMANTIC_AVAILABLE = True
    logger.debug("[QualityScorer] Semantic relevance V2 available")
except ImportError:
    _SEMANTIC_AVAILABLE = False
    logger.warning(
        "[QualityScorer] sentence-transformers not installed — "
        "using keyword relevance V1. Run: pip install sentence-transformers"
    )


# ============================================================
# Weight validation
# ============================================================


# Required keys (Level 1-2 features). Optional keys (Level 3) default to 0.
_REQUIRED_KEYS = frozenset({"venue", "authors", "impact", "relevance"})
_OPTIONAL_KEYS = frozenset({"velocity", "recency"})
_ALL_KEYS = _REQUIRED_KEYS | _OPTIONAL_KEYS


def _validate_weights(weights: dict[str, float]) -> dict[str, float]:
    """Validate and normalize a weight dict to sum=1.0.

    Accepts both 4-key (Level 1-2, backward-compatible) and 6-key (Level 3)
    dictionaries.  Missing optional keys (velocity, recency) default to 0.0
    so that old weight files continue to work without modification.

    Args:
        weights: dict with at minimum keys 'venue', 'authors', 'impact',
            'relevance'.  Optionally also 'velocity' and 'recency'.

    Returns:
        Normalized 6-key weights summing to 1.0.

    Raises:
        ValueError: if a required key is missing, any weight is negative,
            an unknown key is present, or all weights are zero.
    """
    missing = _REQUIRED_KEYS - weights.keys()
    if missing:
        raise ValueError(f"Missing weight keys: {missing}")

    extra = weights.keys() - _ALL_KEYS
    if extra:
        raise ValueError(f"Unknown weight keys: {extra}")

    if any(w < 0 for w in weights.values()):
        raise ValueError(f"Weights must be non-negative, got: {weights}")

    # Fill missing optional keys with 0.0 (backward-compatible defaults)
    full: dict[str, float] = {k: weights.get(k, 0.0) for k in _ALL_KEYS}

    total = sum(full.values())
    if total == 0:
        raise ValueError("All weights are zero - cannot normalize")

    return {k: full[k] / total for k in _ALL_KEYS}


# ============================================================
# Main scoring functions
# ============================================================


def score_article(
    article: Article,
    topic: str,
    weights: Optional[dict[str, float]] = None,
    current_year: Optional[int] = None,
    use_semantic: Optional[bool] = None,
) -> QualityScore:
    """Compute the full QualityScore for one article.

    Args:
        article: the article to score.
        topic: user's topic (used by relevance_score).
        weights: optional override. Defaults to DEFAULT_WEIGHTS.
        current_year: optional override (useful in tests).
        use_semantic: if True, use V2 semantic relevance (sentence-transformers).
            If False, force V1 keyword scoring.
            If None (default), read from config.USE_SEMANTIC_RELEVANCE.
            Falls back to V1 automatically if sentence-transformers is absent.

    Returns:
        A validated QualityScore object.
    """
    weights = _validate_weights(weights or DEFAULT_WEIGHTS)

    # Resolve semantic flag: config default → caller override → availability check
    _semantic = USE_SEMANTIC_RELEVANCE if use_semantic is None else use_semantic
    _semantic = _semantic and _SEMANTIC_AVAILABLE

    # ── Level 1-2 sub-scores ────────────────────────────────────────────────
    v = venue_score(article)
    a = authors_score(article)
    i = impact_score(article, current_year=current_year)

    # ── Level 3 sub-scores ──────────────────────────────────────────────────
    vel = velocity_score(article, current_year=current_year)
    rec = recency_score(article, current_year=current_year)

    # ── Relevance (V2 semantic with V1 fallback) ─────────────────────────────
    if _semantic:
        try:
            r = relevance_score_v2(article, topic)
        except Exception as exc:
            logger.warning(
                "[QualityScorer] Semantic relevance failed for '%s', falling back to V1: %s",
                article.id, exc,
            )
            r = relevance_score_v1(article, topic)
    else:
        r = relevance_score_v1(article, topic)

    # ── Weighted combination (6 features) ────────────────────────────────────
    final = (
        weights["venue"]    * v
        + weights["authors"]  * a
        + weights["impact"]   * i
        + weights["velocity"] * vel
        + weights["recency"]  * rec
        + weights["relevance"] * r
    )

    return QualityScore(
        article_id=article.id,
        venue_score=v,
        authors_score=a,
        impact_score=i,
        velocity_score=vel,
        recency_score=rec,
        relevance_score=r,
        final_score=final,
        weights_used=weights,
    )


def score_articles(
    articles: list[Article],
    topic: str,
    weights: Optional[dict[str, float]] = None,
    current_year: Optional[int] = None,
    use_semantic: Optional[bool] = None,
) -> list[QualityScore]:
    """Score a list of articles (convenience wrapper).

    The topic embedding is computed once and reused for every article
    when semantic scoring is enabled (LRU cache in relevance_score_v2).

    Args:
        articles: list of Articles to score.
        topic: user's topic.
        weights: optional weights override.
        current_year: optional year override.
        use_semantic: propagated to score_article (None = use config default).

    Returns:
        List of QualityScore in the same order as input articles.
    """
    logger.debug(
        "Scoring %d articles for topic '%s' with weights %s (semantic=%s)",
        len(articles), topic, weights or DEFAULT_WEIGHTS, use_semantic,
    )
    return [
        score_article(art, topic, weights, current_year=current_year, use_semantic=use_semantic)
        for art in articles
    ]


# ============================================================
# Filtering
# ============================================================


def filter_top_n(
    articles: list[Article],
    scores: list[QualityScore],
    n: int = DEFAULT_TOP_N,
) -> tuple[list[Article], list[QualityScore]]:
    """Keep only the top-N articles (and their scores), sorted by final_score desc.

    Args:
        articles: input articles.
        scores: corresponding scores (must be same length and order).
        n: how many to keep.

    Returns:
        Tuple (top_n_articles, top_n_scores), both sorted by final_score desc.

    Raises:
        ValueError: if articles and scores have different lengths.
    """
    if len(articles) != len(scores):
        raise ValueError(
            f"articles ({len(articles)}) and scores ({len(scores)}) "
            f"must have the same length"
        )

    # Pair articles with scores, sort by final_score desc
    paired = sorted(
        zip(articles, scores),
        key=lambda pair: pair[1].final_score,
        reverse=True,
    )

    top = paired[:n]
    if not top:
        return [], []

    top_articles, top_scores = zip(*top)
    return list(top_articles), list(top_scores)
