"""Venue credibility score.

Computes a [0, 1] score reflecting the prestige of the journal/conference
where an article was published.

V1 strategy (lightweight, no external CSV needed yet):
    - Use the article's `quartile` if OpenAlex provided it
    - Fall back to a baseline if data is missing
    - Preprints get a neutral 0.5 (we don't penalize them)

V2 strategy (Phase 3):
    - Look up SJR percentile from a downloaded Scimago CSV
    - Combine SJR percentile (50%) + quartile mapping (50%)

We design the API so the V2 upgrade is a drop-in: the function signature stays
the same, only the internals get richer.
"""

from __future__ import annotations

from src.schemas import Article

# ============================================================
# Quartile -> score mapping
# These values are the standard ones used in scientometrics literature.
# Q1 = top 25%, Q2 = 25-50%, Q3 = 50-75%, Q4 = bottom 25%.
# ============================================================

_QUARTILE_MAP: dict[str, float] = {
    "Q1": 1.00,
    "Q2": 0.70,
    "Q3": 0.40,
    "Q4": 0.20,
}

_BASELINE_UNKNOWN = 0.50  # journal with no quartile info
_PREPRINT_SCORE = 0.50    # neutral for preprints (no peer review yet)
_BASELINE_NO_VENUE = 0.30  # article without any venue info at all


def venue_score(article: Article) -> float:
    """Compute the venue credibility score for an article.

    Args:
        article: the article to score.

    Returns:
        A float in [0, 1] where 1.0 means top-tier venue.

    Examples:
        >>> from src.schemas import Article, SourceType
        >>> a = Article(id="x", title="t", year=2024, source=SourceType.OPENALEX,
        ...             url="u", quartile="Q1")
        >>> venue_score(a)
        1.0
    """
    # Preprint: neutral (we don't punish them, they may be high-quality)
    if article.is_preprint:
        return _PREPRINT_SCORE

    # No journal info at all (shouldn't happen for published articles)
    if not article.journal_name:
        return _BASELINE_NO_VENUE

    # Quartile-based scoring
    if article.quartile and article.quartile in _QUARTILE_MAP:
        return _QUARTILE_MAP[article.quartile]

    # Journal exists but no quartile - middle baseline
    return _BASELINE_UNKNOWN
