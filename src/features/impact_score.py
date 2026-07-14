"""Article impact score.

Computes a [0, 1] score reflecting the citation impact of an article.

Strategy:
    - Use citations-per-year (NOT raw citations) to normalize by age
    - A 2010 paper with 100 cites (10/year) scores LOWER than
      a 2020 paper with 100 cites (25/year)
    - Apply log-normalization so very-high-citation papers don't dominate
    - Articles published this year get special treatment (no penalty for
      having no citations yet)

Why citations/year? Otherwise old papers always win, and we'd never surface
recent breakthroughs. This is the standard normalization in scientometrics.
"""

from __future__ import annotations

import math
from datetime import datetime

from src.schemas import Article

# ============================================================
# Calibration constants
# ============================================================

# Citations-per-year value at which the score saturates to 1.0
# 50 cites/year is "very high impact" in most fields
_CITES_PER_YEAR_SATURATION = 50

# Minimum age in years (avoid division by zero for current-year papers)
_MIN_AGE_YEARS = 1

# Score for a paper too new to have citations yet (this year or last year)
# We give it a neutral middle score - the relevance feature will handle the rest
_BASELINE_TOO_NEW = 0.50


def impact_score(article: Article, current_year: int | None = None) -> float:
    """Compute the citation impact score for an article.

    Args:
        article: the article to score.
        current_year: override for the current year (useful for testing).
            Defaults to today's year.

    Returns:
        A float in [0, 1].

    Examples:
        >>> from src.schemas import Article, SourceType
        >>> # Old paper, lots of citations: 200 cites / 10 years = 20/year
        >>> a = Article(id="x", title="t", year=2014,
        ...             source=SourceType.OPENALEX, url="u",
        ...             citation_count=200)
        >>> # log(21)/log(51) ~= 0.774
        >>> 0.76 < impact_score(a, current_year=2024) < 0.79
        True
    """
    year_now = current_year if current_year is not None else datetime.now().year

    # Compute paper age (minimum 1 to avoid division by zero)
    age = max(year_now - article.year + 1, _MIN_AGE_YEARS)

    # Special case: paper too new to have meaningful citations
    # Anything published this year or last year gets the neutral baseline
    if age <= 1 and article.citation_count == 0:
        return _BASELINE_TOO_NEW

    # Citations per year (the key normalization)
    cites_per_year = article.citation_count / age

    # Log normalization
    # cpy=0   -> 0.0
    # cpy=5   -> log(6)/log(51) ~= 0.456
    # cpy=20  -> log(21)/log(51) ~= 0.774
    # cpy=50  -> log(51)/log(51) = 1.000
    # cpy=200 -> capped at 1.0
    score = math.log(cites_per_year + 1) / math.log(_CITES_PER_YEAR_SATURATION + 1)
    return min(score, 1.0)
