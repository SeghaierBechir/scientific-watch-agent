"""Recency score — Level 3 feature.

Rewards recently published articles independently of their citation count.
A 2024 paper with zero citations still gets a high recency score; a 2018
paper with many citations gets a low recency score.

Why add recency separately from impact/velocity?
    impact and velocity both depend on citation_count — they are undefined
    (or penalized) for brand-new papers.  Recency is a *citation-free* signal
    that captures "this is what the field is working on RIGHT NOW", which
    matters especially for fast-moving domains like LLMs or fake news.

Mathematical model — exponential decay with configurable half-life:

    recency(age) = 2^(-age / half_life)
                 = exp(-ln(2) * age / half_life)

    age=0  → 1.000  (published this year)
    age=HL → 0.500  (half-life years old)
    age=2HL→ 0.250
    age=3HL→ 0.125

Default half-life = 3 years: a paper from 3 years ago scores 0.50.
This is appropriate for fast-moving CS/AI fields.  For slower-moving
fields (biology, materials), Optuna will learn a lower w_recency weight
anyway, so the absolute value of the half-life matters less than the
relative ordering it creates.

Domain relevance for meta-learning:
    Fast-moving (LLMs, fake news): high w_recency
    Established (quantum computing): low w_recency
"""

from __future__ import annotations

import math
from datetime import datetime

from src.schemas import Article

# ============================================================
# Calibration constants
# ============================================================

# Half-life in years: age at which recency score = 0.5.
# 3 years is appropriate for CS/AI: a 3-year-old paper is "half as recent"
# as a current one in most fast-moving sub-fields.
_RECENCY_HALF_LIFE = 3.0

# Pre-computed ln(2) / half_life for efficiency (avoids recomputing per call).
_DECAY_RATE = math.log(2) / _RECENCY_HALF_LIFE


def recency_score(article: Article, current_year: int | None = None) -> float:
    """Compute publication recency for *article* — exponential decay, [0, 1].

    Args:
        article: the article to score.
        current_year: override for the current year (useful in tests).
            Defaults to datetime.now().year.

    Returns:
        Float in (0, 1].
        - 1.000: published this year (age = 0)
        - 0.794: 1 year old
        - 0.500: 3 years old  (half-life)
        - 0.250: 6 years old
        - 0.125: 9 years old
        - Never zero (exponential decay asymptotes toward 0)

    Algorithm:
        age   = max(0, current_year - article.year)
        score = exp(-ln(2) / HALF_LIFE * age)
              = 2^(-age / HALF_LIFE)

    Notes:
        Unlike velocity_score and impact_score, recency does not depend on
        citation_count at all.  It is a pure temporal signal.

    Examples:
        >>> from src.schemas import Article, SourceType
        >>> a = Article(id='x', title='t', year=2021,
        ...             source=SourceType.OPENALEX, url='u')
        >>> # age = 3, score = 2^(-3/3) = 2^(-1) = 0.5
        >>> recency_score(a, current_year=2024)
        0.5
    """
    year_now = current_year if current_year is not None else datetime.now().year
    age = max(0, year_now - article.year)
    return math.exp(-_DECAY_RATE * age)
