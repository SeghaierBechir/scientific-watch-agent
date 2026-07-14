"""Citation velocity score — Level 3 feature.

Measures how fast an article is accumulating citations relative to its age.
Differs from impact_score in two deliberate ways:

    impact_score  : log-normalized, saturation at 50 cites/year.
                    Rewards *established* impact (100+ cites).
    velocity_score: linear-normalized, saturation at 20 cites/year.
                    Rewards *momentum* — papers gaining traction quickly.

Why linear instead of log?
    Log compression makes high-velocity papers indistinguishable from
    moderate ones.  Linear keeps the spread: a paper at 5 cpy gets 0.25
    while one at 15 cpy gets 0.75 — a 3× difference that Optuna can use.

Domain relevance for meta-learning:
    Fast-moving fields (fake news, LLMs): Optuna learns high w_velocity.
    Established fields (medical imaging):  Optuna learns low w_velocity.
    This domain-specific divergence is exactly what enables cross-domain
    generalization via meta-learning.
"""

from __future__ import annotations

import math
from datetime import datetime

from src.schemas import Article

# ============================================================
# Calibration constants
# ============================================================

# Citations/year at which velocity score saturates to 1.0.
# 20 is deliberately lower than impact_score's 50: we want velocity
# to saturate earlier to distinguish fast-rising papers (5-20 cpy)
# rather than the small number of citation giants (50+ cpy).
_VELOCITY_SATURATION = 20.0

# Minimum article age in years (avoid division by zero for current-year papers).
_MIN_AGE_YEARS = 1

# Score for papers too new to have meaningful citations (published this year).
# 0.50 = neutral, neither rewarded nor penalized for being new.
_BASELINE_TOO_NEW = 0.50


def velocity_score(article: Article, current_year: int | None = None) -> float:
    """Compute citation velocity for *article* — linear cites/year, [0, 1].

    Args:
        article: the article to score.
        current_year: override for the current year (useful in tests).
            Defaults to datetime.now().year.

    Returns:
        Float in [0, 1].
        - 0.00: zero citations on a non-new paper
        - 0.25: 5 citations/year  (moderate momentum)
        - 0.50: 10 citations/year (good momentum)
        - 1.00: 20+ citations/year (high velocity — saturated)

    Algorithm:
        1. age = max(1, current_year - article.year)
        2. cpy  = citation_count / age
        3. score = min(cpy / VELOCITY_SATURATION, 1.0)
        4. Special case: age <= 1 and no citations → 0.50 (neutral baseline)

    Notes:
        The relationship with impact_score is deliberate:
        - impact_score uses log → compresses high-citation spread
        - velocity_score uses linear → keeps spread at moderate citation rates
        Both are included as separate features so Optuna can weight them
        independently per domain.

    Examples:
        >>> from src.schemas import Article, SourceType
        >>> a = Article(id='x', title='t', year=2022,
        ...             source=SourceType.OPENALEX, url='u',
        ...             citation_count=10)
        >>> # age=2, cpy=5, score=5/20=0.25
        >>> velocity_score(a, current_year=2024)
        0.25
    """
    year_now = current_year if current_year is not None else datetime.now().year
    # No +1 offset: a 2022 paper in 2024 is 2 years old, not 3.
    # _MIN_AGE_YEARS=1 prevents division-by-zero for current-year papers.
    age = max(year_now - article.year, _MIN_AGE_YEARS)

    # Too new (age<=1, no citations yet) → neutral baseline
    if age <= 1 and (article.citation_count or 0) == 0:
        return _BASELINE_TOO_NEW

    cpy = (article.citation_count or 0) / age
    return min(cpy / _VELOCITY_SATURATION, 1.0)
