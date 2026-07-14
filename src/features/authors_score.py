"""Author credibility score.

Computes a [0, 1] score reflecting the credibility of an article's authors.

Strategy:
    - Take the MAX h-index among all authors (one expert is enough to
      credibilize the paper, even if co-authors are juniors)
    - Apply log-normalization so h=100+ saturates at 1.0
    - Use a baseline if no h-index data is available

Why log normalization? An author with h=80 isn't 4x "better" than h=20.
log(h+1)/log(101) captures the perceived prestige curve much better.
"""

from __future__ import annotations

import math

from src.schemas import Article

# ============================================================
# Calibration constants
# ============================================================

# h-index value at which the score saturates to 1.0
# 100 is a reasonable "world-class researcher" threshold across most fields
_H_INDEX_SATURATION = 100

# Score returned when no h-index data is available for any author
# 0.3 is intentionally low so that articles with unknown authors don't
# get an undeserved boost
_BASELINE_NO_DATA = 0.30


def authors_score(article: Article) -> float:
    """Compute the author credibility score for an article.

    Args:
        article: the article to score.

    Returns:
        A float in [0, 1].

    Examples:
        >>> from src.schemas import Article, Author, SourceType
        >>> authors = [Author(name="Junior", h_index=5),
        ...            Author(name="Senior", h_index=85)]
        >>> a = Article(id="x", title="t", year=2024,
        ...             source=SourceType.OPENALEX, url="u", authors=authors)
        >>> # max h-index = 85, log(86)/log(101) ~= 0.965
        >>> 0.95 < authors_score(a) < 0.98
        True
    """
    if not article.authors:
        return _BASELINE_NO_DATA

    # Collect all available h-index values
    h_indices = [a.h_index for a in article.authors if a.h_index is not None]

    if not h_indices:
        return _BASELINE_NO_DATA

    # Take the max - one well-known expert is enough to credibilize the paper
    max_h = max(h_indices)

    # Log normalization: log(h+1) / log(saturation+1)
    # h=0   -> 0.0
    # h=10  -> log(11)/log(101) ~= 0.520
    # h=30  -> log(31)/log(101) ~= 0.745
    # h=100 -> log(101)/log(101) = 1.000
    # h=200 -> capped at 1.0
    score = math.log(max_h + 1) / math.log(_H_INDEX_SATURATION + 1)
    return min(score, 1.0)
