"""Topic relevance score (V1.5 - keyword + bigram phrase matching).

Computes a [0, 1] score reflecting how well an article matches the user's topic.

V1 strategy (unigram, as decided in design):
    - Tokenize topic into significant words (remove stop words)
    - Score = weighted match of these tokens in title + abstract + concepts
    - Title matches count more than abstract matches (TF-IDF-like intuition)

V1.5 upgrade (this file) — bigram phrase matching:
    - Extract consecutive token pairs (bigrams) from the topic
    - Award a BONUS for each bigram that appears as a consecutive pair in
      the article text (title, abstract, concepts)
    - Rationale: "attention mechanism" as a *phrase* is stronger evidence of
      relevance than accidentally having "attention" and "mechanism" in an
      unrelated LSTM paper that mentions both incidentally
    - This fixes the case where "recurrent attention mechanism dynamic position"
      was retrieving pure RNN/LSTM papers that matched only "recurrent"
    - Normalization: max_possible includes the maximum bigram bonus so [0, 1]
      is preserved

V2 strategy (later):
    - Compute cosine similarity between embed(topic) and embed(title+abstract)
    - Use Voyage AI voyage-3-lite or sentence-transformers SPECTER2

The V1.5 fallback is decent for multi-word topics but weaker for single-word
or highly conceptual queries. The V2 upgrade will fix the remaining cases.
"""

from __future__ import annotations

import re
from typing import Iterable

from src.schemas import Article

# ============================================================
# Stop words - common English words that don't carry meaning
# Kept minimal; over-removing stop words can hurt for short queries
# ============================================================

_STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "must", "shall", "can", "this",
    "that", "these", "those", "as", "if", "then", "than", "so", "very",
})

# ============================================================
# Unigram field weights
# Maximum unigram contribution = 3 + 1 + 2 = 6
# ============================================================
_WEIGHT_TITLE = 3.0
_WEIGHT_ABSTRACT = 1.0
_WEIGHT_CONCEPTS = 2.0  # OpenAlex concepts are curated, so they matter

# ============================================================
# Bigram phrase-match bonus weights
# Applied on top of the unigram score when consecutive token pairs match.
# Maximum bigram contribution = 1.5 + 0.5 + 1.0 = 3.0
# Overall max_possible (all unigrams + all bigrams) = 6 + 3 = 9
# ============================================================
_WEIGHT_TITLE_BIGRAM = 1.5
_WEIGHT_ABSTRACT_BIGRAM = 0.5
_WEIGHT_CONCEPTS_BIGRAM = 1.0

# Score floor - even completely irrelevant articles get a tiny baseline
# (otherwise log/sqrt operations downstream would explode)
_SCORE_FLOOR = 0.05


def _tokenize(text: str) -> list[str]:
    """Lowercase + split on non-word chars (including underscores) + remove stop words.

    Underscores are treated as word separators so that slug-style inputs like
    "federated_learning" tokenize identically to "federated learning".
    This matters when users pass CLI arguments that use underscores instead of spaces.

    >>> _tokenize("Detecting fake news with the BERT model.")
    ['detecting', 'fake', 'news', 'bert', 'model']
    >>> _tokenize("federated_learning")  # underscore → two tokens
    ['federated', 'learning']
    """
    # First replace underscores with spaces (underscore is \w so the regex below
    # would leave it intact), then replace remaining non-word chars with spaces.
    normalized = text.lower().replace("_", " ")
    cleaned = re.sub(r"[^\w\s]", " ", normalized)
    tokens = cleaned.split()
    return [t for t in tokens if t and t not in _STOP_WORDS and len(t) > 1]


def _extract_bigrams(tokens: list[str]) -> list[str]:
    """Return all consecutive token pairs as 'tok1 tok2' strings.

    >>> _extract_bigrams(['fake', 'news', 'detection'])
    ['fake news', 'news detection']
    """
    return [f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)]


def _count_matches(query_tokens: Iterable[str], target_tokens: list[str]) -> int:
    """Count how many query tokens appear in target_tokens (any position)."""
    target_set = set(target_tokens)
    return sum(1 for q in query_tokens if q in target_set)


def _bigram_hits(query_bigrams: list[str], tokens: list[str]) -> int:
    """Count how many query bigrams appear as *consecutive* pairs in tokens.

    Uses a set of adjacent pairs rather than substring matching so that
    punctuation-split tokens (already cleaned by _tokenize) are handled
    correctly.  E.g. 'fake-news detection' tokenizes to
    ['fake', 'news', 'detection'] → pair 'fake news' is found.

    Args:
        query_bigrams: list of 'tok1 tok2' strings extracted from the topic.
        tokens: already-tokenized field text (title / abstract / concepts).

    Returns:
        Number of matching bigrams (0 if tokens has fewer than 2 elements).
    """
    if len(tokens) < 2 or not query_bigrams:
        return 0
    # Build the set of consecutive pairs present in the target
    token_pairs = {f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)}
    return sum(1 for bg in query_bigrams if bg in token_pairs)


def relevance_score(article: Article, topic: str) -> float:
    """Compute the topic relevance score for an article (V1.5: unigram + bigram).

    Args:
        article: the article to score.
        topic: the user's search topic (e.g. "fake news detection").

    Returns:
        A float in [0, 1].

    Algorithm:
        1. Unigram match: fraction of topic tokens found in each field,
           weighted by field importance (title > concepts > abstract).
        2. Bigram bonus: fraction of topic bigrams (consecutive pairs) found
           as consecutive pairs in each field, with separate weights.
        3. Normalize by max_possible = 9.0 (all unigrams + all bigrams match).
        4. Apply floor = 0.05 so downstream log/sqrt never see zero.

    Examples:
        >>> from src.schemas import Article, SourceType
        >>> a = Article(id="x",
        ...             title="A novel approach for fake news detection",
        ...             abstract="We propose a new model.",
        ...             year=2024, source=SourceType.OPENALEX, url="u")
        >>> # All 3 topic tokens (fake, news, detection) match in title
        >>> relevance_score(a, "fake news detection") > 0.4
        True
    """
    topic_tokens = _tokenize(topic)
    if not topic_tokens:
        return _SCORE_FLOOR

    n_topic_tokens = len(topic_tokens)
    topic_bigrams = _extract_bigrams(topic_tokens)
    n_bigrams = len(topic_bigrams)

    # ── Unigram: match in title (heavy weight) ────────────────────────────
    title_tokens = _tokenize(article.title)
    title_matches = _count_matches(topic_tokens, title_tokens)
    title_score = (title_matches / n_topic_tokens) * _WEIGHT_TITLE

    # ── Unigram: match in abstract (light weight) ─────────────────────────
    abstract_score = 0.0
    abstract_tokens: list[str] = []
    if article.abstract:
        abstract_tokens = _tokenize(article.abstract)
        abstract_matches = _count_matches(topic_tokens, abstract_tokens)
        abstract_score = (abstract_matches / n_topic_tokens) * _WEIGHT_ABSTRACT

    # ── Unigram: match in OpenAlex concepts (medium weight) ───────────────
    concepts_score = 0.0
    concept_tokens: list[str] = []
    if article.concepts:
        concepts_text = " ".join(article.concepts)
        concept_tokens = _tokenize(concepts_text)
        concept_matches = _count_matches(topic_tokens, concept_tokens)
        concepts_score = (concept_matches / n_topic_tokens) * _WEIGHT_CONCEPTS

    # ── Bigram phrase bonus ───────────────────────────────────────────────
    # Only applies when the topic has ≥2 tokens (single-token topics have
    # no bigrams, so max_possible stays at 6.0 — backward-compatible).
    bigram_bonus = 0.0
    if n_bigrams > 0:
        title_bg = _bigram_hits(topic_bigrams, title_tokens) / n_bigrams
        bigram_bonus += title_bg * _WEIGHT_TITLE_BIGRAM

        if abstract_tokens:
            abs_bg = _bigram_hits(topic_bigrams, abstract_tokens) / n_bigrams
            bigram_bonus += abs_bg * _WEIGHT_ABSTRACT_BIGRAM

        if concept_tokens:
            con_bg = _bigram_hits(topic_bigrams, concept_tokens) / n_bigrams
            bigram_bonus += con_bg * _WEIGHT_CONCEPTS_BIGRAM

    # ── Combine and normalize ─────────────────────────────────────────────
    raw = title_score + abstract_score + concepts_score + bigram_bonus
    max_possible = (
        _WEIGHT_TITLE + _WEIGHT_ABSTRACT + _WEIGHT_CONCEPTS
        + ((_WEIGHT_TITLE_BIGRAM + _WEIGHT_ABSTRACT_BIGRAM + _WEIGHT_CONCEPTS_BIGRAM)
           if n_bigrams > 0 else 0.0)
    )
    normalized = raw / max_possible

    # Apply floor so even irrelevant articles get something
    return max(normalized, _SCORE_FLOOR)
