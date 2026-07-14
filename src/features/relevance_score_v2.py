"""Semantic relevance score — V2 (sentence-transformers cosine similarity).

Replaces the V1 keyword/bigram approach with dense vector representations.
Both the user topic and the article text (title + abstract + concepts) are
embedded by a sentence-transformers model; cosine similarity gives the score.

Why this matters for the thesis
--------------------------------
V1 misses synonyms and conceptual relatedness:
    topic  = "fake news detection"
    V1 misses: "misinformation spread", "computational propaganda",
               "fact-checking neural approaches"  (different words, same domain)
V2 captures these because semantic embeddings encode *meaning*, not keywords.
This raises the theoretical NDCG ceiling from ~0.35 (V1) to ~0.80+.

Design decisions
-----------------
- **Lazy loading**: model imported on first call → import time unaffected.
- **Singleton**: model loaded once per process; shared across all calls.
- **LRU cache**: same text is embedded only once (important for batch scoring
  where the topic is repeated N times for N articles — only 1 embed call).
- **Thread-safe**: double-checked locking for singleton init; lock per cache op.
- **Fallback-friendly**: ImportError bubbles up to quality_scorer.py, which
  falls back to V1 gracefully if sentence-transformers is not installed.

Default model: all-MiniLM-L6-v2
    - 80 MB on disk, 384-dim unit-norm embeddings
    - ~10 ms/article on CPU (fast enough for 50-article batches)
    - Strong semantic similarity on short scientific text
    - HuggingFace: sentence-transformers/all-MiniLM-L6-v2

Score mapping
--------------
cosine ∈ [−1, 1]  →  (cosine + 1) / 2  ∈ [0, 1]  →  max(score, 0.05)

For normalized embeddings, dot product == cosine similarity.
Scientific texts rarely yield negative cosine values, but the mapping is
principled and consistent regardless.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from collections import OrderedDict

import numpy as np

from src.config import RELEVANCE_CACHE_SIZE, RELEVANCE_V2_MODEL
from src.schemas import Article

logger = logging.getLogger(__name__)

# Score floor — consistent with V1 so downstream log/sqrt never see zero.
_SCORE_FLOOR = 0.05


# ============================================================
# Thread-safe LRU cache
# ============================================================


class _LRUCache:
    """Thread-safe LRU cache: text-hash → embedding vector.

    Uses OrderedDict so move_to_end() gives O(1) recency tracking.
    """

    def __init__(self, maxsize: int) -> None:
        self._data: OrderedDict[str, np.ndarray] = OrderedDict()
        self._maxsize = maxsize
        self._lock = threading.Lock()

    def get(self, key: str) -> np.ndarray | None:
        """Return cached vector or None (promotes key to MRU position)."""
        with self._lock:
            if key not in self._data:
                return None
            self._data.move_to_end(key)
            return self._data[key]

    def put(self, key: str, value: np.ndarray) -> None:
        """Insert or update a key, evicting LRU entry if at capacity."""
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
            else:
                if len(self._data) >= self._maxsize:
                    self._data.popitem(last=False)  # remove LRU (front)
                self._data[key] = value

    def clear(self) -> None:
        """Evict all entries."""
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


# ============================================================
# Semantic scorer
# ============================================================


class SemanticRelevanceScorer:
    """Dense-vector semantic relevance scorer using sentence-transformers.

    Encodes the user topic and the article text into unit-norm vectors,
    then returns cosine similarity mapped to [0, 1].

    Thread-safe: double-checked locking for lazy model initialization.
    Embedding cache is shared across all calls to avoid redundant work.
    """

    def __init__(self, model_name: str = RELEVANCE_V2_MODEL) -> None:
        self._model_name = model_name
        self._model = None                   # loaded lazily on first call
        self._init_lock = threading.Lock()
        self._cache = _LRUCache(maxsize=RELEVANCE_CACHE_SIZE)

    # ── Model lifecycle ──────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        """Lazy-load the SentenceTransformer model (once per process).

        The deferred import means that if sentence-transformers is absent,
        ImportError is raised here rather than at module import time, allowing
        quality_scorer.py to catch it and fall back to V1 transparently.
        """
        if self._model is not None:
            return
        with self._init_lock:
            if self._model is not None:
                return  # another thread loaded it while we waited
            logger.info(
                "[RelevanceV2] Loading model '%s' (first call) …", self._model_name
            )
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415
            self._model = SentenceTransformer(self._model_name)
            logger.info(
                "[RelevanceV2] Model ready — cache capacity = %d", RELEVANCE_CACHE_SIZE
            )

    # ── Embedding ────────────────────────────────────────────────────────────

    def _embed(self, text: str) -> np.ndarray:
        """Encode *text* to a unit-norm vector, using LRU cache.

        The same topic string is embedded once then reused for every article
        in a batch — critical for performance when scoring 50+ articles.
        """
        key = hashlib.md5(text.encode("utf-8")).hexdigest()
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        self._ensure_loaded()
        embedding: np.ndarray = self._model.encode(
            text,
            normalize_embeddings=True,  # unit-norm → cosine == dot product
            show_progress_bar=False,
        )
        self._cache.put(key, embedding)
        return embedding

    # ── Scoring ──────────────────────────────────────────────────────────────

    def score(self, article: Article, topic: str) -> float:
        """Return semantic relevance score in [_SCORE_FLOOR, 1.0].

        Args:
            article: article to score.
            topic: user's search topic (e.g. "fake news detection").

        Returns:
            Float in [0.05, 1.0]; higher = more relevant.

        Algorithm:
            1. Build article_text = title + abstract + concepts (joined).
            2. Embed topic and article_text (unit-norm, cached).
            3. cosine_sim = dot(topic_emb, article_emb)  ∈ [−1, 1].
            4. score = (cosine_sim + 1) / 2  ∈ [0, 1].
            5. Apply floor = 0.05.
        """
        parts: list[str] = [article.title]
        if article.abstract:
            parts.append(article.abstract)
        if article.concepts:
            parts.append(" ".join(article.concepts))
        article_text = " ".join(parts)

        topic_emb = self._embed(topic)
        article_emb = self._embed(article_text)

        # Unit vectors → cosine = dot product (fast, no division needed)
        similarity = float(np.dot(topic_emb, article_emb))

        # Linear map [−1, 1] → [0, 1] then apply floor
        mapped = (similarity + 1.0) / 2.0
        return max(mapped, _SCORE_FLOOR)

    # ── Utilities ────────────────────────────────────────────────────────────

    def cache_info(self) -> dict[str, int]:
        """Return current cache occupancy and capacity."""
        return {"size": len(self._cache), "capacity": RELEVANCE_CACHE_SIZE}

    def clear_cache(self) -> None:
        """Evict all cached embeddings (useful between independent runs)."""
        self._cache.clear()
        logger.debug("[RelevanceV2] Embedding cache cleared")


# ============================================================
# Module-level singleton + public API
# ============================================================

_singleton: SemanticRelevanceScorer | None = None
_singleton_lock = threading.Lock()


def _get_scorer(model_name: str = RELEVANCE_V2_MODEL) -> SemanticRelevanceScorer:
    """Return the process-wide scorer instance, creating it on first call."""
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = SemanticRelevanceScorer(model_name)
    return _singleton


def relevance_score_v2(article: Article, topic: str) -> float:
    """Compute semantic relevance of *article* w.r.t. *topic*.

    Drop-in V2 replacement for ``relevance_score`` in quality_scorer.py.
    Same signature, same [0, 1] output range.

    Args:
        article: the Article to score.
        topic: search topic string.

    Returns:
        Float in [0.05, 1.0].

    Raises:
        ImportError: if ``sentence-transformers`` is not installed.
            quality_scorer.py catches this and falls back to V1 automatically.
    """
    return _get_scorer().score(article, topic)


def clear_embedding_cache() -> None:
    """Clear the global embedding cache.

    Call this between independent benchmark runs to ensure no cross-run
    cache hits skew timing measurements.
    """
    if _singleton is not None:
        _singleton.clear_cache()
