"""Tests for relevance_score_v2 (semantic embeddings).

We mock SentenceTransformer entirely — no network call, no 80 MB download.
The mock uses controlled unit-vectors so we can assert exact cosine values.

Test coverage:
    - score() output range: always in [_SCORE_FLOOR, 1.0]
    - identical text → score close to 1.0 (self-similarity)
    - orthogonal vectors → score close to 0.5 (cosine 0.0 → (0+1)/2)
    - anti-parallel vectors → score = _SCORE_FLOOR (floor applied)
    - LRU cache: encode() called only once for repeated identical text
    - clear_cache() resets cache size to 0
    - ImportError propagation when sentence-transformers absent
    - quality_scorer fallback: V1 used when _SEMANTIC_AVAILABLE is False
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.schemas import Article, SourceType

# ============================================================
# Helpers
# ============================================================

_SCORE_FLOOR = 0.05


def _make_article(
    article_id: str = "art_1",
    title: str = "Test article title",
    abstract: str | None = "Test abstract text about fake news detection.",
    concepts: list[str] | None = None,
) -> Article:
    return Article(
        id=article_id,
        title=title,
        abstract=abstract,
        concepts=concepts or [],
        year=2024,
        source=SourceType.OPENALEX,
        url="https://example.com/test",
    )


def _unit(v: np.ndarray) -> np.ndarray:
    """Normalize vector to unit length."""
    return (v / np.linalg.norm(v)).astype(np.float32)


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def mock_st_module():
    """Inject a fake sentence_transformers module into sys.modules.

    The mock SentenceTransformer.encode() returns deterministic unit-vectors
    based on the text content so we can assert exact cosine values.
    """
    # Pre-build controlled embeddings for known text patterns
    dim = 384
    _e1 = _unit(np.ones(dim))           # "parallel" direction
    _e2 = _unit(np.eye(dim)[0])         # e_0 direction

    def _fake_encode(text: str, normalize_embeddings: bool = True, **_kw) -> np.ndarray:
        """Return a deterministic unit-vector from the text hash."""
        rng = np.random.RandomState(abs(hash(text)) % 2 ** 31)
        vec = rng.rand(dim).astype(np.float32)
        return vec / np.linalg.norm(vec) if normalize_embeddings else vec

    mock_model = MagicMock()
    mock_model.encode.side_effect = _fake_encode

    mock_module = types.ModuleType("sentence_transformers")
    mock_module.SentenceTransformer = MagicMock(return_value=mock_model)

    old = sys.modules.get("sentence_transformers")
    sys.modules["sentence_transformers"] = mock_module
    yield mock_model
    # Restore
    if old is None:
        sys.modules.pop("sentence_transformers", None)
    else:
        sys.modules["sentence_transformers"] = old


@pytest.fixture(autouse=True)
def reset_scorer():
    """Reset the module-level singleton between tests to avoid cross-test state."""
    import src.features.relevance_score_v2 as mod
    original = mod._singleton
    mod._singleton = None
    yield
    mod._singleton = original


# ============================================================
# SemanticRelevanceScorer — unit tests
# ============================================================


class TestSemanticRelevanceScorer:

    def test_score_returns_float_in_range(self, mock_st_module):
        from src.features.relevance_score_v2 import SemanticRelevanceScorer
        scorer = SemanticRelevanceScorer()
        article = _make_article()
        s = scorer.score(article, "fake news detection")
        assert isinstance(s, float)
        assert _SCORE_FLOOR <= s <= 1.0

    def test_score_floor_applied(self, mock_st_module):
        """When cosine = -1.0 → mapped to 0.0 → floor kicks in."""
        from src.features.relevance_score_v2 import SemanticRelevanceScorer

        scorer = SemanticRelevanceScorer()
        dim = 384
        # Force anti-parallel embeddings by patching _embed directly
        vec_a = _unit(np.ones(dim))
        vec_b = _unit(-np.ones(dim))

        call_count = [0]

        def controlled_embed(text: str) -> np.ndarray:
            call_count[0] += 1
            return vec_a if call_count[0] == 1 else vec_b

        scorer._embed = controlled_embed
        s = scorer.score(_make_article(), "topic")
        # cosine(-1) → (−1+1)/2 = 0.0 → floor = 0.05
        assert s == pytest.approx(_SCORE_FLOOR, abs=1e-6)

    def test_self_similarity_near_one(self, mock_st_module):
        """Same text for topic and article → cosine ≈ 1.0 → score ≈ 1.0."""
        from src.features.relevance_score_v2 import SemanticRelevanceScorer

        scorer = SemanticRelevanceScorer()
        dim = 384
        vec = _unit(np.ones(dim))
        scorer._embed = lambda _text: vec  # always same vector

        article = _make_article(title="fake news detection", abstract=None, concepts=[])
        s = scorer.score(article, "fake news detection")
        assert s == pytest.approx(1.0, abs=1e-5)

    def test_orthogonal_gives_half(self, mock_st_module):
        """Orthogonal embeddings → cosine = 0.0 → score = 0.5."""
        from src.features.relevance_score_v2 import SemanticRelevanceScorer

        scorer = SemanticRelevanceScorer()
        dim = 384
        vec_a = np.zeros(dim, dtype=np.float32); vec_a[0] = 1.0
        vec_b = np.zeros(dim, dtype=np.float32); vec_b[1] = 1.0

        call_count = [0]

        def alt_embed(text: str) -> np.ndarray:
            call_count[0] += 1
            return vec_a if call_count[0] == 1 else vec_b

        scorer._embed = alt_embed
        s = scorer.score(_make_article(), "topic")
        assert s == pytest.approx(0.5, abs=1e-5)

    def test_no_abstract_still_scores(self, mock_st_module):
        """Article without abstract should not raise."""
        from src.features.relevance_score_v2 import SemanticRelevanceScorer
        scorer = SemanticRelevanceScorer()
        article = _make_article(abstract=None, concepts=[])
        s = scorer.score(article, "deep learning")
        assert _SCORE_FLOOR <= s <= 1.0

    def test_with_concepts(self, mock_st_module):
        """Concepts are appended to article text — encode must be called."""
        from src.features.relevance_score_v2 import SemanticRelevanceScorer
        scorer = SemanticRelevanceScorer()
        article = _make_article(concepts=["misinformation", "social media"])
        s = scorer.score(article, "fake news")
        assert _SCORE_FLOOR <= s <= 1.0


# ============================================================
# LRU cache behavior
# ============================================================


class TestLRUCache:

    def test_repeated_text_encoded_once(self, mock_st_module):
        """Scoring same article + topic twice should call encode() only once each."""
        from src.features.relevance_score_v2 import SemanticRelevanceScorer
        scorer = SemanticRelevanceScorer()
        article = _make_article()
        topic = "fake news detection"

        scorer.score(article, topic)
        encode_calls_after_first = mock_st_module.encode.call_count

        scorer.score(article, topic)
        encode_calls_after_second = mock_st_module.encode.call_count

        # Second call must not trigger new encode() calls
        assert encode_calls_after_second == encode_calls_after_first

    def test_different_texts_encoded_separately(self, mock_st_module):
        """Different articles require separate encode() calls."""
        from src.features.relevance_score_v2 import SemanticRelevanceScorer
        scorer = SemanticRelevanceScorer()

        article_a = _make_article(article_id="a1", title="Alpha article")
        article_b = _make_article(article_id="a2", title="Beta article different")
        topic = "machine learning"

        scorer.score(article_a, topic)
        calls_after_a = mock_st_module.encode.call_count

        scorer.score(article_b, topic)
        calls_after_b = mock_st_module.encode.call_count

        # topic was cached after first score; only article_b text is new
        assert calls_after_b > calls_after_a

    def test_clear_cache_resets_size(self, mock_st_module):
        from src.features.relevance_score_v2 import SemanticRelevanceScorer
        scorer = SemanticRelevanceScorer()
        scorer.score(_make_article(), "test topic")
        assert len(scorer._cache) > 0
        scorer.clear_cache()
        assert len(scorer._cache) == 0

    def test_cache_info_keys(self, mock_st_module):
        from src.features.relevance_score_v2 import SemanticRelevanceScorer
        from src.config import RELEVANCE_CACHE_SIZE
        scorer = SemanticRelevanceScorer()
        info = scorer.cache_info()
        assert "size" in info
        assert "capacity" in info
        assert info["capacity"] == RELEVANCE_CACHE_SIZE

    def test_lru_eviction(self):
        """When cache is full, LRU entry is evicted."""
        from src.features.relevance_score_v2 import _LRUCache
        cache = _LRUCache(maxsize=3)
        for i in range(3):
            cache.put(str(i), np.array([float(i)]))
        # Access key "0" to make it MRU
        cache.get("0")
        # Insert a 4th entry → LRU ("1") should be evicted
        cache.put("3", np.array([3.0]))
        assert cache.get("1") is None   # evicted
        assert cache.get("0") is not None  # survived (was accessed)
        assert cache.get("3") is not None  # new entry


# ============================================================
# Module-level public function
# ============================================================


class TestRelevanceScoreV2Function:

    def test_returns_float_in_range(self, mock_st_module):
        from src.features.relevance_score_v2 import relevance_score_v2
        article = _make_article()
        s = relevance_score_v2(article, "fake news detection")
        assert isinstance(s, float)
        assert _SCORE_FLOOR <= s <= 1.0

    def test_singleton_reused_across_calls(self, mock_st_module):
        """Multiple calls share the same SemanticRelevanceScorer instance."""
        import src.features.relevance_score_v2 as mod
        from src.features.relevance_score_v2 import relevance_score_v2

        relevance_score_v2(_make_article(article_id="x1"), "topic A")
        first_singleton = mod._singleton

        relevance_score_v2(_make_article(article_id="x2"), "topic B")
        second_singleton = mod._singleton

        assert first_singleton is second_singleton

    def test_import_error_if_no_sentence_transformers(self):
        """ImportError raised when sentence-transformers is missing.

        Setting sys.modules["sentence_transformers"] = None forces Python to
        raise ModuleNotFoundError (subclass of ImportError) on any import,
        even if the package is physically installed.
        """
        import src.features.relevance_score_v2 as mod

        # Setting the key to None makes `import sentence_transformers` fail
        # with ModuleNotFoundError regardless of whether it is installed.
        old = sys.modules.get("sentence_transformers", "ABSENT")
        sys.modules["sentence_transformers"] = None  # type: ignore[assignment]

        # Force a fresh scorer so _ensure_loaded() actually tries to import
        mod._singleton = None
        scorer = mod.SemanticRelevanceScorer()
        scorer._model = None

        try:
            with pytest.raises((ImportError, ModuleNotFoundError)):
                scorer._ensure_loaded()
        finally:
            if old == "ABSENT":
                sys.modules.pop("sentence_transformers", None)
            else:
                sys.modules["sentence_transformers"] = old
            mod._singleton = None


# ============================================================
# quality_scorer.py integration: fallback to V1
# ============================================================


class TestQualityScorerFallback:

    def test_uses_v1_when_semantic_unavailable(self):
        """When _SEMANTIC_AVAILABLE=False, quality_scorer uses V1."""
        import src.scoring.quality_scorer as qs_mod
        original = qs_mod._SEMANTIC_AVAILABLE

        try:
            qs_mod._SEMANTIC_AVAILABLE = False
            from src.scoring.quality_scorer import score_article
            article = _make_article()
            result = score_article(article, "fake news detection")
            # Score should still be a valid QualityScore
            assert 0.0 <= result.relevance_score <= 1.0
        finally:
            qs_mod._SEMANTIC_AVAILABLE = original

    def test_use_semantic_false_forces_v1(self, mock_st_module):
        """Passing use_semantic=False forces V1 even if V2 is available."""
        import src.scoring.quality_scorer as qs_mod
        original = qs_mod._SEMANTIC_AVAILABLE

        try:
            qs_mod._SEMANTIC_AVAILABLE = True  # V2 nominally available

            v1_call_count = [0]
            v2_call_count = [0]

            with (
                patch(
                    "src.scoring.quality_scorer.relevance_score_v1",
                    side_effect=lambda a, t: (v1_call_count.__setitem__(0, v1_call_count[0] + 1) or 0.5),
                ),
                patch(
                    "src.scoring.quality_scorer.relevance_score_v2",
                    side_effect=lambda a, t: (v2_call_count.__setitem__(0, v2_call_count[0] + 1) or 0.8),
                ),
            ):
                from src.scoring.quality_scorer import score_article
                score_article(_make_article(), "topic", use_semantic=False)

            assert v1_call_count[0] == 1
            assert v2_call_count[0] == 0
        finally:
            qs_mod._SEMANTIC_AVAILABLE = original

    def test_use_semantic_true_calls_v2(self, mock_st_module):
        """Passing use_semantic=True calls V2 when available."""
        import src.scoring.quality_scorer as qs_mod
        original = qs_mod._SEMANTIC_AVAILABLE

        try:
            qs_mod._SEMANTIC_AVAILABLE = True

            v2_call_count = [0]

            with patch(
                "src.scoring.quality_scorer.relevance_score_v2",
                side_effect=lambda a, t: (v2_call_count.__setitem__(0, v2_call_count[0] + 1) or 0.8),
            ):
                from src.scoring.quality_scorer import score_article
                score_article(_make_article(), "topic", use_semantic=True)

            assert v2_call_count[0] == 1
        finally:
            qs_mod._SEMANTIC_AVAILABLE = original
