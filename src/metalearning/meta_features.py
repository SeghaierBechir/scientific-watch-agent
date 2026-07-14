"""Domain meta-feature extraction — Phase 8a.

Each oracle domain is characterised by a fixed vector of quantitative
meta-features computed from its article corpus.  These features describe the
*structure* of a research field (temporal dynamics, citation inequality,
community size, venue concentration, semantic coherence) rather than the
content of individual papers.

The meta-feature vector is the input to the Phase-8c meta-learner, which
maps it to a set of Optuna-style scoring weights without needing to run
Optuna on that domain.

Why these 13 features?
    Each one is grounded in a theoretical motivation linking field properties
    to which scoring dimension should matter most:

    - pct_recent   ↔ velocity / recency weights (fast-moving fields)
    - citation_gini ↔ impact weight (few dominant papers vs. flat distribution)
    - unique_author_ratio ↔ authors weight (niche specialist community)
    - topic_concept_overlap ↔ relevance weight (semantically focused vs. broad)
    - pct_q1       ↔ venue weight (venue is informative only when most papers
                     appear in known, ranked venues)
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from src.features.scimago_loader import get_quartile as _scimago_quartile
from src.schemas import Article

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_META_DIR = _PROJECT_ROOT / "data" / "metalearning"

_CURRENT_YEAR: int = 2026
_RECENT_CUTOFF: int = 2024      # >= this year → "recent"
_HIGH_CITED_THRESHOLD: int = 100
_HIGH_HINDEX_THRESHOLD: int = 20

# Feature names in canonical order (used by as_vector / Phase 8c)
FEATURE_NAMES: list[str] = [
    "median_year",
    "pct_recent",
    "year_std",
    "citation_gini",
    "citation_median",
    "pct_high_cited",
    "unique_author_ratio",
    "mean_h_index",
    "pct_high_hindex",
    "pct_q1",
    "topic_concept_overlap",
    "gold_ratio",
    "grade2_ratio",
]


# ── Pydantic schema ───────────────────────────────────────────────────────────


class DomainMetaFeatures(BaseModel):
    """Quantitative fingerprint of a research domain corpus.

    All ratio fields are in [0, 1].  Absolute fields (median_year, year_std,
    citation_median, mean_h_index) are kept in their natural units for
    interpretability — the meta-learner normalises them at fit time.
    """

    domain_id: str
    topic: str
    corpus_size: int = Field(..., ge=1)

    # Temporal features
    median_year: float = Field(..., description="Median publication year of corpus articles")
    pct_recent: float = Field(..., ge=0, le=1, description="Fraction published >= 2024")
    year_std: float = Field(..., ge=0, description="Std dev of publication years")

    # Citation distribution features
    citation_gini: float = Field(..., ge=0, le=1, description="Gini of citation counts (0=equal, 1=concentrated)")
    citation_median: float = Field(..., ge=0, description="Median citation count")
    pct_high_cited: float = Field(..., ge=0, le=1, description="Fraction with >100 citations")

    # Author community features
    unique_author_ratio: float = Field(..., ge=0, le=1, description="Unique first authors / corpus size")
    mean_h_index: float = Field(..., ge=0, description="Mean h-index of first authors (non-null only)")
    pct_high_hindex: float = Field(..., ge=0, le=1, description="Fraction with any author h-index >= 20")

    # Venue features
    pct_q1: float = Field(..., ge=0, le=1, description="Fraction of articles in Q1 venues")

    # Semantic features
    topic_concept_overlap: float = Field(
        ..., ge=0, le=1,
        description="Fraction of articles whose concept list contains at least one topic word",
    )

    # Grade-structure features (from oracle)
    gold_ratio: float = Field(..., ge=0, le=1, description="(grade-1 + grade-2) / corpus_size")
    grade2_ratio: float = Field(..., ge=0, le=1, description="grade-2 / corpus_size")

    def as_vector(self) -> list[float]:
        """Return features in canonical FEATURE_NAMES order for the meta-learner."""
        return [getattr(self, name) for name in FEATURE_NAMES]


# ── Internal helpers ──────────────────────────────────────────────────────────


def _gini(values: list[float]) -> float:
    """Gini coefficient of a non-negative sequence (0 = perfect equality)."""
    if not values:
        return 0.0
    total = sum(values)
    if total == 0:
        return 0.0
    sorted_v = sorted(values)
    n = len(sorted_v)
    rank_sum = sum((i + 1) * v for i, v in enumerate(sorted_v))
    return (2 * rank_sum) / (n * total) - (n + 1) / n


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    sv = sorted(values)
    mid = len(sv) // 2
    return (sv[mid - 1] + sv[mid]) / 2 if len(sv) % 2 == 0 else sv[mid]


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))


# ── Public API ────────────────────────────────────────────────────────────────


def extract_meta_features(
    domain_id: str,
    topic: str,
    articles: list[Article],
    gold: dict[str, int],
) -> DomainMetaFeatures:
    """Compute the meta-feature vector for a domain corpus.

    Args:
        domain_id: Short identifier (e.g. "fake_news_detection").
        topic:     Free-text topic (e.g. "fake news detection").
        articles:  Full corpus (relevant + background articles).
        gold:      {article_id: grade} — grade in {1, 2}.

    Returns:
        DomainMetaFeatures with 13 scalar features.
    """
    n = len(articles)
    if n == 0:
        raise ValueError(f"Empty corpus for domain '{domain_id}'")

    topic_words = {w.lower() for w in topic.split() if len(w) > 2}

    # ── Temporal ──────────────────────────────────────────────────────────────
    years = [float(a.year) for a in articles if a.year is not None]
    median_year = _median(years) if years else float(_CURRENT_YEAR)
    pct_recent = sum(1 for y in years if y >= _RECENT_CUTOFF) / n
    year_std = _std(years)

    # ── Citation distribution ─────────────────────────────────────────────────
    citations = [float(a.citation_count) for a in articles if a.citation_count is not None]
    citation_gini = _gini(citations)
    citation_median = _median(citations)
    pct_high_cited = sum(1 for c in citations if c > _HIGH_CITED_THRESHOLD) / n

    # ── Author community ──────────────────────────────────────────────────────
    first_author_names: set[str] = set()
    h_indices: list[float] = []
    high_hindex_count = 0

    for a in articles:
        if a.authors:
            first_author_names.add(a.authors[0].name)
            if a.authors[0].h_index is not None:
                h_indices.append(float(a.authors[0].h_index))
        if any(
            au.h_index is not None and au.h_index >= _HIGH_HINDEX_THRESHOLD
            for au in a.authors
        ):
            high_hindex_count += 1

    unique_author_ratio = len(first_author_names) / n
    mean_h_index = sum(h_indices) / len(h_indices) if h_indices else 0.0
    pct_high_hindex = high_hindex_count / n

    # ── Venue ─────────────────────────────────────────────────────────────────
    q1_count = 0
    for a in articles:
        if a.quartile == "Q1":
            q1_count += 1
        elif a.journal_issn and _scimago_quartile(a.journal_issn) == "Q1":
            q1_count += 1
    pct_q1 = q1_count / n

    # ── Semantic ──────────────────────────────────────────────────────────────
    overlap_count = 0
    for a in articles:
        concepts_text = " ".join(c.lower() for c in (a.concepts or []))
        if any(w in concepts_text for w in topic_words):
            overlap_count += 1
    topic_concept_overlap = overlap_count / n

    # ── Grade structure ───────────────────────────────────────────────────────
    grade2_count = sum(1 for g in gold.values() if g == 2)
    grade1_count = sum(1 for g in gold.values() if g == 1)
    gold_ratio = (grade2_count + grade1_count) / n
    grade2_ratio = grade2_count / n

    return DomainMetaFeatures(
        domain_id=domain_id,
        topic=topic,
        corpus_size=n,
        median_year=median_year,
        pct_recent=pct_recent,
        year_std=year_std,
        citation_gini=citation_gini,
        citation_median=citation_median,
        pct_high_cited=pct_high_cited,
        unique_author_ratio=unique_author_ratio,
        mean_h_index=mean_h_index,
        pct_high_hindex=pct_high_hindex,
        pct_q1=pct_q1,
        topic_concept_overlap=topic_concept_overlap,
        gold_ratio=gold_ratio,
        grade2_ratio=grade2_ratio,
    )


def save_meta_features(features: DomainMetaFeatures) -> Path:
    """Persist meta-features to JSON in data/metalearning/."""
    _META_DIR.mkdir(parents=True, exist_ok=True)
    path = _META_DIR / f"{features.domain_id}_meta_features.json"
    path.write_text(features.model_dump_json(indent=2), encoding="utf-8")
    logger.info(f"[MetaFeatures] Saved -> {path}")
    return path


def load_meta_features(domain_id: str) -> Optional[DomainMetaFeatures]:
    """Load meta-features for one domain.  Returns None if not yet computed."""
    path = _META_DIR / f"{domain_id}_meta_features.json"
    if not path.exists():
        return None
    return DomainMetaFeatures.model_validate_json(path.read_text(encoding="utf-8"))


def load_all_meta_features() -> dict[str, DomainMetaFeatures]:
    """Load all saved meta-features. Keys are domain_ids."""
    if not _META_DIR.exists():
        return {}
    result: dict[str, DomainMetaFeatures] = {}
    for path in sorted(_META_DIR.glob("*_meta_features.json")):
        domain_id = path.stem.replace("_meta_features", "")
        try:
            result[domain_id] = DomainMetaFeatures.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except Exception as exc:
            logger.warning(f"[MetaFeatures] Could not load {path.name}: {exc}")
    return result
