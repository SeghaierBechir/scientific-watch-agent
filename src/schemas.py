"""Pydantic schemas — the communication contract for the multi-agent system.

These schemas define the structure of all data exchanged between agents.
Strict validation in V2; in V1 we use `extra='ignore'` to be lenient with
incoming data from external APIs that may have unexpected fields.

All schemas follow the convention:
    - snake_case field names
    - Optional[X] = None for fields that may be missing
    - Field(...) for required fields with description
    - Field(default_factory=...) for mutable defaults
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ============================================================
# ENUMS — controlled vocabularies
# ============================================================


class SourceType(str, Enum):
    """Where the article was retrieved from."""

    OPENALEX = "openalex"
    ARXIV = "arxiv"
    SEMANTIC_SCHOLAR = "semantic_scholar"


class ArticleStatus(str, Enum):
    """Pipeline state of an article."""

    RAW = "raw"  # just retrieved
    SCORED = "scored"  # passed through Quality Critic
    SUMMARIZED = "summarized"  # summary generated
    REJECTED = "rejected"  # filtered out


# ============================================================
# Author
# ============================================================


class Author(BaseModel):
    """An author of a scientific article.

    Most fields are optional because OpenAlex doesn't always have full data,
    especially for less-known researchers.
    """

    model_config = ConfigDict(extra="ignore")  # tolerate extra fields from API

    name: str
    h_index: Optional[int] = Field(None, description="Hirsch index of the author")
    citation_count: Optional[int] = Field(None, description="Total citations")
    affiliation: Optional[str] = Field(None, description="Primary institution")
    orcid: Optional[str] = Field(None, description="ORCID identifier")
    openalex_id: Optional[str] = Field(None, description="OpenAlex author ID")


# ============================================================
# Article — the central data entity
# ============================================================


class Article(BaseModel):
    """A scientific article retrieved from any source.

    This is the central data object. It accumulates metadata as it flows
    through the pipeline. Some fields are populated at different stages.
    """

    model_config = ConfigDict(extra="ignore")

    # === Identifiers ===
    id: str = Field(..., description="Internal unique ID, e.g. 'openalex_W123456'")
    doi: Optional[str] = Field(None, description="DOI without 'https://doi.org/' prefix")

    # === Metadata ===
    title: str
    abstract: Optional[str] = None
    authors: list[Author] = Field(default_factory=list)
    year: int

    # === Source & venue ===
    source: SourceType
    journal_name: Optional[str] = None
    journal_issn: Optional[str] = Field(None, description="ISSN-L preferred")
    is_preprint: bool = False

    # === Bibliometrics ===
    citation_count: int = 0
    sjr_score: Optional[float] = Field(None, description="Scimago Journal Rank")
    quartile: Optional[Literal["Q1", "Q2", "Q3", "Q4"]] = None

    # === Access ===
    url: str
    open_access: bool = False

    # === Pipeline state ===
    status: ArticleStatus = ArticleStatus.RAW

    # === Topics/concepts (from OpenAlex) ===
    concepts: list[str] = Field(default_factory=list, description="Topic labels")


# ============================================================
# QualityScore — output of Quality Critic agent
# ============================================================


class QualityScore(BaseModel):
    """Output of the Quality Critic agent for one article.

    Level 3 adds velocity_score and recency_score as optional fields so
    that old code and persisted JSON remain backward-compatible:
    missing fields default to 0.0.
    """

    article_id: str
    venue_score: float = Field(..., ge=0, le=1)
    authors_score: float = Field(..., ge=0, le=1)
    impact_score: float = Field(..., ge=0, le=1)
    relevance_score: float = Field(..., ge=0, le=1)
    # Level 3 features — optional, default 0.0 for backward compatibility
    velocity_score: float = Field(default=0.0, ge=0, le=1)
    recency_score: float = Field(default=0.0, ge=0, le=1)
    final_score: float = Field(..., ge=0, le=1)
    weights_used: dict[str, float] = Field(
        ..., description="Snapshot of weights used, for traceability"
    )


# ============================================================
# ArticleSummary — output of Summarizer agent
# ============================================================


class ArticleSummary(BaseModel):
    """Structured summary of one article, produced by the Summarizer."""

    article_id: str
    problem: str = Field(..., description="Research problem addressed")
    method: str = Field(..., description="Method/approach used")
    dataset: Optional[str] = Field(None, description="Datasets used or built")
    results: str = Field(..., description="Main results")
    limitations: Optional[str] = Field(None, description="Limitations identified")
    key_contributions: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.now)

    @model_validator(mode="before")
    @classmethod
    def coerce_types(cls, data: object) -> object:
        """Coerce common model output mismatches before field validation.

        Two known failure modes across small / instruction-tuned models:

        1. Null string fields (Llama 8B, Groq models):
           Some models return `null` for required `str` fields when they
           cannot extract the information.  We coerce `None → ""` so ProTeGi
           can still score the example (empty string → low ROUGE / Judge score)
           rather than skip it entirely.

        2. String key_contributions (Haiku with optimised prompts):
           When the optimised prompt describes key_contributions in prose form,
           Haiku sometimes returns a plain string instead of a JSON array.
           We wrap it as a single-element list so validation passes and the
           content is still scored normally by the Judge.
        """
        if isinstance(data, dict):
            # Fix 1 — None → "" for required string fields
            for field_name in ("problem", "method", "results"):
                if data.get(field_name) is None:
                    data[field_name] = ""
            # Fix 2 — str → [str] for key_contributions
            kc = data.get("key_contributions")
            if isinstance(kc, str):
                data["key_contributions"] = [kc] if kc.strip() else []
        return data


# ============================================================
# NarrativeSummary — output of Summarizer in narrative mode
# ============================================================


class NarrativeSummary(BaseModel):
    """Free-text narrative summary produced by the Summarizer in narrative mode.

    Unlike ArticleSummary (6 structured fields), this is a single flowing
    paragraph — the same format as a scientific abstract.  This makes ROUGE
    comparison with PubMed gold abstracts directly meaningful.
    """

    article_id: str
    text: str = Field(
        ...,
        description="Narrative summary paragraph (150-250 words), past tense, prose style",
    )
    generated_at: datetime = Field(default_factory=datetime.now)


# ============================================================
# Synthesis — output of Synthesizer agent
# ============================================================


class Synthesis(BaseModel):
    """Global synthesis across all article summaries."""

    topic: str
    overview: str = Field(..., description="High-level overview of the field")
    main_approaches: list[str] = Field(..., description="Families of approaches")
    common_datasets: list[str] = Field(default_factory=list)
    key_findings: list[str]
    article_count: int = Field(..., ge=0)


# ============================================================
# TrendAnalysis — output of Trend Analyst agent
# ============================================================


class Trend(BaseModel):
    """One identified research trend."""

    name: str
    description: str
    evidence_article_ids: list[str] = Field(
        default_factory=list,
        description="IDs of articles supporting this trend",
    )
    maturity: Literal["emerging", "established", "declining"]


class ResearchGap(BaseModel):
    """A gap in the current research."""

    description: str
    importance: Literal["low", "medium", "high"]
    suggested_directions: list[str] = Field(default_factory=list)


class TrendAnalysis(BaseModel):
    """Output of the Trend Analyst agent."""

    trends: list[Trend] = Field(default_factory=list)
    gaps: list[ResearchGap] = Field(default_factory=list)
    # LLMs occasionally omit this field — default to [] so validation never
    # fails on a partial response. The retry in trend_analyst.run() will
    # attempt to fill it before falling back to the empty default.
    future_perspectives: list[str] = Field(default_factory=list)


# ============================================================
# ReActThoughtAction — one step in the ReAct loop (QueryExpander)
# ============================================================


class ReActThoughtAction(BaseModel):
    """One step produced by the ReAct QueryExpander at each iteration.

    The LLM outputs this schema to express:
      - its reasoning (thought)
      - the action it wants to take (search or stop)
      - the query to send to OpenAlex (if action='search')
      - why it stops (if action='stop')

    The agent loop reads this, executes the action, feeds back the
    observation, and calls the LLM again with the updated history.
    """

    thought: str = Field(
        ...,
        description="Reasoning about current coverage gaps and what to explore next",
    )
    action: Literal["search", "stop"] = Field(
        ...,
        description="'search' to probe OpenAlex, 'stop' when coverage is sufficient",
    )
    search_query: Optional[str] = Field(
        None,
        description="2-5 word query string for OpenAlex (required when action='search')",
    )
    stop_reason: Optional[str] = Field(
        None,
        description="Short explanation of why stopping (required when action='stop')",
    )


# ============================================================
# QueryExpansion — legacy single-shot schema (kept for reference)
# ============================================================


class QueryExpansion(BaseModel):
    """Expanded search queries produced by the QueryExpander agent.

    The expander takes one topic and returns N diverse queries that together
    cover the topic from multiple angles (synonyms, sub-tasks, adjacent terms).
    The original topic is always included as the first query.
    """

    queries: list[str] = Field(
        ...,
        min_length=1,
        max_length=6,
        description="Search queries, original first",
    )
    reasoning: str = Field(
        ...,
        description="Why these queries cover the topic well",
    )


# ============================================================
# CriticFeedback — for the Reflexion pattern
# ============================================================


class CriticFeedback(BaseModel):
    """Critique of an agent output, used in the Reflexion loop."""

    target: Literal["synthesis", "trends"]
    iteration: int = Field(..., ge=0)
    overall_quality: Literal["poor", "acceptable", "good", "excellent"]
    issues: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    needs_revision: bool


# ============================================================
# AgentLog — execution traces
# ============================================================


class AgentLog(BaseModel):
    """Trace of a single agent execution, for debugging and analytics."""

    agent_name: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    status: Literal["running", "success", "failed"]
    tokens_used: int = 0
    api_calls: int = 0
    error: Optional[str] = None
