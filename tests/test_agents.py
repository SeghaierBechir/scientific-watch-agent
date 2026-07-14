"""Tests for the V2 agent layer.

We mock OpenAlexClient (Searcher), and inject fake LLMClients into the
LLM-using agents (Summarizer, Synthesizer, TrendAnalyst). No real API call
is made.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.agents import (
    quality_critic,
    searcher,
    summarizer,
    synthesizer,
    trend_analyst,
)
from src.agents.graph import build_graph, run_pipeline
from src.llm.base import LLMResponse, LLMStructuredOutputError, Message
from src.schemas import (
    Article,
    ArticleStatus,
    ArticleSummary,
    Author,
    QualityScore,
    ResearchGap,
    SourceType,
    Synthesis,
    Trend,
    TrendAnalysis,
)


# ============================================================
# Helpers
# ============================================================


def make_article(idx: int = 1, with_abstract: bool = True) -> Article:
    return Article(
        id=f"openalex_W{idx}",
        title=f"Detection of Fake News Using Method {idx}",
        abstract=(
            "We propose a transformer-based classifier trained on the LIAR dataset. "
            "Results show 88% accuracy, outperforming bag-of-words baselines."
        )
        if with_abstract
        else None,
        authors=[Author(name=f"Author {idx}", h_index=20 + idx, citation_count=500 + idx * 10)],
        year=2023,
        source=SourceType.OPENALEX,
        journal_name="ACL Findings",
        citation_count=50,
        url=f"https://openalex.org/W{idx}",
        concepts=["fake news", "transformers"],
    )


def make_summary(article_id: str) -> ArticleSummary:
    return ArticleSummary(
        article_id=article_id,
        problem="Fake news classification on social media.",
        method="Transformer fine-tuning.",
        dataset="LIAR",
        results="88% accuracy.",
        limitations="Limited to English.",
        key_contributions=["Transformer baseline", "Released splits"],
    )


class FakeLLM:
    """Configurable fake LLM that returns prepared structured outputs."""

    def __init__(self, structured_returns: list, provider: str = "anthropic", model: str = "fake-1"):
        self._queue = list(structured_returns)
        self.calls = 0
        self._provider = provider
        self._model = model

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def model(self) -> str:
        return self._model

    def chat(self, system, messages, temperature=0.0, max_tokens=1024):
        raise NotImplementedError

    def chat_structured(self, system, messages, schema, temperature=0.0, max_tokens=2048):
        self.calls += 1
        if not self._queue:
            raise AssertionError("FakeLLM queue exhausted")
        item = self._queue.pop(0)
        if isinstance(item, Exception):
            raise item
        resp = LLMResponse(
            content="{}",
            model=self._model,
            provider=self._provider,  # type: ignore[arg-type]
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.001,
        )
        return item, resp


# ============================================================
# Searcher
# ============================================================


class TestSearcher:
    def test_run_returns_articles(self):
        fake_articles = [make_article(1), make_article(2)]
        with patch("src.agents.searcher.OpenAlexClient") as MockClient:
            MockClient.return_value.search.return_value = fake_articles
            state = {"topic": "fake news detection", "config": {"n_raw": 10, "from_year": 2020}}
            out = searcher.run(state)

        assert len(out["raw_articles"]) == 2
        assert out["logs"][0].status == "success"

    def test_run_handles_openalex_error(self):
        from src.sources.openalex import OpenAlexError

        with patch("src.agents.searcher.OpenAlexClient") as MockClient:
            MockClient.return_value.search.side_effect = OpenAlexError("503")
            out = searcher.run({"topic": "x", "config": {}})

        assert out["raw_articles"] == []
        assert out["logs"][0].status == "failed"
        assert "Searcher" in out["errors"][0]


# ============================================================
# QualityCritic
# ============================================================


class TestQualityCritic:
    def test_filters_to_top_n(self):
        articles = [make_article(i) for i in range(1, 6)]
        state = {
            "topic": "fake news detection",
            "config": {"top_n": 3},
            "raw_articles": articles,
        }
        out = quality_critic.run(state)

        assert len(out["quality_scores"]) == 5
        assert len(out["top_articles"]) == 3
        assert all(a.status == ArticleStatus.SCORED for a in out["top_articles"])
        # Sorted descending
        scores = [s.final_score for s in out["top_scores"]]
        assert scores == sorted(scores, reverse=True)

    def test_handles_empty_raw(self):
        out = quality_critic.run({"topic": "x", "config": {}, "raw_articles": []})
        assert out["top_articles"] == []
        assert out["logs"][0].status == "success"

    def test_off_topic_articles_excluded_by_relevance_threshold(self):
        """Articles whose relevance_score < MIN_RELEVANCE_SCORE are dropped from
        top_articles even if their final_score is high (good venue / impact).
        All scores are still returned in quality_scores for traceability.
        """
        from unittest.mock import patch

        from src.config import MIN_RELEVANCE_SCORE
        from src.schemas import QualityScore

        articles = [make_article(i) for i in range(1, 4)]

        # Article 1: high-impact but off-topic (relevance below threshold)
        # Article 2 & 3: on-topic
        fake_scores = [
            QualityScore(
                article_id="openalex_W1",
                venue_score=0.9, authors_score=0.9, impact_score=0.9,
                relevance_score=0.05,   # << off-topic
                final_score=0.75,
                weights_used={"venue": 0.25, "authors": 0.20, "impact": 0.25, "relevance": 0.30},
            ),
            QualityScore(
                article_id="openalex_W2",
                venue_score=0.7, authors_score=0.7, impact_score=0.7,
                relevance_score=0.65,   # on-topic
                final_score=0.68,
                weights_used={"venue": 0.25, "authors": 0.20, "impact": 0.25, "relevance": 0.30},
            ),
            QualityScore(
                article_id="openalex_W3",
                venue_score=0.6, authors_score=0.6, impact_score=0.6,
                relevance_score=0.55,   # on-topic
                final_score=0.60,
                weights_used={"venue": 0.25, "authors": 0.20, "impact": 0.25, "relevance": 0.30},
            ),
        ]

        with patch("src.agents.quality_critic.score_articles", return_value=fake_scores):
            out = quality_critic.run({
                "topic": "attention mechanism transformer",
                "config": {"top_n": 3},
                "raw_articles": articles,
            })

        # ALL 3 scores kept for traceability
        assert len(out["quality_scores"]) == 3
        # Off-topic article excluded → only 2 relevant remain (top_n=3 but only 2 pass)
        assert len(out["top_articles"]) == 2
        top_ids = [a.id for a in out["top_articles"]]
        assert "openalex_W1" not in top_ids, (
            "High-impact off-topic article must NOT appear in top_articles"
        )
        assert out["logs"][0].status == "success"

    def test_all_off_topic_returns_empty_top(self):
        """When ALL articles are off-topic, top_articles is empty (pipeline continues)."""
        from unittest.mock import patch

        from src.schemas import QualityScore

        articles = [make_article(1)]
        fake_scores = [
            QualityScore(
                article_id="openalex_W1",
                venue_score=1.0, authors_score=1.0, impact_score=1.0,
                relevance_score=0.05,   # below threshold
                final_score=0.80,
                weights_used={"venue": 0.25, "authors": 0.20, "impact": 0.25, "relevance": 0.30},
            )
        ]

        with patch("src.agents.quality_critic.score_articles", return_value=fake_scores):
            out = quality_critic.run({
                "topic": "attention mechanism",
                "config": {"top_n": 5},
                "raw_articles": articles,
            })

        assert out["top_articles"] == []
        assert out["quality_scores"] == fake_scores  # full scores preserved
        assert out["logs"][0].status == "success"    # graceful degradation

    def test_adaptive_gate_lowers_when_automl_relevance_weight_low(self):
        """When AutoML weights have relevance < 0.15, gate lowers to 0.06.

        Regression: 'federated_learning' domain had relevance weight=0.073,
        causing ALL articles to be excluded (score=0.05 floor < gate=0.20).
        The adaptive gate should kick in and allow articles through.
        """
        from unittest.mock import patch

        from src.schemas import QualityScore

        articles = [make_article(1)]
        # Article has relevance_score=0.10 (above floor but below default gate 0.20)
        fake_scores = [
            QualityScore(
                article_id="openalex_W1",
                venue_score=0.8, authors_score=0.7, impact_score=0.9,
                relevance_score=0.10,   # above floor (0.05), below default gate (0.20)
                final_score=0.75,
                weights_used={"venue": 0.10, "authors": 0.31, "impact": 0.51, "relevance": 0.07},
            )
        ]
        # AutoML learned weights: relevance=0.073 (< 0.15 threshold)
        low_relevance_weights = {
            "venue": 0.10, "authors": 0.31, "impact": 0.51, "relevance": 0.073
        }

        with patch("src.agents.quality_critic.score_articles", return_value=fake_scores), \
             patch("src.agents.quality_critic.load_weights_for_topic",
                   return_value=low_relevance_weights):
            out = quality_critic.run({
                "topic": "federated_learning",
                "config": {"top_n": 5},
                "raw_articles": articles,
            })

        # With adaptive gate (0.06), relevance_score=0.10 should pass
        assert len(out["top_articles"]) == 1, (
            "Adaptive gate failed: article with relevance_score=0.10 was excluded "
            "despite AutoML relevance weight < 0.15"
        )


# ============================================================
# Summarizer
# ============================================================


class TestSummarizer:
    def test_summarizes_each_top_article(self):
        articles = [make_article(1), make_article(2)]
        fake_summaries = [
            make_summary("will-be-overwritten-1"),
            make_summary("will-be-overwritten-2"),
        ]
        llm = FakeLLM(structured_returns=fake_summaries)
        state = {"topic": "x", "config": {}, "top_articles": articles}

        out = summarizer.run(state, llm=llm)

        assert len(out["summaries"]) == 2
        # IDs are stamped from the source articles, not from the LLM.
        assert out["summaries"][0].article_id == "openalex_W1"
        assert out["summaries"][1].article_id == "openalex_W2"
        assert llm.calls == 2
        assert out["logs"][0].api_calls == 2

    def test_partial_failure_keeps_successes(self):
        from src.llm.base import LLMStructuredOutputError

        articles = [make_article(1), make_article(2)]
        llm = FakeLLM(
            structured_returns=[
                make_summary("x"),
                LLMStructuredOutputError("bad json"),
            ]
        )
        out = summarizer.run({"topic": "x", "config": {}, "top_articles": articles}, llm=llm)
        assert len(out["summaries"]) == 1
        assert "Summarizer" in out.get("errors", [""])[0]

    def test_no_top_articles(self):
        out = summarizer.run({"topic": "x", "config": {}, "top_articles": []}, llm=FakeLLM([]))
        assert out["summaries"] == []


# ============================================================
# Synthesizer
# ============================================================


class TestSynthesizer:
    def test_calls_llm_once_with_all_summaries(self):
        synth = Synthesis(
            topic="fake news detection",
            overview="Field overview.",
            main_approaches=["Transformers", "Graph neural nets"],
            common_datasets=["LIAR"],
            key_findings=["Transformers dominate.", "Datasets are small."],
            article_count=2,
        )
        llm = FakeLLM(structured_returns=[synth])
        state = {
            "topic": "fake news detection",
            "summaries": [make_summary("a"), make_summary("b")],
        }
        out = synthesizer.run(state, llm=llm)

        assert llm.calls == 1
        assert out["synthesis"].article_count == 2
        assert out["synthesis"].topic == "fake news detection"

    def test_skips_when_no_summaries(self):
        out = synthesizer.run({"topic": "x", "summaries": []}, llm=FakeLLM([]))
        assert out["synthesis"] is None


# ============================================================
# TrendAnalyst
# ============================================================


class TestTrendAnalyst:
    def test_emits_trend_analysis(self):
        synth = Synthesis(
            topic="t",
            overview="o",
            main_approaches=["a"],
            common_datasets=[],
            key_findings=["k"],
            article_count=1,
        )
        ta = TrendAnalysis(
            trends=[
                Trend(
                    name="LLM-based detection",
                    description="Use of LLMs.",
                    evidence_article_ids=["a"],
                    maturity="emerging",
                )
            ],
            gaps=[
                ResearchGap(
                    description="Few multilingual datasets.",
                    importance="high",
                    suggested_directions=["Build a benchmark."],
                )
            ],
            future_perspectives=["Cross-lingual transfer."],
        )
        llm = FakeLLM(structured_returns=[ta])

        state = {
            "topic": "t",
            "summaries": [make_summary("a")],
            "synthesis": synth,
        }
        out = trend_analyst.run(state, llm=llm)

        assert llm.calls == 1
        assert len(out["trend_analysis"].trends) == 1

    def test_retry_when_future_perspectives_missing(self):
        """If first call returns no future_perspectives, agent retries once."""
        synth = Synthesis(
            topic="t", overview="o", main_approaches=["a"],
            common_datasets=[], key_findings=["k"], article_count=1,
        )
        # First response: valid TrendAnalysis but future_perspectives is empty.
        ta_partial = TrendAnalysis(
            trends=[Trend(name="T", description="d",
                          evidence_article_ids=["a"], maturity="emerging")],
            gaps=[ResearchGap(description="g", importance="low",
                              suggested_directions=[])],
            future_perspectives=[],   # ← missing!
        )
        # Second response (retry): complete.
        ta_complete = TrendAnalysis(
            trends=ta_partial.trends,
            gaps=ta_partial.gaps,
            future_perspectives=["Future direction 1.", "Future direction 2."],
        )
        llm = FakeLLM(structured_returns=[ta_partial, ta_complete])
        state = {"topic": "t", "summaries": [make_summary("a")], "synthesis": synth}
        out = trend_analyst.run(state, llm=llm)

        # Two LLM calls: original + retry.
        assert llm.calls == 2
        assert out["logs"][0].api_calls == 2
        assert len(out["trend_analysis"].future_perspectives) == 2

    def test_partial_result_kept_when_retry_fails(self):
        """If retry also fails validation, the partial first result is kept."""
        synth = Synthesis(
            topic="t", overview="o", main_approaches=["a"],
            common_datasets=[], key_findings=["k"], article_count=1,
        )
        ta_partial = TrendAnalysis(
            trends=[Trend(name="T", description="d",
                          evidence_article_ids=["a"], maturity="emerging")],
            gaps=[],
            future_perspectives=[],
        )
        llm = FakeLLM(structured_returns=[ta_partial,
                                          LLMStructuredOutputError("bad json")])
        state = {"topic": "t", "summaries": [make_summary("a")], "synthesis": synth}
        out = trend_analyst.run(state, llm=llm)

        # Two calls attempted; partial result still returned (not None).
        assert llm.calls == 2
        assert out["trend_analysis"] is not None
        assert len(out["trend_analysis"].trends) == 1

    def test_skips_when_synthesis_missing(self):
        out = trend_analyst.run(
            {"topic": "x", "summaries": [make_summary("a")], "synthesis": None},
            llm=FakeLLM([]),
        )
        assert out["trend_analysis"] is None


# ============================================================
# Graph integration (mocked everything)
# ============================================================


class TestGraph:
    def test_build_graph_returns_compiled(self):
        graph = build_graph()
        assert graph is not None  # compiled langgraph object

    def test_pipeline_end_to_end_with_mocks(self):
        articles = [make_article(i) for i in range(1, 4)]
        synth = Synthesis(
            topic="t",
            overview="o",
            main_approaches=["a", "b"],
            common_datasets=[],
            key_findings=["k1", "k2", "k3"],
            article_count=3,
        )
        ta = TrendAnalysis(
            trends=[
                Trend(
                    name="x",
                    description="d",
                    evidence_article_ids=["openalex_W1"],
                    maturity="emerging",
                )
            ],
            gaps=[
                ResearchGap(description="g", importance="medium", suggested_directions=["d1"])
            ],
            future_perspectives=["p1"],
        )

        # ── Patch (1) OpenAlex clients  (2) get_llm_for_task ────────────────
        summarizer_llm = FakeLLM([make_summary("x") for _ in range(3)])
        synth_llm  = FakeLLM([synth])
        trend_llm  = FakeLLM([ta])

        # ReAct expander: 1 search then stop
        from src.schemas import ReActThoughtAction as RAT
        expander_llm = FakeLLM([
            RAT(thought="Start with main topic", action="search",
                search_query="fake news detection"),
            RAT(thought="Good coverage", action="stop",
                stop_reason="One angle covered."),
        ])

        # Critic approves immediately (quality=good, needs_revision=False).
        from src.schemas import CriticFeedback as CF
        critic_llm = FakeLLM([
            CF(target="synthesis", iteration=1, overall_quality="good",
               issues=[], suggestions=[], needs_revision=False)
        ])

        def fake_get_llm(task):
            return {
                "query_expansion": expander_llm,
                "summarize":       summarizer_llm,
                "synthesize":      synth_llm,
                "critic":          critic_llm,
                "trend_analysis":  trend_llm,
            }[task]

        with patch("src.agents.searcher.OpenAlexClient") as MockSearcher, \
             patch("src.agents.query_expander.OpenAlexClient") as MockExpander, \
             patch("src.agents.query_expander.get_llm_for_task", side_effect=fake_get_llm), \
             patch("src.agents.summarizer.get_llm_for_task",     side_effect=fake_get_llm), \
             patch("src.agents.synthesizer.get_llm_for_task",    side_effect=fake_get_llm), \
             patch("src.agents.critic.get_llm_for_task",         side_effect=fake_get_llm), \
             patch("src.agents.trend_analyst.get_llm_for_task",  side_effect=fake_get_llm):
            MockSearcher.return_value.search.return_value = articles
            MockExpander.return_value.search.return_value = articles[:2]  # ReAct probe
            final = run_pipeline("fake news detection", n_raw=3, top_n=3)

        assert len(final["raw_articles"]) == 3
        assert len(final["top_articles"]) == 3
        assert len(final["summaries"]) == 3
        assert final["synthesis"] is not None
        assert final["trend_analysis"] is not None
        # All 7 agents logged (QueryExpander + Critic added).
        assert len(final["logs"]) == 7
        assert final.get("errors", []) == []
        # ReAct steps are recorded
        assert len(final.get("react_steps", [])) >= 1
