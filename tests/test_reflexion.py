"""Tests for Phase 6 — Reflexion pattern (Critic + Synthesizer revision loop).

Covers:
    - Critic evaluates synthesis and sets needs_revision correctly
    - Critic forces approval when MAX_REFLEXION_ITERATIONS reached
    - Critic LLM failure falls back to approved (pipeline never blocked)
    - Synthesizer uses critic feedback on revision runs
    - Synthesizer increments synthesis_iteration each run
    - route_after_critic routing function
    - End-to-end: loop runs until quality OK or max iterations
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.agents import critic, synthesizer
from src.agents.graph import build_graph, route_after_critic, run_pipeline
from src.llm.base import LLMError, LLMResponse, Message
from src.schemas import (
    ArticleSummary,
    Author,
    Article,
    CriticFeedback,
    ResearchGap,
    SourceType,
    Synthesis,
    Trend,
    TrendAnalysis,
)


# ============================================================
# Helpers
# ============================================================

def make_synthesis(overview: str = "Good overview.") -> Synthesis:
    return Synthesis(
        topic="fake news detection",
        overview=overview,
        main_approaches=["Transformers", "Graph NNs"],
        common_datasets=["LIAR"],
        key_findings=["BERT achieves 88% F1.", "Graph methods handle context."],
        article_count=3,
    )


def make_summary(idx: int = 1) -> ArticleSummary:
    return ArticleSummary(
        article_id=f"openalex_W{idx}",
        problem="Fake news classification.",
        method="BERT fine-tuning.",
        dataset="LIAR",
        results="88% F1.",
        limitations=None,
        key_contributions=["Baseline", "Dataset split"],
    )


def make_feedback(
    quality: str = "acceptable",
    needs_revision: bool = True,
    iteration: int = 1,
) -> CriticFeedback:
    return CriticFeedback(
        target="synthesis",
        iteration=iteration,
        overall_quality=quality,
        issues=["Overview too vague.", "No datasets mentioned."],
        suggestions=["Add LIAR dataset.", "Cite specific F1 scores."],
        needs_revision=needs_revision,
    )


class FakeLLM:
    def __init__(self, returns):
        self._queue = list(returns)
        self.calls = 0

    @property
    def provider(self): return "anthropic"

    @property
    def model(self): return "claude-sonnet-4-6"

    def chat(self, *a, **kw): raise NotImplementedError

    def chat_structured(self, system, messages, schema, **kw):
        self.calls += 1
        item = self._queue.pop(0)
        if isinstance(item, Exception):
            raise item
        resp = LLMResponse(
            content="{}", model="claude-sonnet-4-6", provider="anthropic",
            input_tokens=200, output_tokens=100, cost_usd=0.005,
        )
        return item, resp


# ============================================================
# Critic agent
# ============================================================

class TestCritic:

    def test_returns_feedback_with_needs_revision_true(self):
        feedback = make_feedback(quality="acceptable", needs_revision=True)
        llm = FakeLLM([feedback])
        state = {
            "synthesis": make_synthesis(),
            "summaries": [make_summary()],
            "synthesis_iteration": 1,
            "critic_feedbacks": [],
        }
        out = critic.run(state, llm=llm)

        assert len(out["critic_feedbacks"]) == 1
        assert out["critic_feedbacks"][0].needs_revision is True
        assert out["logs"][0].status == "success"
        assert llm.calls == 1

    def test_good_quality_sets_needs_revision_false(self):
        feedback = make_feedback(quality="good", needs_revision=False)
        llm = FakeLLM([feedback])
        state = {
            "synthesis": make_synthesis(),
            "summaries": [make_summary()],
            "synthesis_iteration": 1,
            "critic_feedbacks": [],
        }
        out = critic.run(state, llm=llm)
        assert out["critic_feedbacks"][0].needs_revision is False

    def test_iteration_stamp_overrides_llm_value(self):
        """The Critic stamps the iteration from state, not from LLM output."""
        feedback = make_feedback(iteration=99)   # LLM says 99
        llm = FakeLLM([feedback])
        state = {
            "synthesis": make_synthesis(),
            "summaries": [make_summary()],
            "synthesis_iteration": 2,   # real value
            "critic_feedbacks": [],
        }
        out = critic.run(state, llm=llm)
        assert out["critic_feedbacks"][0].iteration == 2

    def test_max_iterations_forces_approval_without_llm_call(self):
        llm = FakeLLM([])   # empty — should not be called
        state = {
            "synthesis": make_synthesis(),
            "summaries": [make_summary()],
            "synthesis_iteration": 3,   # = MAX_REFLEXION_ITERATIONS
            "critic_feedbacks": [],
        }
        out = critic.run(state, llm=llm)

        assert llm.calls == 0
        assert out["critic_feedbacks"][0].needs_revision is False

    def test_llm_failure_approves_as_fallback(self):
        llm = FakeLLM([LLMError("timeout")])
        state = {
            "synthesis": make_synthesis(),
            "summaries": [make_summary()],
            "synthesis_iteration": 1,
            "critic_feedbacks": [],
        }
        out = critic.run(state, llm=llm)

        # Never block the pipeline on LLM failure.
        assert out["critic_feedbacks"][0].needs_revision is False
        assert out["logs"][0].status == "failed"
        assert "errors" in out

    def test_no_synthesis_skips_llm(self):
        llm = FakeLLM([])
        out = critic.run(
            {"synthesis": None, "summaries": [], "synthesis_iteration": 0,
             "critic_feedbacks": []},
            llm=llm,
        )
        assert llm.calls == 0
        assert out["critic_feedbacks"] == []

    def test_quality_threshold_overrides_llm_needs_revision(self):
        """LLM may return needs_revision=False for 'acceptable', but config says
        we need at least 'good' — so needs_revision must be True."""
        feedback = make_feedback(quality="acceptable", needs_revision=False)
        llm = FakeLLM([feedback])
        state = {
            "synthesis": make_synthesis(),
            "summaries": [make_summary()],
            "synthesis_iteration": 1,
            "critic_feedbacks": [],
        }
        out = critic.run(state, llm=llm)
        # REFLEXION_MIN_QUALITY = "good" so "acceptable" → needs_revision=True
        assert out["critic_feedbacks"][0].needs_revision is True


# ============================================================
# Synthesizer — revision mode
# ============================================================

class TestSynthesizerReflexion:

    def test_first_run_no_feedback(self):
        synth = make_synthesis()
        llm = FakeLLM([synth])
        state = {
            "topic": "fake news detection",
            "summaries": [make_summary()],
            "synthesis": None,
            "synthesis_iteration": 0,
            "critic_feedbacks": [],
        }
        out = synthesizer.run(state, llm=llm)

        assert out["synthesis"] is not None
        assert out["synthesis_iteration"] == 1
        assert llm.calls == 1

    def test_revision_includes_feedback_in_prompt(self):
        """Check that on revision, the user message contains feedback keywords."""
        synth_v2 = make_synthesis(overview="Improved overview with LIAR dataset.")
        captured_messages = []

        class CaptureLLM:
            provider = "anthropic"
            model = "claude-sonnet-4-6"

            def chat_structured(self, system, messages, schema, **kw):
                captured_messages.extend(messages)
                resp = LLMResponse(
                    content="{}", model="claude-sonnet-4-6", provider="anthropic",
                    input_tokens=300, output_tokens=150, cost_usd=0.007,
                )
                return synth_v2, resp

        state = {
            "topic": "fake news detection",
            "summaries": [make_summary()],
            "synthesis": make_synthesis(overview="Vague overview."),
            "synthesis_iteration": 1,
            "critic_feedbacks": [make_feedback(quality="acceptable", needs_revision=True)],
        }
        out = synthesizer.run(state, llm=CaptureLLM())

        assert out["synthesis_iteration"] == 2
        # The user message must contain the critic feedback.
        user_content = captured_messages[0].content
        assert "CRITIC FEEDBACK" in user_content
        assert "Overview too vague." in user_content
        assert "Add LIAR dataset." in user_content

    def test_iteration_counter_increments_each_run(self):
        synth = make_synthesis()
        llm = FakeLLM([synth, make_synthesis("v2"), make_synthesis("v3")])
        base_state = {
            "topic": "t",
            "summaries": [make_summary()],
            "synthesis": None,
            "critic_feedbacks": [],
        }
        for expected_iteration in [1, 2, 3]:
            base_state["synthesis_iteration"] = expected_iteration - 1
            out = synthesizer.run(base_state, llm=llm)
            assert out["synthesis_iteration"] == expected_iteration
            base_state["synthesis"] = out["synthesis"]


# ============================================================
# route_after_critic
# ============================================================

class TestRouteAfterCritic:

    def test_routes_to_synthesizer_when_needs_revision(self):
        state = {
            "critic_feedbacks": [make_feedback(needs_revision=True)],
            "synthesis_iteration": 1,
        }
        assert route_after_critic(state) == "synthesizer"

    def test_routes_to_trends_when_approved(self):
        state = {
            "critic_feedbacks": [make_feedback(quality="good", needs_revision=False)],
            "synthesis_iteration": 1,
        }
        assert route_after_critic(state) == "trend_analyst"

    def test_routes_to_trends_when_max_iterations_reached(self):
        state = {
            "critic_feedbacks": [make_feedback(needs_revision=True)],
            "synthesis_iteration": 3,  # = MAX_REFLEXION_ITERATIONS
        }
        assert route_after_critic(state) == "trend_analyst"

    def test_routes_to_trends_when_no_feedbacks(self):
        state = {"critic_feedbacks": [], "synthesis_iteration": 0}
        assert route_after_critic(state) == "trend_analyst"


# ============================================================
# End-to-end: Reflexion loop
# ============================================================

class TestReflexionEndToEnd:

    def _make_article(self, idx: int) -> Article:
        # Titles and concepts must contain topic words ("fake", "news") so that
        # the MIN_RELEVANCE_SCORE gate in QualityCritic lets them through.
        return Article(
            id=f"openalex_W{idx}",
            title=f"Fake News Detection Using Deep Learning Method {idx}",
            abstract="We propose a method for detecting fake news on social media.",
            authors=[Author(name="A")],
            year=2023,
            source=SourceType.OPENALEX,
            citation_count=10,
            url=f"https://openalex.org/W{idx}",
            concepts=["fake news", "detection", "NLP"],
        )

    def test_loop_stops_after_one_revision(self):
        """Synthesizer runs twice: initial + 1 revision, then Critic approves."""
        synth_v1 = make_synthesis("Vague v1.")
        synth_v2 = make_synthesis("Good v2 with LIAR dataset and 88% F1.")
        feedback_revision = make_feedback(quality="acceptable", needs_revision=True, iteration=1)
        feedback_approved = make_feedback(quality="good", needs_revision=False, iteration=2)

        trend = TrendAnalysis(
            trends=[Trend(name="T", description="d", evidence_article_ids=["openalex_W1"],
                         maturity="emerging")],
            gaps=[ResearchGap(description="g", importance="medium",
                             suggested_directions=["d1"])],
            future_perspectives=["p1"],
        )
        from src.schemas import ReActThoughtAction as RAT
        expander_llm = FakeLLM([
            RAT(thought="Search main topic", action="search",
                search_query="fake news detection"),
            RAT(thought="Done", action="stop"),
        ])
        synth_llm    = FakeLLM([synth_v1, synth_v2])
        critic_llm   = FakeLLM([feedback_revision, feedback_approved])
        trend_llm    = FakeLLM([trend])
        summ_llm     = FakeLLM([make_summary(i) for i in range(1, 3)])

        def fake_get_llm(task):
            return {
                "query_expansion": expander_llm,
                "summarize":       summ_llm,
                "synthesize":      synth_llm,
                "critic":          critic_llm,
                "trend_analysis":  trend_llm,
            }[task]

        articles = [self._make_article(i) for i in range(1, 3)]

        with patch("src.agents.searcher.OpenAlexClient") as MockSearcher, \
             patch("src.agents.query_expander.OpenAlexClient") as MockExpander, \
             patch("src.agents.query_expander.get_llm_for_task", side_effect=fake_get_llm), \
             patch("src.agents.summarizer.get_llm_for_task",     side_effect=fake_get_llm), \
             patch("src.agents.synthesizer.get_llm_for_task",    side_effect=fake_get_llm), \
             patch("src.agents.critic.get_llm_for_task",         side_effect=fake_get_llm), \
             patch("src.agents.trend_analyst.get_llm_for_task",  side_effect=fake_get_llm):
            MockSearcher.return_value.search.return_value = articles
            MockExpander.return_value.search.return_value = articles[:1]  # ReAct probe
            final = run_pipeline("fake news detection", n_raw=2, top_n=2)

        # Synthesizer called twice (initial + 1 revision).
        assert synth_llm.calls == 2
        # Critic called twice (once per Synthesizer run).
        assert critic_llm.calls == 2
        assert final["synthesis_iteration"] == 2
        assert len(final["critic_feedbacks"]) == 2
        # Final synthesis is v2.
        assert "v2" in final["synthesis"].overview
        assert final["trend_analysis"] is not None

    def test_loop_stops_at_max_iterations_without_approval(self):
        """Even if Critic keeps saying 'poor', loop stops at MAX_REFLEXION_ITERATIONS."""
        synths = [make_synthesis(f"v{i}") for i in range(1, 5)]
        # Critic always says needs_revision=True, but 3rd call should be
        # forced-approved by the iteration cap (no LLM call).
        feedbacks = [
            make_feedback(quality="poor", needs_revision=True, iteration=i)
            for i in range(1, 4)
        ]
        from src.schemas import ReActThoughtAction as RAT
        trend = TrendAnalysis(
            trends=[Trend(name="T", description="d",
                         evidence_article_ids=["openalex_W1"], maturity="emerging")],
            gaps=[ResearchGap(description="g", importance="low",
                             suggested_directions=["d1"])],
            future_perspectives=["p1"],
        )

        expander_llm = FakeLLM([
            RAT(thought="Search main topic", action="search",
                search_query="fake news"),
            RAT(thought="Done", action="stop"),
        ])
        synth_llm    = FakeLLM(synths)
        critic_llm   = FakeLLM(feedbacks)   # only 2 real calls (3rd is skipped)
        trend_llm    = FakeLLM([trend])
        summ_llm     = FakeLLM([make_summary(1)])

        def fake_get_llm(task):
            return {
                "query_expansion": expander_llm,
                "summarize":       summ_llm,
                "synthesize":      synth_llm,
                "critic":          critic_llm,
                "trend_analysis":  trend_llm,
            }[task]

        articles = [self._make_article(1)]
        with patch("src.agents.searcher.OpenAlexClient") as MockSearcher, \
             patch("src.agents.query_expander.OpenAlexClient") as MockExpander, \
             patch("src.agents.query_expander.get_llm_for_task", side_effect=fake_get_llm), \
             patch("src.agents.summarizer.get_llm_for_task",     side_effect=fake_get_llm), \
             patch("src.agents.synthesizer.get_llm_for_task",    side_effect=fake_get_llm), \
             patch("src.agents.critic.get_llm_for_task",         side_effect=fake_get_llm), \
             patch("src.agents.trend_analyst.get_llm_for_task",  side_effect=fake_get_llm):
            MockSearcher.return_value.search.return_value = articles
            MockExpander.return_value.search.return_value = articles[:1]  # ReAct probe
            final = run_pipeline("fake news", n_raw=1, top_n=1)

        # Synthesizer ran 3 times (MAX_REFLEXION_ITERATIONS).
        assert synth_llm.calls == 3
        # Critic only made 2 real LLM calls; 3rd was forced-approved (no call).
        assert critic_llm.calls == 2
        assert final["synthesis_iteration"] == 3
        # Pipeline still produces trend_analysis despite never reaching "good".
        assert final["trend_analysis"] is not None
