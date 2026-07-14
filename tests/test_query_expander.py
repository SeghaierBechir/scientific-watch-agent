"""Tests for the ReAct QueryExpander agent and Searcher multi-query logic.

Covers the ReAct pattern:
    - LLM iterates: THINK → SEARCH (probe) → OBSERVE → repeat → STOP
    - Duplicate queries are skipped
    - Zero-result queries are not added to expanded_queries
    - Loop stops at MAX_REACT_ITERATIONS even if LLM keeps asking for search
    - LLM failure falls back to [topic]
    - react_steps are recorded in the output
    - The prompt grows with history at each iteration (context accumulation)
"""

from __future__ import annotations

from unittest.mock import patch, call

import pytest

from src.agents import query_expander, searcher
from src.config import MAX_REACT_ITERATIONS
from src.llm.base import LLMError, LLMResponse, Message
from src.schemas import Article, Author, ReActThoughtAction, SourceType


# ============================================================
# Helpers
# ============================================================

def make_article(idx: int, concepts: list[str] | None = None) -> Article:
    return Article(
        id=f"openalex_W{idx}",
        title=f"Paper {idx} about fake news",
        abstract="Abstract text.",
        authors=[Author(name=f"Author {idx}")],
        year=2023,
        source=SourceType.OPENALEX,
        citation_count=10 + idx,
        url=f"https://openalex.org/W{idx}",
        concepts=concepts or ["fake news", "NLP"],
    )


def _step(thought: str, action: str, query: str | None = None,
          stop_reason: str | None = None) -> ReActThoughtAction:
    return ReActThoughtAction(
        thought=thought,
        action=action,                   # type: ignore[arg-type]
        search_query=query,
        stop_reason=stop_reason,
    )


class FakeLLM:
    """Returns queued ReActThoughtAction objects one per chat_structured call."""

    def __init__(self, returns: list):
        self._queue = list(returns)
        self.calls = 0
        self.captured_prompts: list[str] = []   # for testing prompt content

    @property
    def provider(self): return "openai"
    @property
    def model(self):    return "gpt-4o-mini"

    def chat(self, *a, **kw): raise NotImplementedError

    def chat_structured(self, system, messages, schema, **kw):
        self.calls += 1
        self.captured_prompts.append(messages[0].content)
        item = self._queue.pop(0)
        if isinstance(item, Exception):
            raise item
        resp = LLMResponse(
            content="{}", model="gpt-4o-mini", provider="openai",
            input_tokens=80, output_tokens=40, cost_usd=0.0001,
        )
        return item, resp


# ============================================================
# ReAct QueryExpander
# ============================================================

class TestReActQueryExpander:

    def _run(self, llm_steps: list, search_side_effects: list | None = None):
        """Helper: patch OpenAlex and run query_expander with given LLM steps."""
        with patch("src.agents.query_expander.OpenAlexClient") as MockOA:
            if search_side_effects is not None:
                MockOA.return_value.search.side_effect = search_side_effects
            else:
                MockOA.return_value.search.return_value = []

            llm = FakeLLM(llm_steps)
            out = query_expander.run(
                {"topic": "fake news detection", "config": {"n_raw": 20}},
                llm=llm,
            )
        return out, llm, MockOA

    # ── Regression: correct parameter name ───────────────────────────────

    def test_probe_uses_n_results_not_n(self):
        """Regression: _search_probe must call search(n_results=…), not search(n=…).

        If the wrong kwarg is used, OpenAlexClient.search() raises TypeError,
        the probe silences it with [], queries never match, and only the
        original topic is returned — the exact bug we fixed.
        """
        arts = [make_article(1), make_article(2)]
        out, _, MockOA = self._run(
            llm_steps=[
                _step("Search topic", "search", "fake news detection"),
                _step("Try synonym",  "search", "misinformation NLP"),
                _step("Stop",        "stop"),
            ],
            search_side_effects=[arts, arts],
        )
        # The mock verifies the keyword argument used
        call_kwargs = MockOA.return_value.search.call_args_list[0].kwargs
        assert "n_results" in call_kwargs, (
            "probe called search(n=…) instead of search(n_results=…) — "
            "this silently returns [] and collapses all queries to [topic]"
        )
        assert "n" not in call_kwargs

    # ── Basic flow ────────────────────────────────────────────────────────

    def test_searches_then_stops(self):
        """Typical flow: 2 productive searches then stop."""
        arts1 = [make_article(1), make_article(2)]
        arts2 = [make_article(3)]
        out, llm, MockOA = self._run(
            llm_steps=[
                _step("Start with topic",     "search", "fake news detection"),
                _step("Try synonym",          "search", "misinformation NLP"),
                _step("Enough coverage",      "stop",   stop_reason="Two angles covered."),
            ],
            search_side_effects=[arts1, arts2],
        )

        assert llm.calls == 3
        assert MockOA.return_value.search.call_count == 2
        assert out["expanded_queries"] == ["fake news detection", "misinformation NLP"]
        assert out["logs"][0].status == "success"
        assert out["logs"][0].api_calls == 3

    def test_original_topic_always_in_result(self):
        """Even if LLM stops immediately, the original topic is added as fallback."""
        out, llm, _ = self._run(
            llm_steps=[
                _step("Skip directly", "stop", stop_reason="Topic is clear enough."),
            ],
        )
        # LLM never searched → queries_found is empty → topic inserted automatically
        assert "fake news detection" in out["expanded_queries"]

    def test_stops_at_max_iterations_without_stop_action(self):
        """Loop exits after MAX_REACT_ITERATIONS even if LLM never says stop."""
        # Prepare enough search steps to fill the loop
        search_steps = [
            _step(f"Try angle {i}", "search", f"query {i}")
            for i in range(MAX_REACT_ITERATIONS)
        ]
        articles_per_query = [[make_article(i)] for i in range(MAX_REACT_ITERATIONS)]

        out, llm, MockOA = self._run(
            llm_steps=search_steps,
            search_side_effects=articles_per_query,
        )

        assert llm.calls == MAX_REACT_ITERATIONS
        assert MockOA.return_value.search.call_count == MAX_REACT_ITERATIONS
        assert len(out["expanded_queries"]) <= MAX_REACT_ITERATIONS + 1  # +1 for forced topic

    # ── Duplicate handling ────────────────────────────────────────────────

    def test_duplicate_query_not_re_searched(self):
        """If LLM repeats the same query, OpenAlex is called only once for it."""
        arts = [make_article(1)]
        out, llm, MockOA = self._run(
            llm_steps=[
                _step("First try",    "search", "fake news detection"),
                _step("Same again",   "search", "fake news detection"),   # duplicate
                _step("Stop now",     "stop"),
            ],
            search_side_effects=[arts],   # only called once
        )

        assert MockOA.return_value.search.call_count == 1
        # Appears only once in results
        assert out["expanded_queries"].count("fake news detection") == 1

    def test_duplicate_case_insensitive(self):
        """'Fake News Detection' and 'fake news detection' are the same query."""
        arts = [make_article(1)]
        out, llm, MockOA = self._run(
            llm_steps=[
                _step("Lower case",  "search", "fake news detection"),
                _step("Upper case",  "search", "Fake News Detection"),   # same query
                _step("Stop",        "stop"),
            ],
            search_side_effects=[arts],
        )
        assert MockOA.return_value.search.call_count == 1

    # ── Zero-result handling ──────────────────────────────────────────────

    def test_zero_result_query_not_added_to_expanded(self):
        """A query that returns 0 articles is NOT added to expanded_queries."""
        out, llm, _ = self._run(
            llm_steps=[
                _step("Empty angle", "search", "obscure query xyz"),
                _step("Stop",        "stop"),
            ],
            search_side_effects=[[]],   # 0 articles
        )
        assert "obscure query xyz" not in out["expanded_queries"]
        # Topic added as fallback
        assert "fake news detection" in out["expanded_queries"]

    # ── LLM failure ───────────────────────────────────────────────────────

    def test_llm_failure_on_first_call_falls_back(self):
        """Complete LLM failure → fallback to [topic], status=failed."""
        out, llm, _ = self._run(
            llm_steps=[LLMError("timeout")],
        )
        assert out["expanded_queries"] == ["fake news detection"]
        assert out["logs"][0].status == "failed"
        assert "errors" in out

    def test_llm_failure_mid_loop_uses_what_was_found(self):
        """If LLM fails after 1 successful search, keep that query."""
        arts = [make_article(1)]
        out, llm, _ = self._run(
            llm_steps=[
                _step("First search", "search", "fake news detection"),
                LLMError("timeout"),   # fails on 2nd iteration
            ],
            search_side_effects=[arts],
        )
        assert "fake news detection" in out["expanded_queries"]
        # status=success because at least 1 LLM call worked
        assert out["logs"][0].status == "success"

    # ── react_steps output ────────────────────────────────────────────────

    def test_react_steps_recorded(self):
        """react_steps must contain one entry per iteration."""
        arts = [make_article(1)]
        out, _, _ = self._run(
            llm_steps=[
                _step("Search", "search", "fake news detection"),
                _step("Stop",   "stop",   stop_reason="done"),
            ],
            search_side_effects=[arts],
        )
        steps = out["react_steps"]
        assert len(steps) == 2
        assert steps[0]["action"] == "search"
        assert steps[0]["iteration"] == 1
        assert steps[0]["query"] == "fake news detection"
        assert steps[1]["action"] == "stop"
        assert steps[1]["stop_reason"] == "done"

    def test_react_steps_contain_observation(self):
        """Each search step must have an 'observation' key with article info."""
        arts = [make_article(1, concepts=["fake news", "BERT"])]
        out, _, _ = self._run(
            llm_steps=[
                _step("Search", "search", "fake news detection"),
                _step("Stop",   "stop"),
            ],
            search_side_effects=[arts],
        )
        obs = out["react_steps"][0]["observation"]
        assert "articles found" in obs
        # Concepts should appear in the observation
        assert "fake news" in obs or "BERT" in obs

    # ── Prompt accumulates history ─────────────────────────────────────────

    def test_prompt_grows_with_history(self):
        """The user message at iteration 2 must include iteration 1's thought."""
        arts = [make_article(1)]
        _, llm, _ = self._run(
            llm_steps=[
                _step("My first thought", "search", "fake news detection"),
                _step("Stop now",         "stop"),
            ],
            search_side_effects=[arts],
        )
        # llm.captured_prompts[0] = prompt for iter 1 (no history yet)
        # llm.captured_prompts[1] = prompt for iter 2 (must include iter 1)
        assert len(llm.captured_prompts) == 2
        assert "My first thought" in llm.captured_prompts[1]
        assert "fake news detection" in llm.captured_prompts[1]

    def test_log_tokens_accumulated_across_iterations(self):
        """Total tokens = sum over all LLM calls (80+40 per call from FakeLLM)."""
        arts = [make_article(1)]
        out, llm, _ = self._run(
            llm_steps=[
                _step("Search", "search", "fake news detection"),
                _step("Stop",   "stop"),
            ],
            search_side_effects=[arts],
        )
        # 2 LLM calls × 120 tokens each = 240
        assert out["logs"][0].tokens_used == 240
        assert out["logs"][0].api_calls == 2


# ============================================================
# Searcher — multi-query logic (unchanged, kept as regression)
# ============================================================

class TestSearcherMultiQuery:

    def test_searches_each_expanded_query(self):
        articles_q1 = [make_article(1), make_article(2)]
        articles_q2 = [make_article(3), make_article(4)]

        with patch("src.agents.searcher.OpenAlexClient") as MockClient:
            MockClient.return_value.search.side_effect = [articles_q1, articles_q2]
            out = searcher.run({
                "topic": "fake news",
                "config": {"n_raw": 20, "from_year": 2020},
                "expanded_queries": ["fake news detection", "misinformation NLP"],
            })

        assert MockClient.return_value.search.call_count == 2
        assert len(out["raw_articles"]) == 4
        assert out["logs"][0].api_calls == 2

    def test_deduplicates_across_queries(self):
        shared = make_article(1)
        articles_q1 = [shared, make_article(2)]
        articles_q2 = [shared, make_article(3)]

        with patch("src.agents.searcher.OpenAlexClient") as MockClient:
            MockClient.return_value.search.side_effect = [articles_q1, articles_q2]
            out = searcher.run({
                "topic": "fake news",
                "config": {"n_raw": 20, "from_year": 2020},
                "expanded_queries": ["q1", "q2"],
            })

        ids = [a.id for a in out["raw_articles"]]
        assert len(ids) == len(set(ids))
        assert len(out["raw_articles"]) == 3

    def test_falls_back_to_topic_when_no_expanded_queries(self):
        articles = [make_article(1)]
        with patch("src.agents.searcher.OpenAlexClient") as MockClient:
            MockClient.return_value.search.return_value = articles
            out = searcher.run({
                "topic": "fake news",
                "config": {"n_raw": 10, "from_year": 2020},
            })

        MockClient.return_value.search.assert_called_once()
        assert MockClient.return_value.search.call_args.kwargs["query"] == "fake news"

    def test_partial_query_failure_continues(self):
        from src.sources.openalex import OpenAlexError
        articles_q2 = [make_article(5), make_article(6)]
        with patch("src.agents.searcher.OpenAlexClient") as MockClient:
            MockClient.return_value.search.side_effect = [
                OpenAlexError("500"),
                articles_q2,
            ]
            out = searcher.run({
                "topic": "fake news",
                "config": {"n_raw": 20, "from_year": 2020},
                "expanded_queries": ["q1", "q2"],
            })

        assert len(out["raw_articles"]) == 2
        assert out["logs"][0].status == "success"

    def test_empty_expanded_queries_uses_topic(self):
        articles = [make_article(1)]
        with patch("src.agents.searcher.OpenAlexClient") as MockClient:
            MockClient.return_value.search.return_value = articles
            out = searcher.run({
                "topic": "NLP",
                "config": {"n_raw": 10, "from_year": 2020},
                "expanded_queries": [],
            })

        MockClient.return_value.search.assert_called_once()
        assert out["raw_articles"] == articles
