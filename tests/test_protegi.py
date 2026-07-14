"""Tests for the ProTeGi optimization module.

Tests are structured in three classes:
    TestSummaryMetrics    -- unit tests for summary_metrics.py (ROUGE + flatten)
    TestLLMJudge          -- unit tests for LLM Judge (with mocked LLM)
    TestProTeGiOptimizer  -- unit tests for ProTeGi algorithm (with mocked LLMs)
    TestProTeGiResult     -- Pydantic schema and serialization tests

No real LLM calls. All LLM interactions are mocked with pytest-mock.
No real PubMed download. Examples are constructed inline.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.optimization.summary_metrics import (
    JudgeOutput,
    compute_all_metrics,
    compute_rouge,
    flatten_summary,
)
from src.optimization.protegi_optimizer import (
    Example,
    EvalResult,
    ProTeGiOptimizer,
    ProTeGiResult,
    _mean,
)
from src.schemas import ArticleSummary


# ============================================================
# Helpers
# ============================================================


def _make_summary(**overrides) -> ArticleSummary:
    """Build an ArticleSummary with sensible defaults."""
    defaults = {
        "article_id": "test_1",
        "problem": "We study fake news detection.",
        "method": "We use a BERT-based classifier.",
        "results": "Achieved 95% accuracy on FakeNewsNet.",
        "dataset": "FakeNewsNet",
        "limitations": "Only tested on English news.",
        "key_contributions": [
            "Novel BERT adaptation for fake news",
            "New evaluation benchmark",
        ],
    }
    defaults.update(overrides)
    return ArticleSummary(**defaults)


def _make_example(
    article_text: str = "A study on neural machine translation.",
    reference: str = "Translation accuracy improved by 5 BLEU points.",
    article_id: str = "ex_1",
) -> Example:
    return Example(
        article_text=article_text,
        reference=reference,
        article_id=article_id,
    )


def _mock_llm_for_summary(summary: ArticleSummary):
    """Return a mock LLMClient that returns `summary` from chat_structured."""
    from src.llm.base import LLMResponse

    mock_resp = LLMResponse(
        content="",
        model="gpt-4o-mini",
        provider="openai",
        input_tokens=100,
        output_tokens=80,
        cost_usd=0.0001,
    )
    mock_llm = MagicMock()
    mock_llm.chat_structured.return_value = (summary, mock_resp)
    return mock_llm


def _mock_llm_for_judge(faithfulness: float = 0.8, coverage: float = 0.7):
    """Return a mock LLMClient for the LLM Judge."""
    from src.llm.base import LLMResponse

    judge_out = JudgeOutput(
        faithfulness=faithfulness,
        coverage=coverage,
        reasoning="Test reasoning.",
    )
    mock_resp = LLMResponse(
        content="",
        model="gpt-4o-mini",
        provider="openai",
        input_tokens=200,
        output_tokens=50,
        cost_usd=0.0002,
    )
    mock_llm = MagicMock()
    mock_llm.chat_structured.return_value = (judge_out, mock_resp)
    return mock_llm


def _mock_llm_for_gradient(gradient_text: str = "The prompt fails because..."):
    """Return a mock LLMClient for gradient generation."""
    from src.llm.base import LLMResponse

    mock_resp = LLMResponse(
        content=gradient_text,
        model="gpt-4o-mini",
        provider="openai",
        input_tokens=300,
        output_tokens=120,
        cost_usd=0.0003,
    )
    mock_llm = MagicMock()
    mock_llm.chat.return_value = mock_resp
    return mock_llm


def _mock_llm_for_candidates(candidates: list[str]):
    """Return a mock LLMClient that returns candidate prompts as JSON."""
    from src.llm.base import LLMResponse

    mock_resp = LLMResponse(
        content=json.dumps(candidates),
        model="gpt-4o-mini",
        provider="openai",
        input_tokens=500,
        output_tokens=400,
        cost_usd=0.0005,
    )
    mock_llm = MagicMock()
    mock_llm.chat.return_value = mock_resp
    return mock_llm


# ============================================================
# TestSummaryMetrics
# ============================================================


class TestSummaryMetrics:
    """Unit tests for ROUGE computation and flatten_summary."""

    # ── Fallback ROUGE (built-in, no external library) ─────────────────────

    def test_rouge_builtin_fallback_works_without_library(self):
        """The built-in ROUGE fallback must work even when rouge-score is absent."""
        import sys
        import importlib
        from src.optimization.summary_metrics import _compute_rouge_builtin

        # Call the fallback directly — should never raise ImportError
        scores = _compute_rouge_builtin(
            "attention mechanism transformer architecture",
            "transformer model uses attention mechanism",
        )
        assert set(scores.keys()) == {"rouge1", "rouge2", "rougeL"}
        assert scores["rouge1"] > 0.4   # "attention", "mechanism", "transformer" match
        assert 0.0 <= scores["rouge2"] <= 1.0
        assert 0.0 <= scores["rougeL"] <= 1.0

    def test_rouge_fallback_empty_inputs(self):
        """Fallback returns zeros for empty strings."""
        from src.optimization.summary_metrics import _compute_rouge_builtin
        scores = _compute_rouge_builtin("", "reference text")
        assert scores == {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    def test_rouge_fallback_perfect_match(self):
        """Identical texts give ROUGE-L = 1.0 in the fallback."""
        from src.optimization.summary_metrics import _compute_rouge_builtin
        text = "deep learning model for fake news detection"
        scores = _compute_rouge_builtin(text, text)
        assert scores["rougeL"] > 0.99

    def test_compute_rouge_does_not_raise_when_library_absent(self, monkeypatch):
        """compute_rouge() must fall back silently — no ImportError to caller."""
        import builtins
        real_import = builtins.__import__

        def _block_rouge_score(name, *args, **kwargs):
            if name == "rouge_score":
                raise ImportError("simulated: rouge-score not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _block_rouge_score)

        # Should NOT raise — falls back to built-in
        scores = compute_rouge("BERT model achieves high accuracy", "high accuracy BERT")
        assert "rouge1" in scores
        assert scores["rouge1"] > 0.3

    def test_rouge_fallback_ranking_preserved(self):
        """Fallback must preserve ranking: more-similar texts score higher."""
        from src.optimization.summary_metrics import _compute_rouge_builtin
        reference = "deep learning for fake news detection"
        high = _compute_rouge_builtin("deep learning fake news detection model", reference)
        low  = _compute_rouge_builtin("quantum computing entanglement", reference)
        assert high["rouge1"] > low["rouge1"]
        assert high["rougeL"] > low["rougeL"]

    # ── Standard ROUGE tests (use whatever implementation is available) ─────

    def test_rouge_perfect_match(self):
        """Identical generated and reference should give ROUGE scores close to 1."""
        text = "deep learning model for fake news detection accuracy"
        scores = compute_rouge(text, text)
        assert scores["rouge1"] > 0.99
        assert scores["rougeL"] > 0.99

    def test_rouge_no_overlap(self):
        """Completely different texts should give low ROUGE scores."""
        generated = "quantum computing entanglement superposition"
        reference = "social media misinformation viral spread"
        scores = compute_rouge(generated, reference)
        assert scores["rouge1"] < 0.2
        assert scores["rouge2"] == 0.0

    def test_rouge_partial_overlap(self):
        """Partial overlap should produce intermediate scores."""
        generated = "deep learning model for fake news"
        reference = "deep neural network for fake news detection"
        scores = compute_rouge(generated, reference)
        assert 0.3 < scores["rouge1"] < 0.9
        assert 0.0 < scores["rougeL"] < 0.9

    def test_rouge_empty_generated(self):
        """Empty generated text should give 0.0 scores."""
        scores = compute_rouge("", "some reference text here")
        assert scores["rouge1"] == 0.0
        assert scores["rouge2"] == 0.0
        assert scores["rougeL"] == 0.0

    def test_rouge_empty_reference(self):
        """Empty reference should give 0.0 scores."""
        scores = compute_rouge("some generated text", "")
        assert scores["rouge1"] == 0.0

    def test_rouge_returns_all_three_keys(self):
        """compute_rouge must always return rouge1, rouge2, rougeL."""
        scores = compute_rouge("test text", "reference text")
        assert set(scores.keys()) == {"rouge1", "rouge2", "rougeL"}

    def test_rouge_scores_in_range(self):
        """All ROUGE scores must be in [0, 1]."""
        scores = compute_rouge(
            "attention mechanism transformer architecture self-attention",
            "transformer model with multi-head attention and position encoding",
        )
        for key, val in scores.items():
            assert 0.0 <= val <= 1.0, f"{key}={val} out of range"

    @pytest.mark.parametrize("generated,reference,expected_r1", [
        # Stemming: "running" and "run" should match with use_stemmer=True
        ("running algorithm detection", "run algorithm for detection", 0.5),
        # Order doesn't matter for ROUGE-1 (bag of words)
        ("detection news fake", "fake news detection", 0.9),
    ])
    def test_rouge_stemming_and_order(self, generated, reference, expected_r1):
        scores = compute_rouge(generated, reference)
        # These are approximate thresholds, not exact
        assert scores["rouge1"] > expected_r1 * 0.8

    # ── flatten_summary ────────────────────────────────────────────────────

    def test_flatten_summary_all_fields(self):
        """All fields should appear in the flattened text."""
        summary = _make_summary()
        flat = flatten_summary(summary)
        assert "fake news detection" in flat.lower()
        assert "BERT" in flat or "bert" in flat.lower()
        assert "95%" in flat
        assert "FakeNewsNet" in flat
        assert "English news" in flat
        assert "benchmark" in flat.lower()

    def test_flatten_summary_null_optional_fields(self):
        """None fields should be silently skipped."""
        summary = _make_summary(dataset=None, limitations=None)
        flat = flatten_summary(summary)
        assert "Dataset:" not in flat
        assert "Limitations:" not in flat
        assert len(flat) > 0

    def test_flatten_summary_not_specified_excluded(self):
        """'Not specified' (LLM placeholder) should not appear in output."""
        summary = _make_summary(dataset="Not specified", limitations="Not specified")
        flat = flatten_summary(summary)
        assert "not specified" not in flat.lower()

    def test_flatten_summary_empty_contributions(self):
        """Empty key_contributions list should produce no crash."""
        summary = _make_summary(key_contributions=[])
        flat = flatten_summary(summary)
        assert len(flat) > 0  # other fields still present

    def test_flatten_summary_returns_string(self):
        summary = _make_summary()
        assert isinstance(flatten_summary(summary), str)


# ============================================================
# TestLLMJudge
# ============================================================


class TestLLMJudge:
    """Unit tests for the LLM Judge (mocked LLM calls)."""

    def test_judge_returns_expected_keys(self):
        """compute_llm_judge must return faithfulness, coverage, reasoning, cost."""
        from src.optimization.summary_metrics import compute_llm_judge

        mock_llm = _mock_llm_for_judge(faithfulness=0.9, coverage=0.8)
        result = compute_llm_judge(
            generated="BERT model achieves 95% accuracy on FakeNewsNet.",
            article_text="We propose a BERT model for fake news detection...",
            llm=mock_llm,
        )
        assert "faithfulness" in result
        assert "coverage" in result
        assert "reasoning" in result
        assert "judge_cost_usd" in result

    def test_judge_scores_in_range(self):
        """Faithfulness and coverage must be in [0, 1]."""
        from src.optimization.summary_metrics import compute_llm_judge

        mock_llm = _mock_llm_for_judge(faithfulness=0.7, coverage=0.6)
        result = compute_llm_judge("summary", "article", mock_llm)
        assert 0.0 <= result["faithfulness"] <= 1.0
        assert 0.0 <= result["coverage"] <= 1.0

    def test_judge_error_returns_neutral(self):
        """If LLM Judge raises an exception, return neutral scores (0.5)."""
        from src.optimization.summary_metrics import compute_llm_judge

        mock_llm = MagicMock()
        mock_llm.chat_structured.side_effect = RuntimeError("API timeout")

        result = compute_llm_judge("summary", "article", mock_llm)
        assert result["faithfulness"] == 0.5
        assert result["coverage"] == 0.5

    def test_compute_all_metrics_with_judge(self):
        """compute_all_metrics with a Judge LLM includes judge fields."""
        mock_llm = _mock_llm_for_judge(faithfulness=0.85, coverage=0.75)
        metrics = compute_all_metrics(
            generated="Model achieves 92% F1 on SemEval.",
            reference="SemEval benchmark with 92% F1 score.",
            article_text="Full article text here...",
            llm=mock_llm,
        )
        assert "faithfulness" in metrics
        assert "coverage" in metrics
        assert "composite" in metrics
        # Composite includes faithfulness when judge is present
        assert metrics["composite"] > 0.0

    def test_compute_all_metrics_without_judge(self):
        """compute_all_metrics without Judge uses ROUGE-only composite."""
        metrics = compute_all_metrics(
            generated="Model achieves 92% F1 on SemEval.",
            reference="SemEval benchmark with 92% F1 score.",
        )
        assert "rouge1" in metrics
        assert "composite" in metrics
        assert "faithfulness" not in metrics

    def test_composite_formula_with_judge(self):
        """Verify the composite formula: 0.4*ROUGE-L + 0.4*faithful + 0.2*coverage."""
        mock_llm = _mock_llm_for_judge(faithfulness=0.8, coverage=0.6)
        text = "identical reference text"  # ROUGE-L ~ 1.0
        metrics = compute_all_metrics(
            generated=text,
            reference=text,
            article_text="source",
            llm=mock_llm,
        )
        expected_composite = 0.40 * metrics["rougeL"] + 0.40 * 0.8 + 0.20 * 0.6
        assert abs(metrics["composite"] - expected_composite) < 0.01

    def test_composite_formula_without_judge(self):
        """Verify ROUGE-only composite = mean(R1, R2, RL)."""
        metrics = compute_all_metrics(
            generated="fake news detection deep learning",
            reference="deep learning for fake news detection",
        )
        expected = (metrics["rouge1"] + metrics["rouge2"] + metrics["rougeL"]) / 3
        assert abs(metrics["composite"] - expected) < 0.001


# ============================================================
# TestProTeGiOptimizer
# ============================================================


class TestProTeGiOptimizer:
    """Unit tests for the ProTeGi algorithm using mocked LLMs."""

    def _make_optimizer(
        self,
        summary: ArticleSummary | None = None,
        gradient_text: str = "The prompt fails because it lacks numerical details.",
        candidates: list[str] | None = None,
        faithfulness: float = 0.8,
        coverage: float = 0.7,
    ) -> ProTeGiOptimizer:
        """Build an optimizer with all LLMs mocked."""
        if summary is None:
            summary = _make_summary()
        if candidates is None:
            candidates = [
                "Improved prompt version 1: extract numerical results explicitly.",
                "Improved prompt version 2: focus on methods and datasets.",
                "Improved prompt version 3: require citations from abstract.",
                "Improved prompt version 4: be concise and structured.",
            ]

        # Build a single LLM mock that handles both chat and chat_structured
        mock_llm = MagicMock()

        # For summarization (chat_structured returns ArticleSummary)
        from src.llm.base import LLMResponse
        summary_resp = LLMResponse(
            content="",
            model="gpt-4o-mini",
            provider="openai",
            input_tokens=100,
            output_tokens=80,
            cost_usd=0.0001,
        )

        # For judge (chat_structured returns JudgeOutput)
        judge_out = JudgeOutput(
            faithfulness=faithfulness,
            coverage=coverage,
            reasoning="Test.",
        )
        judge_resp = LLMResponse(
            content="",
            model="gpt-4o-mini",
            provider="openai",
            input_tokens=150,
            output_tokens=50,
            cost_usd=0.0001,
        )

        # chat_structured returns different things depending on schema argument
        def _chat_structured_side_effect(system, messages, schema, **kwargs):
            if schema is ArticleSummary:
                return (summary, summary_resp)
            if schema is JudgeOutput:
                return (judge_out, judge_resp)
            raise ValueError(f"Unexpected schema: {schema}")

        mock_llm.chat_structured.side_effect = _chat_structured_side_effect

        # For gradient and candidate generation (chat returns text)
        def _chat_side_effect(system, messages, **kwargs):
            if system == "GRADIENT_SYSTEM" or "gradient" in system.lower() or "Gradient" in system:
                return LLMResponse(
                    content=gradient_text,
                    model="gpt-4o-mini",
                    provider="openai",
                    input_tokens=200,
                    output_tokens=100,
                    cost_usd=0.0002,
                )
            # Candidate generation
            return LLMResponse(
                content=json.dumps(candidates),
                model="gpt-4o-mini",
                provider="openai",
                input_tokens=400,
                output_tokens=300,
                cost_usd=0.0004,
            )

        mock_llm.chat.side_effect = _chat_side_effect

        return ProTeGiOptimizer(
            summarize_llm=mock_llm,
            gradient_llm=mock_llm,
            judge_llm=mock_llm,
            n_iterations=2,
            beam_size=2,
            n_worst=2,
            use_llm_judge=True,
        )

    def _make_examples(self, n: int = 5, prefix: str = "test") -> list[Example]:
        return [
            Example(
                article_text=f"Article {i}: Study on transformers and attention.",
                reference=f"Summary {i}: Transformers improve NLP by 5% on GLUE.",
                article_id=f"{prefix}_{i}",
            )
            for i in range(n)
        ]

    # ── _evaluate_set tests ────────────────────────────────────────────────

    def test_evaluate_set_returns_one_result_per_example(self):
        optimizer = self._make_optimizer()
        examples = self._make_examples(n=3)
        results = optimizer._evaluate_set("Test prompt", examples)
        assert len(results) == 3

    def test_evaluate_set_result_structure(self):
        optimizer = self._make_optimizer()
        examples = self._make_examples(n=2)
        results = optimizer._evaluate_set("Test prompt", examples)
        for r in results:
            assert isinstance(r, EvalResult)
            assert 0.0 <= r.composite <= 1.0
            assert 0.0 <= r.rouge1 <= 1.0
            assert 0.0 <= r.rougeL <= 1.0

    def test_evaluate_set_uses_article_id(self):
        optimizer = self._make_optimizer()
        examples = self._make_examples(n=2, prefix="check")
        results = optimizer._evaluate_set("Test prompt", examples)
        ids = {r.example_id for r in results}
        assert ids == {"check_0", "check_1"}

    def test_evaluate_set_graceful_on_exception(self):
        """If summarize_llm raises, result should have composite=0.0 and error set."""
        mock_llm = MagicMock()
        mock_llm.chat_structured.side_effect = RuntimeError("timeout")
        mock_llm.chat.return_value = MagicMock(content="fallback", model="x", provider="openai")

        optimizer = ProTeGiOptimizer(
            summarize_llm=mock_llm,
            gradient_llm=mock_llm,
            n_iterations=1,
            beam_size=1,
            use_llm_judge=False,
        )
        examples = self._make_examples(n=2)
        results = optimizer._evaluate_set("Prompt", examples)
        assert len(results) == 2
        for r in results:
            assert r.composite == 0.0
            assert r.error is not None

    # ── _generate_gradient tests ───────────────────────────────────────────

    def test_generate_gradient_returns_string(self):
        optimizer = self._make_optimizer(
            gradient_text="The prompt lacks specificity about results."
        )
        worst = [
            EvalResult("t1", composite=0.1, rouge1=0.1, rouge2=0.0, rougeL=0.1),
            EvalResult("t2", composite=0.15, rouge1=0.15, rouge2=0.05, rougeL=0.12),
        ]
        gradient = optimizer._generate_gradient("Test prompt", worst, [])
        assert isinstance(gradient, str)
        assert len(gradient) > 10

    def test_generate_gradient_fallback_on_error(self):
        """If gradient LLM fails, return a non-empty fallback string."""
        mock_llm = MagicMock()
        mock_llm.chat.side_effect = RuntimeError("API error")

        optimizer = ProTeGiOptimizer(
            summarize_llm=mock_llm,
            gradient_llm=mock_llm,
            n_iterations=1,
            beam_size=1,
        )
        worst = [EvalResult("t1", composite=0.1, rouge1=0.1, rouge2=0.0, rougeL=0.1)]
        gradient = optimizer._generate_gradient("prompt", worst, [])
        assert isinstance(gradient, str)
        assert len(gradient) > 10

    # ── _generate_candidates tests ─────────────────────────────────────────

    def test_generate_candidates_returns_list(self):
        candidates = [
            "Candidate prompt A: focus on methodology details.",
            "Candidate prompt B: extract quantitative results.",
        ]
        optimizer = self._make_optimizer(candidates=candidates)
        result = optimizer._generate_candidates("Initial prompt", "Gradient text")
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_generate_candidates_respects_beam_size(self):
        candidates = ["P1: detailed", "P2: concise", "P3: structured", "P4: faithful"]
        optimizer = self._make_optimizer(candidates=candidates)
        optimizer.beam_size = 2  # only keep 2
        result = optimizer._generate_candidates("prompt", "gradient")
        assert len(result) <= 2

    def test_generate_candidates_fallback_on_invalid_json(self):
        """If the LLM returns invalid JSON, return the current prompt as fallback."""
        from src.llm.base import LLMResponse
        mock_llm = MagicMock()
        mock_llm.chat.return_value = LLMResponse(
            content="not valid json at all !!!",
            model="gpt-4o-mini",
            provider="openai",
        )
        optimizer = ProTeGiOptimizer(
            summarize_llm=mock_llm,
            gradient_llm=mock_llm,
            n_iterations=1,
            beam_size=2,
        )
        result = optimizer._generate_candidates("current prompt", "gradient")
        assert result == ["current prompt"]

    # ── optimize() end-to-end tests ───────────────────────────────────────

    def test_optimize_returns_protegi_result(self):
        optimizer = self._make_optimizer()
        train = self._make_examples(n=4, prefix="tr")
        val = self._make_examples(n=3, prefix="va")
        result = optimizer.optimize("Initial system prompt", train, val)
        assert isinstance(result, ProTeGiResult)

    def test_optimize_result_has_best_prompt(self):
        optimizer = self._make_optimizer()
        train = self._make_examples(n=4)
        val = self._make_examples(n=3)
        result = optimizer.optimize("Initial prompt", train, val)
        assert isinstance(result.best_prompt, str)
        assert len(result.best_prompt) > 10

    def test_optimize_records_iteration_history(self):
        optimizer = self._make_optimizer()
        train = self._make_examples(n=4)
        val = self._make_examples(n=3)
        result = optimizer.optimize("Initial prompt", train, val)
        assert len(result.iterations) == optimizer.n_iterations
        for it in result.iterations:
            assert "iteration" in it
            assert "train_composite" in it
            assert "best_candidate_composite" in it

    def test_optimize_saves_to_disk(self, tmp_path):
        optimizer = self._make_optimizer()
        train = self._make_examples(n=3)
        val = self._make_examples(n=2)
        result = optimizer.optimize("Initial prompt", train, val, save_dir=tmp_path)
        assert result.prompt_saved_to is not None
        saved = Path(result.prompt_saved_to)
        assert saved.exists()
        # Also check the metadata JSON was saved
        json_files = list(tmp_path.glob("protegi_results_*.json"))
        assert len(json_files) == 1

    def test_optimize_without_save(self):
        optimizer = self._make_optimizer()
        train = self._make_examples(n=3)
        val = self._make_examples(n=2)
        result = optimizer.optimize("Initial prompt", train, val, save_dir=None)
        assert result.prompt_saved_to is None

    # ── no-judge mode ─────────────────────────────────────────────────────

    def test_optimize_no_judge_mode(self):
        """Optimizer should work correctly without LLM Judge."""
        from src.llm.base import LLMResponse
        summary = _make_summary()
        mock_llm = MagicMock()
        resp = LLMResponse(
            content="", model="gpt-4o-mini", provider="openai"
        )
        mock_llm.chat_structured.return_value = (summary, resp)
        mock_llm.chat.return_value = LLMResponse(
            content=json.dumps([
                "Prompt A: extract numerical findings.",
                "Prompt B: focus on contributions."
            ]),
            model="gpt-4o-mini",
            provider="openai",
        )

        optimizer = ProTeGiOptimizer(
            summarize_llm=mock_llm,
            gradient_llm=mock_llm,
            n_iterations=1,
            beam_size=2,
            use_llm_judge=False,  # key: no judge
        )
        train = self._make_examples(n=3)
        val = self._make_examples(n=2)
        result = optimizer.optimize("Initial prompt", train, val)
        assert isinstance(result, ProTeGiResult)
        # Judge should never have been called (use_llm_judge=False)
        # chat_structured should only be called for summarization (ArticleSummary schema)
        for call in mock_llm.chat_structured.call_args_list:
            assert call.kwargs.get("schema") is ArticleSummary or \
                   call[1].get("schema") is ArticleSummary or \
                   call[0][2] is ArticleSummary  # positional args


# ============================================================
# TestProTeGiResult
# ============================================================


class TestProTeGiResult:
    """Pydantic schema tests for ProTeGiResult."""

    def _make_result(self, **overrides) -> ProTeGiResult:
        defaults = {
            "best_prompt": "Optimized system prompt text",
            "initial_prompt": "Initial system prompt text",
            "n_iterations": 3,
            "initial_composite": 0.35,
            "final_composite": 0.48,
            "improvement_pct": 37.1,
            "iterations": [
                {"iteration": 1, "train_composite": 0.33, "improved": True},
                {"iteration": 2, "train_composite": 0.40, "improved": False},
                {"iteration": 3, "train_composite": 0.45, "improved": True},
            ],
            "duration_seconds": 45.2,
        }
        defaults.update(overrides)
        return ProTeGiResult(**defaults)

    def test_result_creation(self):
        result = self._make_result()
        assert result.best_prompt == "Optimized system prompt text"
        assert result.n_iterations == 3
        assert result.improvement_pct == 37.1

    def test_result_json_roundtrip(self):
        result = self._make_result()
        json_str = result.model_dump_json()
        reloaded = ProTeGiResult.model_validate_json(json_str)
        assert reloaded.best_prompt == result.best_prompt
        assert reloaded.initial_composite == result.initial_composite

    def test_composite_bounds_validation(self):
        with pytest.raises(Exception):  # Pydantic ValidationError
            self._make_result(initial_composite=1.5)  # > 1.0

    def test_default_prompt_saved_to_is_none(self):
        result = self._make_result()
        assert result.prompt_saved_to is None


# ============================================================
# TestMeanHelper
# ============================================================


class TestMeanHelper:
    """Unit tests for the _mean() helper function."""

    def test_mean_empty_list(self):
        assert _mean([], "composite") == 0.0

    def test_mean_single_element(self):
        results = [EvalResult("a", composite=0.6, rouge1=0.5, rouge2=0.3, rougeL=0.4)]
        assert _mean(results, "composite") == 0.6

    def test_mean_multiple_elements(self):
        results = [
            EvalResult("a", composite=0.4, rouge1=0.4, rouge2=0.2, rougeL=0.3),
            EvalResult("b", composite=0.6, rouge1=0.6, rouge2=0.4, rougeL=0.5),
        ]
        assert abs(_mean(results, "composite") - 0.5) < 1e-9

    def test_mean_rouge_l(self):
        results = [
            EvalResult("a", composite=0.5, rouge1=0.3, rouge2=0.1, rougeL=0.2),
            EvalResult("b", composite=0.5, rouge1=0.7, rouge2=0.5, rougeL=0.6),
        ]
        assert abs(_mean(results, "rougeL") - 0.4) < 1e-9
