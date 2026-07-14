"""ProTeGi: Prompt Optimization with Textual Gradients.

Reference paper:
    "Automatic Prompt Optimization with Gradient Descent and Beam Search"
    (Pryzant et al., 2023) — https://arxiv.org/abs/2305.03495

What is ProTeGi?
----------------
ProTeGi treats prompt optimization like gradient descent in ML:
    - "loss"      = low ROUGE / faithfulness score on examples
    - "gradient"  = an LLM-generated text that says *why* the prompt fails
    - "step"      = a new prompt that the LLM proposes to fix the gradient

Instead of numeric gradients (like backprop), the gradients are in *natural
language* — hence "textual gradients". This makes the method model-agnostic
and interpretable.

Algorithm (one iteration):
    1. EVALUATE  : run Summarizer with current_prompt on train set
                   → compute composite metric per example
    2. FIND WORST: select K examples with lowest scores
                   → these are the "signal" (where the prompt fails)
    3. GRADIENT  : LLM analyzes the failures → writes a textual gradient
                   "The prompt fails because it does not ask for X..."
    4. CANDIDATES: LLM proposes `beam_size` improved prompts
                   inspired by the gradient
    5. VALIDATE  : evaluate each candidate on a held-out validation set
    6. SELECT    : keep the candidate with the highest average score

Repeat for n_iterations. Return the prompt with the best validation score.

Integration in this project:
    - Task: Article → ArticleSummary (Summarizer agent)
    - Metric: ROUGE-L + LLM Judge composite (see summary_metrics.py)
    - Dataset: PubMed article-abstract pairs (from Kaggle)
    - Optimizes: SYSTEM_PROMPT of src/agents/summarizer.py
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from src.llm.base import LLMClient

logger = logging.getLogger(__name__)


# ============================================================
# Data structures
# ============================================================


@dataclass
class Example:
    """One training / validation example.

    Attributes:
        article_text : input fed to the Summarizer (truncated article body)
        reference    : gold summary (e.g. PubMed author-written abstract)
        article_id   : optional unique ID for tracing
    """

    article_text: str
    reference: str
    article_id: str = ""


@dataclass
class EvalResult:
    """Per-example evaluation result under one prompt."""

    example_id: str
    composite: float           # primary optimization target
    rouge1: float
    rouge2: float
    rougeL: float
    faithfulness: float = 0.5  # default = neutral (when Judge disabled)
    coverage: float = 0.5
    generated: str = ""        # first 300 chars of generated text (for debug)
    error: Optional[str] = None


@dataclass
class IterationResult:
    """Summary of one ProTeGi iteration (for the history log)."""

    iteration: int
    current_prompt: str
    train_composite: float
    val_composite: float        # best candidate score on validation
    improved: bool
    gradient_text: str = ""
    n_candidates_tried: int = 0
    best_candidate_preview: str = ""  # first 200 chars of winning candidate


class ProTeGiResult(BaseModel):
    """Final output of a ProTeGi optimization run — Pydantic for JSON export."""

    best_prompt: str = Field(..., description="Optimized system prompt")
    initial_prompt: str = Field(..., description="Starting system prompt")
    n_iterations: int
    initial_composite: float = Field(..., ge=0.0, le=1.0)
    final_composite: float = Field(..., ge=0.0, le=1.0)
    improvement_pct: float = Field(
        ..., description="Relative improvement vs initial, in percent"
    )
    iterations: list[dict] = Field(
        default_factory=list, description="Per-iteration detail log"
    )
    duration_seconds: float
    optimized_at: datetime = Field(default_factory=datetime.now)
    prompt_saved_to: Optional[str] = None


# ============================================================
# Meta-LLM system prompts (used internally by ProTeGi)
# ============================================================

_GRADIENT_SYSTEM = """\
You are an expert at improving AI prompts for scientific paper summarization.

You will receive:
  1. CURRENT PROMPT: the system prompt being evaluated
  2. FAILURE EXAMPLES: cases where the current prompt produced poor summaries
     (low ROUGE-L overlap with the gold abstract, or low faithfulness scores)

Your job is to write a GRADIENT — a paragraph that precisely diagnoses
what the current prompt is failing to produce. Cover:
  - What patterns of error appear across the failures
    (e.g. "the prompt produces vague problem statements")
  - What specific instruction is missing or poorly worded
    (e.g. "the prompt does not ask for quantitative results")
  - What concrete change would fix the problem
    (e.g. "add 'include specific numbers such as accuracy or F1 score'")

Be concrete and reference the actual failure examples. The gradient is
used by another LLM to write improved prompts — make it actionable."""


_CANDIDATE_SYSTEM = """\
You are an expert prompt engineer specializing in scientific NLP tasks.

You will receive:
  1. ORIGINAL PROMPT: the current system prompt for a scientific summarizer
  2. GRADIENT: an analysis of what the prompt is doing wrong
  3. N: how many improved prompt variants to generate

Produce exactly N improved prompts. Each improved prompt must:
  - Be a complete, self-contained system prompt (not a patch, the full text)
  - Fix the specific issues identified in the gradient
  - Keep the core task: extract structured summary from a scientific abstract
  - Keep the same 6-field output structure:
      problem, method, dataset, results, limitations, key_contributions
  - Stay under 600 words
  - Not add new output fields (would break the Pydantic schema downstream)

Output exactly a valid JSON array of strings, nothing else:
["prompt 1 full text...", "prompt 2 full text...", ...]
No markdown, no explanation — only the JSON array."""

# ── Narrative-mode variants of the two meta-prompts ──────────────────────────
# Used when ProTeGiOptimizer is created with narrative_mode=True.
# The key difference: candidates must produce a prose paragraph, NOT a JSON
# object with named fields.  ROUGE directly compares the paragraph to the
# PubMed gold abstract — so writing style and lexical overlap matter.

_GRADIENT_SYSTEM_NARRATIVE = """\
You are an expert at improving AI prompts for scientific paper summarization.

You will receive:
  1. CURRENT PROMPT: the system prompt being evaluated
  2. FAILURE EXAMPLES: cases where the current prompt produced poor summaries
     (low ROUGE-L overlap with the PubMed gold abstract, or low faithfulness)

Your job is to write a GRADIENT — a paragraph that precisely diagnoses
what the current prompt is failing to produce. Cover:
  - What patterns of error appear across the failures
    (e.g. "the prompt omits quantitative findings")
  - What specific instruction is missing or poorly worded
    (e.g. "the prompt does not ask for specific numbers")
  - What concrete change would fix the problem
    (e.g. "add 'include statistics such as p-values and percentages'")

The summary format is a NARRATIVE PARAGRAPH (prose), not structured fields.
Be concrete and reference the actual failure examples. The gradient is
used by another LLM to write improved prompts — make it actionable."""

_CANDIDATE_SYSTEM_NARRATIVE = """\
You are an expert prompt engineer specializing in scientific NLP tasks.

You will receive:
  1. ORIGINAL PROMPT: the current system prompt for a scientific summarizer
  2. GRADIENT: an analysis of what the prompt is doing wrong
  3. N: how many improved prompt variants to generate

Produce exactly N improved prompts. Each improved prompt must:
  - Be a complete, self-contained system prompt (not a patch, the full text)
  - Fix the specific issues identified in the gradient
  - Keep the core task: write a NARRATIVE PARAGRAPH summarizing a scientific article
  - The output format is prose (150-250 words), NOT a JSON object or bullet list
  - Stay under 400 words
  - Not introduce structured fields or JSON output

Output exactly a valid JSON array of strings, nothing else:
["prompt 1 full text...", "prompt 2 full text...", ...]
No markdown, no explanation — only the JSON array."""


# ============================================================
# ProTeGi optimizer class
# ============================================================


class ProTeGiOptimizer:
    """Optimize a Summarizer system prompt via iterative textual gradient descent.

    This class is framework-agnostic: it only needs LLMClient instances, not
    a full LangGraph state. It can be used standalone or wired into the agent
    pipeline.

    Args:
        summarize_llm : LLMClient used to generate summaries during evaluation
                        (recommended: gpt-4o-mini for cost efficiency)
        gradient_llm  : LLMClient used for gradient + candidate generation
                        (can be the same as summarize_llm, or a stronger model)
        judge_llm     : LLMClient for LLM Judge evaluation
                        (defaults to gradient_llm if not provided)
        n_iterations  : Number of optimization cycles (3 is a good default)
        beam_size     : How many candidate prompts to generate per iteration
                        (4 gives good coverage, >6 becomes expensive)
        n_worst       : Number of worst examples to use as gradient signal
                        (5 is enough; more adds noise)
        use_llm_judge : Whether to include LLM Judge in the composite metric
                        (set False for fast/cheap runs)
        use_rouge     : Whether to include ROUGE in the composite metric.
                        Set False when the reference format does not match the
                        generated format (e.g. PubMed narrative abstract vs
                        structured 6-field summary). Requires use_llm_judge=True
                        to keep a meaningful signal.
    """

    def __init__(
        self,
        summarize_llm: "LLMClient",
        gradient_llm: "LLMClient",
        judge_llm: Optional["LLMClient"] = None,
        n_iterations: int = 3,
        beam_size: int = 4,
        n_worst: int = 5,
        use_llm_judge: bool = True,
        use_rouge: bool = True,
        narrative_mode: bool = False,
    ) -> None:
        self.summarize_llm = summarize_llm
        self.gradient_llm = gradient_llm
        self.judge_llm = judge_llm or gradient_llm
        self.n_iterations = n_iterations
        self.beam_size = beam_size
        self.n_worst = n_worst
        self.use_llm_judge = use_llm_judge
        self.use_rouge = use_rouge
        self.narrative_mode = narrative_mode

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize(
        self,
        initial_prompt: str,
        train_examples: list[Example],
        val_examples: list[Example],
        save_dir: Optional[Path] = None,
        run_name: str = "",
    ) -> ProTeGiResult:
        """Run the full ProTeGi loop and return the best prompt.

        Args:
            initial_prompt  : starting SYSTEM_PROMPT (e.g. from summarizer.py)
            train_examples  : used to compute gradient (find failures)
            val_examples    : held-out set used to select best candidate
            save_dir        : if provided, saves the best prompt + results JSON
            run_name        : optional label included in saved filenames
                              e.g. "llama-haiku-sonnet" →
                              summarizer_prompt_llama-haiku-sonnet_20260526.txt

        Returns:
            ProTeGiResult with best_prompt and full iteration history.
        """
        start_time = datetime.now()
        current_prompt = initial_prompt
        history: list[dict] = []

        logger.info(
            "ProTeGi start: %d iterations, beam=%d, judge=%s",
            self.n_iterations, self.beam_size, self.use_llm_judge,
        )
        logger.info(
            "  Dataset: %d train | %d val examples",
            len(train_examples), len(val_examples),
        )

        # ── Baseline: evaluate initial prompt on validation set ───────────
        baseline_results = self._evaluate_set(initial_prompt, val_examples)
        baseline_composite = _mean(baseline_results, "composite")
        best_composite = baseline_composite
        best_prompt = initial_prompt

        logger.info("  Baseline composite (val): %.4f", baseline_composite)

        # ── Main optimization loop ────────────────────────────────────────
        for i in range(self.n_iterations):
            logger.info("=== Iteration %d / %d ===", i + 1, self.n_iterations)

            # Step 1: evaluate current prompt on TRAIN set to find failures
            train_results = self._evaluate_set(current_prompt, train_examples)
            train_composite = _mean(train_results, "composite")

            # Step 2: select K worst examples (gradient signal)
            worst = sorted(train_results, key=lambda r: r.composite)[: self.n_worst]
            logger.info(
                "  Worst %d (train) composite: %.4f ... %.4f",
                self.n_worst,
                worst[0].composite if worst else 0,
                worst[-1].composite if worst else 0,
            )

            # Map worst results back to their Example objects
            id_to_example = {ex.article_id: ex for ex in train_examples}
            worst_examples = [
                id_to_example.get(r.example_id)
                for r in worst
                if r.example_id in id_to_example
            ]
            worst_examples = [e for e in worst_examples if e is not None]

            # Step 3: generate textual gradient
            gradient = self._generate_gradient(current_prompt, worst, worst_examples)
            logger.info("  ┌─ GRADIENT (diagnosis) ─────────────────────────")
            for line in gradient.splitlines():
                logger.info("  │  %s", line)
            logger.info("  └────────────────────────────────────────────────")

            # Step 4: generate beam_size candidate prompts
            candidates = self._generate_candidates(current_prompt, gradient)
            logger.info("  Generated %d valid candidates", len(candidates))

            # Step 5: evaluate each candidate on VALIDATION set
            best_candidate = current_prompt
            best_candidate_composite = best_composite  # must beat the current best

            candidates_detail: list[dict] = []
            for j, candidate in enumerate(candidates):
                # ── Display candidate prompt ──────────────────────────────
                logger.info(
                    "  ┌─ Candidate %d/%d ─────────────────────────────────",
                    j + 1, len(candidates),
                )
                for line in candidate.splitlines():
                    logger.info("  │  %s", line)
                logger.info(
                    "  └────────────────────────────────────────────────────"
                )

                # Pause between candidates so Groq's 60-second TPM window can
                # partially reset. Without this, all N_val Groq calls from the
                # previous candidate accumulate and push us over the 6 000 TPM
                # free-tier cap the moment the next candidate evaluation starts.
                if j > 0 and self._uses_groq():
                    pause = 25
                    logger.info(
                        "  [Groq TPM pause] waiting %ds before candidate %d/%d …",
                        pause, j + 1, len(candidates),
                    )
                    time.sleep(pause)

                cand_results = self._evaluate_set(candidate, val_examples)
                cand_composite = _mean(cand_results, "composite")
                is_best = cand_composite > best_candidate_composite
                logger.info(
                    "  Candidate %d/%d score: composite=%.4f  %s",
                    j + 1, len(candidates), cand_composite,
                    "★ NEW BEST" if is_best else "",
                )
                candidates_detail.append({
                    "rank":      j + 1,
                    "prompt":    candidate,
                    "composite": round(cand_composite, 4),
                    "rouge1":    round(_mean(cand_results, "rouge1"), 4),
                    "rouge2":    round(_mean(cand_results, "rouge2"), 4),
                    "rougeL":    round(_mean(cand_results, "rougeL"), 4),
                    "faithfulness": round(_mean(cand_results, "faithfulness"), 4),
                    "coverage":     round(_mean(cand_results, "coverage"), 4),
                    "is_best":   is_best,
                })
                if is_best:
                    best_candidate_composite = cand_composite
                    best_candidate = candidate

            # Step 6: update if improved
            improved = best_candidate_composite > best_composite
            if improved:
                delta = best_candidate_composite - best_composite
                logger.info(
                    "  [IMPROVED] +%.4f -> new composite=%.4f",
                    delta, best_candidate_composite,
                )
                best_composite = best_candidate_composite
                best_prompt = best_candidate
                current_prompt = best_candidate
            else:
                logger.info("  [NO CHANGE] keeping current prompt")

            history.append({
                "iteration":               i + 1,
                "train_composite":         round(train_composite, 4),
                "best_candidate_composite": round(best_candidate_composite, 4),
                "improved":                improved,
                # Full texts saved for the detail file
                "gradient":                gradient,
                "candidates":              candidates_detail,
                # Short previews kept for the summary JSON (backward compat)
                "gradient_preview":        gradient[:300],
                "n_candidates":            len(candidates),
                "best_candidate_preview":  best_candidate[:200],
            })

        # ── Final metrics ─────────────────────────────────────────────────
        duration = (datetime.now() - start_time).total_seconds()
        improvement_pct = (
            (best_composite - baseline_composite) / max(baseline_composite, 1e-6) * 100
        )

        logger.info(
            "ProTeGi done: %.4f -> %.4f (%+.1f%%) in %.1fs",
            baseline_composite, best_composite, improvement_pct, duration,
        )

        # ── Save results ──────────────────────────────────────────────────
        saved_path: Optional[str] = None
        if save_dir is not None:
            saved_path = str(
                self._save_results(
                    save_dir=save_dir,
                    best_prompt=best_prompt,
                    run_name=run_name,
                    metadata={
                        "run_name": run_name,
                        "initial_composite": round(baseline_composite, 4),
                        "final_composite": round(best_composite, 4),
                        "improvement_pct": round(improvement_pct, 2),
                        "n_iterations": self.n_iterations,
                        "beam_size": self.beam_size,
                        "n_worst": self.n_worst,
                        "use_llm_judge": self.use_llm_judge,
                        "iterations": history,
                    },
                )
            )

        return ProTeGiResult(
            best_prompt=best_prompt,
            initial_prompt=initial_prompt,
            n_iterations=self.n_iterations,
            initial_composite=round(baseline_composite, 4),
            final_composite=round(best_composite, 4),
            improvement_pct=round(improvement_pct, 2),
            iterations=history,
            duration_seconds=round(duration, 1),
            prompt_saved_to=saved_path,
        )

    # ------------------------------------------------------------------
    # Internal: helpers
    # ------------------------------------------------------------------

    def _uses_groq(self) -> bool:
        """Return True if the summarize LLM is backed by Groq.

        Used to decide whether to insert inter-candidate TPM pause: Groq's
        free tier has a 6 000 TPM cap that requires pacing; other providers
        (Anthropic, OpenAI) have much higher limits and don't need the pause.
        """
        provider = getattr(self.summarize_llm, "provider", "")
        # GroqClient reports provider="openai" (reuses the OpenAI SDK) but its
        # base_url contains "groq" — check the class name as a fallback.
        return (
            provider == "groq"
            or "groq" in type(self.summarize_llm).__name__.lower()
        )

    # ------------------------------------------------------------------
    # Internal: evaluate
    # ------------------------------------------------------------------

    def _evaluate_set(
        self, prompt: str, examples: list[Example]
    ) -> list[EvalResult]:
        """Run the Summarizer on all examples, compute metrics for each."""
        from src.llm.base import Message
        from src.optimization.summary_metrics import (
            compute_all_metrics,
            flatten_summary,
        )
        from src.schemas import ArticleSummary

        results: list[EvalResult] = []
        for ex in examples:
            try:
                # Format input the same way summarizer.py does
                user_text = (
                    f"Title: Scientific Article\n\n"
                    f"Abstract:\n{ex.article_text}\n"
                )
                messages = [Message(role="user", content=user_text)]

                if self.narrative_mode:
                    # Narrative mode: plain text call, no schema validation
                    resp = self.summarize_llm.chat(
                        system=prompt,
                        messages=messages,
                        temperature=0.0,
                        max_tokens=512,
                    )
                    generated_text = resp.content.strip()
                else:
                    # Structured mode: force 6-field JSON output
                    parsed, _resp = self.summarize_llm.chat_structured(
                        system=prompt,
                        messages=messages,
                        schema=ArticleSummary,
                        temperature=0.0,
                        max_tokens=1024,
                    )
                    parsed.article_id = ex.article_id or "protegi_eval"
                    generated_text = flatten_summary(parsed)
                metrics = compute_all_metrics(
                    generated=generated_text,
                    reference=ex.reference,
                    article_text=ex.article_text if self.use_llm_judge else None,
                    llm=self.judge_llm if self.use_llm_judge else None,
                    use_rouge=self.use_rouge,
                )

                results.append(
                    EvalResult(
                        example_id=ex.article_id or "?",
                        composite=float(metrics.get("composite", 0.0)),
                        rouge1=float(metrics.get("rouge1", 0.0)),
                        rouge2=float(metrics.get("rouge2", 0.0)),
                        rougeL=float(metrics.get("rougeL", 0.0)),
                        faithfulness=float(metrics.get("faithfulness", 0.5)),
                        coverage=float(metrics.get("coverage", 0.5)),
                        generated=generated_text[:300],
                    )
                )

            except Exception as exc:
                logger.warning(
                    "Evaluation failed for example %s: %s",
                    ex.article_id, exc,
                )
                results.append(
                    EvalResult(
                        example_id=ex.article_id or "?",
                        composite=0.0,
                        rouge1=0.0,
                        rouge2=0.0,
                        rougeL=0.0,
                        error=str(exc),
                    )
                )

        return results

    # ------------------------------------------------------------------
    # Internal: gradient generation
    # ------------------------------------------------------------------

    def _generate_gradient(
        self,
        current_prompt: str,
        worst_results: list[EvalResult],
        worst_examples: list[Example],
    ) -> str:
        """Ask gradient_llm to diagnose what the prompt is doing wrong."""
        from src.llm.base import Message

        # Build compact failure descriptions
        failure_lines: list[str] = []
        for i, r in enumerate(worst_results[:self.n_worst]):
            # Try to find the matching example for context
            ex = next(
                (e for e in worst_examples if e.article_id == r.example_id),
                None,
            )
            article_snippet = (ex.article_text[:300] if ex else "[unavailable]")
            failure_lines.append(
                f"Example {i + 1} (id={r.example_id}):\n"
                f"  Scores: composite={r.composite:.4f}, "
                f"ROUGE-L={r.rougeL:.4f}, "
                f"faithfulness={r.faithfulness:.2f}\n"
                f"  Article excerpt: {article_snippet}...\n"
                f"  Generated: {r.generated[:250]}..."
            )

        failure_text = "\n\n".join(failure_lines)
        user_text = (
            f"CURRENT PROMPT:\n{current_prompt}\n\n"
            f"FAILURE EXAMPLES (lowest scoring):\n{failure_text}"
        )
        messages = [Message(role="user", content=user_text)]

        gradient_sys = _GRADIENT_SYSTEM_NARRATIVE if self.narrative_mode else _GRADIENT_SYSTEM
        try:
            resp = self.gradient_llm.chat(
                system=gradient_sys,
                messages=messages,
                temperature=0.3,
                max_tokens=600,
            )
            return resp.content.strip()
        except Exception as exc:
            logger.warning("Gradient generation failed: %s", exc)
            # Minimal fallback gradient
            return (
                "The prompt needs to explicitly request quantitative results "
                "(numbers, percentages, dataset names) and discourage vague "
                "or generic language."
            )

    # ------------------------------------------------------------------
    # Internal: candidate generation
    # ------------------------------------------------------------------

    def _generate_candidates(
        self, current_prompt: str, gradient: str
    ) -> list[str]:
        """Ask gradient_llm to propose beam_size improved prompts."""
        from src.llm.base import Message

        user_text = (
            f"ORIGINAL PROMPT:\n{current_prompt}\n\n"
            f"GRADIENT (what is wrong and how to fix it):\n{gradient}\n\n"
            f"N: {self.beam_size}\n\n"
            f"Generate exactly {self.beam_size} improved prompt variants "
            f"as a JSON array of strings."
        )
        messages = [Message(role="user", content=user_text)]

        candidate_sys = _CANDIDATE_SYSTEM_NARRATIVE if self.narrative_mode else _CANDIDATE_SYSTEM
        try:
            resp = self.gradient_llm.chat(
                system=candidate_sys,
                messages=messages,
                temperature=0.7,          # some diversity between candidates
                max_tokens=self.beam_size * 700,  # ~600 words per prompt
            )
            content = resp.content.strip()

            # Extract JSON array (the LLM might add explanation before/after)
            start = content.find("[")
            end = content.rfind("]") + 1
            if start != -1 and end > start:
                parsed = json.loads(content[start:end])
                if isinstance(parsed, list):
                    valid = [p for p in parsed if isinstance(p, str) and len(p) > 50]
                    if valid:
                        return valid[: self.beam_size]

        except json.JSONDecodeError as exc:
            logger.warning("Candidate JSON parsing failed: %s", exc)
        except Exception as exc:
            logger.warning("Candidate generation failed: %s", exc)

        # Fallback: return the current prompt as the only "candidate"
        logger.warning("Falling back to current prompt (no valid candidates)")
        return [current_prompt]

    # ------------------------------------------------------------------
    # Internal: persistence
    # ------------------------------------------------------------------

    @staticmethod
    def _save_results(
        save_dir: Path, best_prompt: str, metadata: dict, run_name: str = ""
    ) -> Path:
        """Write the best prompt and run metadata to disk.

        Creates three files (all prefixed with run_name when provided):
            {save_dir}/summarizer_prompt_{run_name}_{timestamp}.txt
            {save_dir}/protegi_results_{run_name}_{timestamp}.json
            {save_dir}/protegi_detail_{run_name}_{timestamp}.json
        """
        import re
        save_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Sanitize run_name for use in filenames (replace unsafe chars with _)
        safe_name = re.sub(r"[^\w\-]", "_", run_name).strip("_") if run_name else ""
        prefix = f"{safe_name}_" if safe_name else ""

        # 1. Best prompt plain text
        prompt_path = save_dir / f"summarizer_prompt_{prefix}{timestamp}.txt"
        prompt_path.write_text(best_prompt, encoding="utf-8")
        logger.info("Saved optimized prompt : %s", prompt_path)

        # 2. Summary metadata (compact — no full prompt/gradient texts)
        summary = {k: v for k, v in metadata.items() if k != "iterations"}
        summary["iterations"] = [
            {k: v for k, v in it.items() if k not in ("gradient", "candidates")}
            for it in metadata.get("iterations", [])
        ]
        meta_path = save_dir / f"protegi_results_{prefix}{timestamp}.json"
        meta_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        logger.info("Saved ProTeGi summary   : %s", meta_path)

        # 3. Full detail file — gradient + all candidate prompts + scores
        detail_path = save_dir / f"protegi_detail_{prefix}{timestamp}.json"
        detail_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("Saved ProTeGi detail    : %s", detail_path)

        return prompt_path


# ============================================================
# Helpers
# ============================================================


def _mean(results: list[EvalResult], attr: str) -> float:
    """Compute the mean of a numeric attribute across EvalResult list."""
    if not results:
        return 0.0
    values = [getattr(r, attr, 0.0) for r in results]
    return sum(values) / len(values)
