"""Metrics for evaluating article summaries — used by ProTeGi.

Three metric tiers, ordered by cost:

    Tier 1 — ROUGE  (rouge-score library OR built-in fallback)
        Zero cost, instant. Measures word-overlap between the generated
        summary and a gold reference text (e.g. the PubMed abstract).
        - ROUGE-1 : unigram recall/precision/F1
        - ROUGE-2 : bigram overlap (captures bigrams like "deep learning")
        - ROUGE-L : longest common subsequence (order-aware)

        Implementation priority:
          1. rouge-score library (pip install rouge-score) — uses stemming
          2. Built-in pure-Python fallback — no external dependency,
             slightly different scores but same ranking behaviour

    Tier 2 — LLM Judge (one LLM call per summary, ~$0.001/call)
        Asks an LLM to score two semantic dimensions:
        - faithfulness : is every claim grounded in the source article?
        - coverage     : are the key findings captured?
        This catches hallucinations and omissions that ROUGE misses because
        ROUGE is purely lexical (synonym-blind, order-naive).

    Tier 3 — BERTScore (optional, heavy ~400 MB model)
        Contextual embedding similarity. Better than ROUGE for paraphrases
        but requires loading a transformer model. Imported lazily to avoid
        cost when unused. Omitted from the default composite.

All public functions accept:
    generated     : flat text produced by flatten_summary()
    reference     : gold text (PubMed abstract, or any reference summary)
    article_text  : source article body (required only for LLM Judge)
    llm           : any LLMClient instance (required only for LLM Judge)

Return value: dict[str, float | str]  — metric names to values in [0, 1]
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from src.llm.base import LLMClient
    from src.schemas import ArticleSummary

logger = logging.getLogger(__name__)


# ============================================================
# Pure-Python ROUGE fallback (no external dependencies)
# ============================================================
# Used automatically when `rouge-score` is not installed.
# Implements the same F1 formula: 2*P*R / (P+R)
#   ROUGE-1: unigram overlap
#   ROUGE-2: bigram overlap
#   ROUGE-L: longest common subsequence (LCS)


def _rouge_tokenize(text: str) -> list[str]:
    """Lowercase + keep only alphanumeric tokens.

    Unlike the relevance-score tokenizer, we do NOT remove stop words here —
    ROUGE is meant to measure raw word overlap faithfully.
    """
    return re.findall(r"[a-z0-9]+", text.lower())


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    """Return all n-grams from a token list."""
    return [tuple(tokens[i: i + n]) for i in range(len(tokens) - n + 1)]


def _overlap_f1(gen_items: list, ref_items: list) -> float:
    """Compute F1 of item overlap using Counter (handles duplicates correctly)."""
    if not gen_items or not ref_items:
        return 0.0
    gen_c = Counter(gen_items)
    ref_c = Counter(ref_items)
    overlap = sum(min(gen_c[item], ref_c[item]) for item in gen_c)
    precision = overlap / len(gen_items)
    recall = overlap / len(ref_items)
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Compute length of the Longest Common Subsequence (DP, O(m*n)).

    Space-optimised: only keeps two rows in memory.
    For typical text inputs (< 500 tokens each) this is very fast.
    """
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


def _compute_rouge_builtin(generated: str, reference: str) -> dict[str, float]:
    """Pure-Python ROUGE-1, ROUGE-2, ROUGE-L.  No external library required."""
    gen_tok = _rouge_tokenize(generated)
    ref_tok = _rouge_tokenize(reference)

    if not gen_tok or not ref_tok:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    # ROUGE-1
    r1 = _overlap_f1(gen_tok, ref_tok)

    # ROUGE-2
    gen_bg = _ngrams(gen_tok, 2)
    ref_bg = _ngrams(ref_tok, 2)
    r2 = _overlap_f1(gen_bg, ref_bg)

    # ROUGE-L (LCS-based F1)
    lcs = _lcs_length(gen_tok, ref_tok)
    if lcs == 0:
        rl = 0.0
    else:
        p = lcs / len(gen_tok)
        r = lcs / len(ref_tok)
        rl = 2.0 * p * r / (p + r)

    return {
        "rouge1": round(r1, 4),
        "rouge2": round(r2, 4),
        "rougeL": round(rl, 4),
    }


# ============================================================
# Tier 1 — ROUGE (library-first, fallback to built-in)
# ============================================================


def compute_rouge(generated: str, reference: str) -> dict[str, float]:
    """Compute ROUGE-1, ROUGE-2, ROUGE-L F1 scores.

    Tries the `rouge-score` library first (uses stemming for better accuracy).
    Falls back to the built-in pure-Python implementation automatically when
    `rouge-score` is not installed — no ImportError is ever raised.

    Args:
        generated : text generated by the Summarizer (via flatten_summary)
        reference : gold reference text (e.g. PubMed abstract)

    Returns:
        {"rouge1": float, "rouge2": float, "rougeL": float}

    Example:
        >>> scores = compute_rouge("deep learning model", "deep neural network")
        >>> 0.0 <= scores["rouge1"] <= 1.0
        True
    """
    if not generated or not reference:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    try:
        from rouge_score import rouge_scorer  # type: ignore[import]

        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )
        scores = scorer.score(reference, generated)
        return {
            "rouge1": round(scores["rouge1"].fmeasure, 4),
            "rouge2": round(scores["rouge2"].fmeasure, 4),
            "rougeL": round(scores["rougeL"].fmeasure, 4),
        }
    except ImportError:
        # rouge-score not installed — use the built-in pure-Python fallback.
        # Scores are slightly different (no stemming) but ranking is preserved.
        logger.debug("rouge-score not installed; using built-in ROUGE fallback")
        return _compute_rouge_builtin(generated, reference)


# ============================================================
# Tier 2 — LLM Judge
# ============================================================


class JudgeOutput(BaseModel):
    """Structured output from the LLM Judge (used in chat_structured)."""

    faithfulness: float = Field(
        ..., ge=0.0, le=1.0,
        description=(
            "Is the summary faithful to the source article? "
            "0.0 = contains fabricated information not in the article, "
            "1.0 = every statement is grounded in the article"
        ),
    )
    coverage: float = Field(
        ..., ge=0.0, le=1.0,
        description=(
            "Does the summary cover the key points? "
            "0.0 = misses all key points (problem, method, results), "
            "1.0 = captures problem + method + main results + contributions"
        ),
    )
    reasoning: str = Field(
        ...,
        description=(
            "Brief explanation of both scores in 1-3 sentences. "
            "Cite specific evidence from the article."
        ),
    )


_JUDGE_SYSTEM_PROMPT = """\
You are an expert scientific text evaluator.

You will receive:
  1. ARTICLE: the source scientific article text
  2. SUMMARY: a structured summary generated by an AI

Score the summary on two criteria, each from 0.0 to 1.0:

faithfulness (is the summary accurate?):
  0.0 = contains fabricated numbers, methods, or findings not in the article
  0.5 = mostly accurate but has minor inaccuracies or vague claims
  1.0 = every statement can be verified from the article text

coverage (does the summary capture the essentials?):
  0.0 = misses the research problem, method, and results entirely
  0.5 = captures 1-2 key aspects but misses important findings or methods
  1.0 = problem + method + main results + key contributions all present

Be strict: a summary that only paraphrases the title without substance scores
low on coverage. A summary that invents specific numbers scores low on faithfulness."""


def compute_llm_judge(
    generated: str,
    article_text: str,
    llm: "LLMClient",
) -> dict[str, float | str]:
    """Evaluate faithfulness and coverage using an LLM as judge.

    One LLM call per summary. Uses chat_structured to get reliable scores.

    Args:
        generated    : flat text generated by flatten_summary()
        article_text : source article body (truncated to 3000 chars internally)
        llm          : any LLMClient (gpt-4o-mini is recommended for cost)

    Returns:
        {
            "faithfulness"  : float in [0, 1],
            "coverage"      : float in [0, 1],
            "reasoning"     : str,
            "judge_cost_usd": float,
        }
    """
    from src.llm.base import Message  # local import — avoids circular dependency

    # Truncate article to avoid blowing token budget
    article_snippet = article_text[:3000]

    user_text = (
        f"ARTICLE:\n{article_snippet}\n\n"
        f"SUMMARY:\n{generated}"
    )
    messages = [Message(role="user", content=user_text)]

    try:
        result, resp = llm.chat_structured(
            system=_JUDGE_SYSTEM_PROMPT,
            messages=messages,
            schema=JudgeOutput,
            temperature=0.0,
            max_tokens=512,
        )
        return {
            "faithfulness": round(result.faithfulness, 4),
            "coverage": round(result.coverage, 4),
            "reasoning": result.reasoning,
            "judge_cost_usd": resp.cost_usd,
        }
    except Exception as exc:
        logger.warning("LLM Judge call failed: %s", exc)
        # Neutral fallback — does not penalize or reward
        return {
            "faithfulness": 0.5,
            "coverage": 0.5,
            "reasoning": f"[Judge error: {exc}]",
            "judge_cost_usd": 0.0,
        }


# ============================================================
# Tier 3 — BERTScore (optional, lazy import)
# ============================================================


def compute_bertscore(
    generated: str,
    reference: str,
    model_type: str = "distilbert-base-uncased",
) -> dict[str, float]:
    """Compute BERTScore F1 (optional tier, ~400 MB model download).

    Uses contextual embeddings to measure semantic similarity — better than
    ROUGE for paraphrases but much heavier. Lazily imported.

    Args:
        generated   : flat generated text
        reference   : gold reference text
        model_type  : HuggingFace model for embeddings (default: distilbert)

    Returns:
        {"bertscore_f1": float}

    Raises:
        ImportError: if bert-score is not installed.
    """
    try:
        from bert_score import score as bs_score  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "bert-score is not installed. Run: pip install bert-score"
        ) from exc

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _P, _R, F1 = bs_score(
            [generated], [reference],
            model_type=model_type,
            verbose=False,
        )
    return {"bertscore_f1": round(float(F1[0]), 4)}


# ============================================================
# Aggregate: all metrics together
# ============================================================


def compute_all_metrics(
    generated: str,
    reference: str,
    article_text: Optional[str] = None,
    llm: Optional["LLMClient"] = None,
    use_bertscore: bool = False,
    use_rouge: bool = True,
) -> dict[str, float | str]:
    """Compute all available metrics and return a composite score.

    ROUGE is computed by default but can be disabled with ``use_rouge=False``.
    LLM Judge is computed when `llm` and `article_text` are provided.
    BERTScore is computed when `use_bertscore=True` and bert-score installed.

    Composite score formula:
        - ROUGE + Judge  : 0.40*ROUGE-L + 0.40*faithfulness + 0.20*coverage
        - Judge only     : 0.60*faithfulness + 0.40*coverage
          (use when the reference text is a different format than the output,
           e.g. PubMed narrative abstract vs structured 6-field summary)
        - ROUGE only     : mean(ROUGE-1, ROUGE-2, ROUGE-L)

    Args:
        generated     : flat text produced by flatten_summary()
        reference     : gold reference text (only used when use_rouge=True)
        article_text  : source article body (required only for LLM Judge)
        llm           : any LLMClient instance (required only for LLM Judge)
        use_bertscore : enable optional BERTScore tier
        use_rouge     : set False to skip ROUGE entirely and rely on LLM Judge
                        alone — useful when the reference format does not match
                        the generated format (e.g. PubMed abstract vs structured
                        summary). Requires llm + article_text to be provided.

    Returns:
        dict containing all individual metrics plus "composite" in [0, 1].
    """
    metrics: dict[str, float | str] = {}

    # --- Tier 1: ROUGE (skippable) ---
    if use_rouge:
        rouge = compute_rouge(generated, reference)
        metrics.update(rouge)
    else:
        # Fill with neutral zeros so downstream code (EvalResult) never KeyErrors
        metrics.update({"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0})
        logger.debug("ROUGE disabled (use_rouge=False) — using LLM Judge only")

    # --- Tier 2: LLM Judge (when configured) ---
    if llm is not None and article_text is not None:
        judge = compute_llm_judge(generated, article_text, llm)
        metrics.update(judge)

    # --- Tier 3: BERTScore (optional) ---
    if use_bertscore:
        try:
            bs = compute_bertscore(generated, reference)
            metrics.update(bs)
        except ImportError:
            logger.warning("BERTScore skipped: bert-score not installed")

    # --- Composite ---
    has_judge = "faithfulness" in metrics
    if has_judge and use_rouge:
        # Full composite: ROUGE-L + Judge
        composite = (
            0.40 * float(metrics["rougeL"])
            + 0.40 * float(metrics["faithfulness"])
            + 0.20 * float(metrics["coverage"])
        )
    elif has_judge and not use_rouge:
        # Judge only: faithfulness weighted higher (main quality signal)
        composite = (
            0.60 * float(metrics["faithfulness"])
            + 0.40 * float(metrics["coverage"])
        )
    else:
        # ROUGE only (no Judge)
        composite = (
            float(metrics["rouge1"])
            + float(metrics["rouge2"])
            + float(metrics["rougeL"])
        ) / 3.0

    metrics["composite"] = round(composite, 4)
    return metrics


# ============================================================
# Utility: flatten ArticleSummary to text
# ============================================================


def flatten_summary(summary: "ArticleSummary") -> str:
    """Convert an ArticleSummary into a single text string for metric computation.

    Concatenates all non-null fields so ROUGE has enough substance to compare.
    Fields marked 'Not specified' (LLM placeholder) are excluded.

    Args:
        summary: ArticleSummary produced by the Summarizer agent.

    Returns:
        Multi-sentence text representing the full structured summary.

    Example:
        >>> # Produces: "Problem. Method. Results. Key: contrib1. contrib2."
    """
    _NOT_SPECIFIED = "not specified"

    def _include(s: Optional[str]) -> bool:
        return bool(s) and s.lower().strip() != _NOT_SPECIFIED

    parts: list[str] = []

    if _include(summary.problem):
        parts.append(summary.problem)
    if _include(summary.method):
        parts.append(summary.method)
    if _include(summary.results):
        parts.append(summary.results)
    if _include(summary.dataset):
        parts.append(f"Dataset: {summary.dataset}")
    if _include(summary.limitations):
        parts.append(f"Limitations: {summary.limitations}")
    if summary.key_contributions:
        for contrib in summary.key_contributions:
            if _include(contrib):
                parts.append(contrib)

    return " ".join(parts)
