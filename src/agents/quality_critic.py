"""QualityCritic agent: scores raw articles and keeps the top-N.

Inputs (from state):
    - topic, config (top_n, weights), raw_articles

Outputs:
    - quality_scores: scores for ALL raw articles (kept for traceability)
    - top_articles, top_scores: filtered top-N

Weight priority (Phase 3+):
    1. Learned weights from Optuna (WEIGHTS_DIR/{topic}_weights.json) — best
    2. Weights provided in state config["weights"] — explicit override
    3. DEFAULT_WEIGHTS from config.py — fallback
"""

from __future__ import annotations

import logging

from src.agents.base import finish_log, start_log
from src.agents.state import WatchState
from src.config import DEFAULT_TOP_N, DEFAULT_WEIGHTS, MIN_RELEVANCE_SCORE
from src.scoring.automl_scorer import load_weights_for_topic
from src.scoring.quality_scorer import filter_top_n, score_articles
from src.schemas import ArticleStatus

logger = logging.getLogger(__name__)

AGENT_NAME = "QualityCritic"


def run(state: WatchState) -> dict:
    log = start_log(AGENT_NAME)
    cfg = state.get("config", {})
    topic = state["topic"]
    raw_articles = state.get("raw_articles", [])
    top_n = cfg.get("top_n", DEFAULT_TOP_N)

    # ── Weight resolution: AutoML > config override > default ─────────────
    # Try learned weights first (Phase 3 AutoML).  Falls back gracefully.
    learned_weights = load_weights_for_topic(topic)
    if learned_weights is not None:
        weights = learned_weights
        logger.info("[%s] using AutoML-learned weights for topic '%s'", AGENT_NAME, topic)
    else:
        weights = cfg.get("weights", DEFAULT_WEIGHTS)
        logger.debug("[%s] using %s weights for topic '%s'",
                     AGENT_NAME,
                     "config-provided" if "weights" in cfg else "DEFAULT",
                     topic)

    if not raw_articles:
        logger.warning("[%s] no raw_articles to score", AGENT_NAME)
        return {
            "quality_scores": [],
            "top_articles": [],
            "top_scores": [],
            "logs": [finish_log(log, "success")],
        }

    try:
        scores = score_articles(raw_articles, topic, weights=weights)

        # ── Relevance gate: exclude off-topic articles before top-N ──────
        # A hard minimum ensures that high-citation papers from adjacent
        # fields (e.g. pure RNN papers for an attention-mechanism topic)
        # cannot override low-relevance with high venue/impact scores.
        #
        # Adaptive gate: when AutoML learned a low relevance weight (< 0.15),
        # it means the keyword relevance signal is noisy for this domain
        # (e.g. "federated learning" papers often avoid the exact phrase).
        # In that case we lower the gate to the score floor + a small margin,
        # effectively keeping any article that has at least some relevance signal
        # rather than excluding everything with a strict threshold.
        min_rel = cfg.get("min_relevance", MIN_RELEVANCE_SCORE)
        if learned_weights is not None:
            rel_weight = learned_weights.get("relevance", 1.0)
            if rel_weight < 0.15:
                # AutoML says relevance is a weak signal here — use lenient gate
                adaptive_min = 0.06  # just above the floor (0.05) to filter noise
                if min_rel > adaptive_min:
                    logger.info(
                        "[%s] AutoML relevance weight=%.3f < 0.15 → lowering gate "
                        "from %.2f to %.2f for topic '%s'",
                        AGENT_NAME, rel_weight, min_rel, adaptive_min, topic,
                    )
                    min_rel = adaptive_min
        relevant_pairs = [
            (art, score)
            for art, score in zip(raw_articles, scores)
            if score.relevance_score >= min_rel
        ]
        n_excluded = len(raw_articles) - len(relevant_pairs)
        if n_excluded:
            logger.info(
                "[%s] excluded %d off-topic articles (relevance_score < %.2f)",
                AGENT_NAME, n_excluded, min_rel,
            )

        if relevant_pairs:
            rel_arts, rel_scores = zip(*relevant_pairs)
            top_arts, top_scores = filter_top_n(
                list(rel_arts), list(rel_scores), n=top_n
            )
        else:
            top_arts, top_scores = [], []

        # Mark kept articles with SCORED status (single-writer ownership).
        for art in top_arts:
            art.status = ArticleStatus.SCORED

        logger.info(
            "[%s] scored %d articles, kept top %d (max=%.3f, min=%.3f)",
            AGENT_NAME,
            len(scores),
            len(top_arts),
            top_scores[0].final_score if top_scores else 0,
            top_scores[-1].final_score if top_scores else 0,
        )
        return {
            "quality_scores": scores,   # ALL scores kept for traceability
            "top_articles": top_arts,
            "top_scores": top_scores,
            "logs": [finish_log(log, "success")],
        }
    except Exception as exc:
        logger.exception("[%s] scoring failed", AGENT_NAME)
        return {
            "quality_scores": [],
            "top_articles": [],
            "top_scores": [],
            "logs": [finish_log(log, "failed", error=str(exc))],
            "errors": [f"{AGENT_NAME}: {exc}"],
        }
