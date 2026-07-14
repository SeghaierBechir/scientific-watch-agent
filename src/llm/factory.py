"""LLM factory: pick the right client for a given task.

Single entry point for agents to obtain an LLM:

    from src.llm.factory import get_llm_for_task
    llm = get_llm_for_task("summarize")
    parsed, resp = llm.chat_structured(system, messages, ArticleSummary)

This indirection means agents never know whether they got Claude or GPT, and
benchmark mode can swap models without touching agent code.
"""

from __future__ import annotations

import logging
from typing import Literal

from src.config import TASK_MODELS
from src.llm.base import LLMClient
from src.llm.claude_client import ClaudeClient
from src.llm.groq_client import GroqClient
from src.llm.nebius_client import NebiusClient
from src.llm.openai_client import OpenAIClient

logger = logging.getLogger(__name__)


TaskName = Literal[
    "query_expansion",
    "summarize",
    "synthesize",
    "trend_analysis",
    "critic",
    "judge",
]

Provider = Literal["anthropic", "openai", "groq", "nebius"]


def get_llm(provider: Provider, model: str) -> LLMClient:
    """Build an LLMClient for an explicit (provider, model) pair."""
    if provider == "anthropic":
        return ClaudeClient(model=model)
    if provider == "openai":
        return OpenAIClient(model=model)
    if provider == "groq":
        return GroqClient(model=model)
    if provider == "nebius":
        return NebiusClient(model=model)
    raise ValueError(f"Unknown provider {provider!r}")


def get_llm_for_task(task: TaskName) -> LLMClient:
    """Build the LLM associated with a task in `TASK_MODELS`.

    Falls back loudly if the task is not configured — better to crash early
    than silently send work to the wrong model.
    """
    if task not in TASK_MODELS:
        raise KeyError(
            f"Task {task!r} is not in TASK_MODELS. "
            f"Available: {list(TASK_MODELS.keys())}"
        )
    provider, model = TASK_MODELS[task]
    logger.debug("Building LLM for task %s: %s/%s", task, provider, model)
    return get_llm(provider, model)
