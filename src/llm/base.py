"""Unified LLM client abstraction.

Defines the `LLMClient` Protocol that both ClaudeClient and OpenAIClient
implement. Agents depend on this Protocol, never on a concrete provider.

Two methods are exposed:
    - chat(): plain text completion
    - chat_structured(): structured output, returns a Pydantic-validated object

Each call returns an `LLMResponse` carrying token usage and cost, so we can
populate `AgentLog` and track per-run budget.
"""

from __future__ import annotations

from typing import Literal, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, Field

from src.config import MODEL_PRICING

T = TypeVar("T", bound=BaseModel)


# ============================================================
# Message and response schemas
# ============================================================


class Message(BaseModel):
    """One turn in a chat conversation."""

    role: Literal["user", "assistant"]
    content: str


class LLMResponse(BaseModel):
    """Result of an LLM call, including usage and cost.

    `cache_read_tokens` and `cache_creation_tokens` are Anthropic-specific
    (prompt caching). For OpenAI they stay at 0.
    """

    content: str = ""
    model: str
    provider: Literal["anthropic", "openai"]

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    cost_usd: float = 0.0

    raw: dict = Field(default_factory=dict, description="Provider-specific raw payload")


# ============================================================
# Cost helper
# ============================================================


def estimate_cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
) -> float:
    """Estimate USD cost for a call, using `MODEL_PRICING` from config.

    Anthropic cache tokens have specific pricing:
        - cache_creation: 1.25x base input price
        - cache_read:     0.10x base input price (90% off)
    For models we don't have pricing for, returns 0.0.
    """
    pricing = MODEL_PRICING.get(model)
    if pricing is None:
        return 0.0

    input_price = pricing["input"] / 1_000_000
    output_price = pricing["output"] / 1_000_000

    fresh_input = max(0, input_tokens - cache_read_tokens - cache_creation_tokens)
    cost = (
        fresh_input * input_price
        + cache_creation_tokens * input_price * 1.25
        + cache_read_tokens * input_price * 0.10
        + output_tokens * output_price
    )
    return round(cost, 6)


# ============================================================
# LLMClient Protocol
# ============================================================


@runtime_checkable
class LLMClient(Protocol):
    """Unified LLM interface. Both ClaudeClient and OpenAIClient implement this."""

    @property
    def provider(self) -> Literal["anthropic", "openai"]: ...

    @property
    def model(self) -> str: ...

    def chat(
        self,
        system: str,
        messages: list[Message],
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Plain text completion. Returns the raw assistant text plus usage."""
        ...

    def chat_structured(
        self,
        system: str,
        messages: list[Message],
        schema: type[T],
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> tuple[T, LLMResponse]:
        """Structured completion. Returns a parsed Pydantic instance plus usage.

        Implementations:
            - Anthropic: forced tool use (`tool_choice={"type": "tool"}`)
            - OpenAI:    `response_format` with strict JSON schema

        Raises `LLMError` if the provider returns malformed structured data.
        """
        ...


# ============================================================
# Errors
# ============================================================


class LLMError(Exception):
    """Base error for the LLM layer."""


class LLMStructuredOutputError(LLMError):
    """Raised when the model returns malformed structured output."""
