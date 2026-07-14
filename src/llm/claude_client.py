"""Claude (Anthropic) LLM client.

Wraps the official `anthropic` SDK and conforms to the `LLMClient` Protocol.

Key features:
    - Prompt caching enabled by default on the system prompt (90% off on
      cache reads, ~5 min TTL). System prompts in this project are reused
      across many articles (e.g., the Summarizer hits the same system prompt
      N times in a single run), so caching is a critical cost lever.
    - Forced tool use for structured outputs: more reliable than hoping the
      model returns valid JSON in plain text.
    - Retry with exponential backoff on transient errors.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Literal, TypeVar

import anthropic
from pydantic import BaseModel, ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import ANTHROPIC_API_KEY
from src.llm.base import (
    LLMResponse,
    LLMStructuredOutputError,
    Message,
    estimate_cost_usd,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Models that no longer accept the `temperature` parameter.
# Claude 4 "thinking" models and some Opus variants have removed this knob.
# Sending it causes HTTP 400: "'temperature' is deprecated for this model."
_NO_TEMPERATURE_MODELS: frozenset[str] = frozenset({
    "claude-opus-4-7",
    "claude-opus-4-5",
})

# Models that are unreliable with forced tool use (return empty {} or wrap
# output under a phantom key like "$PARAMETER_NAME").
# For these models we fall back to schema-in-system-prompt + JSON parsing,
# exactly like GroqClient does for Llama models.
_NO_TOOL_USE_MODELS: frozenset[str] = frozenset({
    "claude-opus-4-7",
    "claude-opus-4-5",
})

# Models that use extended thinking (interleaved reasoning).
# Without `thinking={"type": "enabled", "budget_tokens": N}` they return ONLY
# thinking blocks and no text blocks → _extract_text returns "" → empty JSON.
# We must enable thinking explicitly to get a text block with the final answer.
_EXTENDED_THINKING_MODELS: frozenset[str] = frozenset({
    "claude-opus-4-7",
    "claude-opus-4-5",
})

# Minimum thinking budget (tokens).  Kept small for the Judge use-case:
# we only need faithfulness + coverage + 1-3 sentence reasoning.
# Raise if you use claude-opus-4-7 for complex tasks (synthesize, trends…).
THINKING_BUDGET_TOKENS = 1024


_TRANSIENT_ERRORS = (
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    anthropic.RateLimitError,
    anthropic.InternalServerError,
)


class ClaudeClient:
    """LLMClient implementation backed by Anthropic's API."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        cache_system: bool = True,
    ):
        """Init.

        Args:
            model: e.g. "claude-haiku-4-5", "claude-sonnet-4-6", "claude-opus-4-7".
            api_key: optional override; defaults to env ANTHROPIC_API_KEY.
            cache_system: if True (default), tags the system prompt with
                `cache_control: ephemeral` so subsequent calls within ~5 min
                get cached input pricing.
        """
        key = api_key or ANTHROPIC_API_KEY
        if not key:
            raise ValueError("ANTHROPIC_API_KEY is missing — set it in .env")
        self._client = anthropic.Anthropic(api_key=key)
        self._model = model
        self._cache_system = cache_system

    @property
    def provider(self) -> Literal["anthropic", "openai"]:
        return "anthropic"

    @property
    def model(self) -> str:
        return self._model

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------

    def chat(
        self,
        system: str,
        messages: list[Message],
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Plain text completion."""
        raw = self._call(
            system=system,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=None,
        )
        text = self._extract_text(raw)
        return self._build_response(raw, content=text)

    def chat_structured(
        self,
        system: str,
        messages: list[Message],
        schema: type[T],
        temperature: float = 0.0,
        max_tokens: int = 2048,
        max_attempts: int = 3,
    ) -> tuple[T, LLMResponse]:
        """Structured completion — strategy depends on the model.

        Standard models (Haiku, Sonnet…): forced tool use — the model is
        required to emit a tool_use block, which is then validated with Pydantic.

        Claude 4 thinking models (Opus 4.x…): tool use is unreliable (returns
        empty dicts or phantom-key wrappers).  We fall back to schema-in-system-
        prompt + JSON parsing, the same strategy as GroqClient for Llama models.

        All models: retried up to `max_attempts` times on LLMStructuredOutputError
        (e.g. empty tool input {}, missing fields).  Transient API failures are
        handled separately by the tenacity retry in _call().
        """
        last_exc: Exception = RuntimeError("unreachable")
        for attempt in range(1, max_attempts + 1):
            try:
                if self._model in _NO_TOOL_USE_MODELS:
                    return self._chat_structured_json_mode(
                        system=system,
                        messages=messages,
                        schema=schema,
                        max_tokens=max_tokens,
                    )
                return self._chat_structured_tool_use(
                    system=system,
                    messages=messages,
                    schema=schema,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except LLMStructuredOutputError as exc:
                last_exc = exc
                if attempt < max_attempts:
                    logger.warning(
                        "chat_structured attempt %d/%d failed for %s: %s — retrying",
                        attempt, max_attempts, self._model, exc,
                    )
        raise last_exc

    # ------------------------------------------------------------
    # chat_structured back-ends
    # ------------------------------------------------------------

    def _chat_structured_tool_use(
        self,
        system: str,
        messages: list[Message],
        schema: type[T],
        temperature: float,
        max_tokens: int,
    ) -> tuple[T, LLMResponse]:
        """Structured output via forced tool use (standard Claude models)."""
        tool_name = "emit_" + schema.__name__.lower()
        tool = {
            "name": tool_name,
            "description": f"Emit a structured {schema.__name__} object.",
            "input_schema": schema.model_json_schema(),
        }

        raw = self._call(
            system=system,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=[tool],
            tool_choice={"type": "tool", "name": tool_name},
        )

        # Locate the tool_use block in the response.
        tool_input = None
        for block in raw.content:
            if getattr(block, "type", None) == "tool_use" and block.name == tool_name:
                tool_input = block.input
                break

        if tool_input is None:
            raise LLMStructuredOutputError(
                f"Claude did not emit the expected tool {tool_name!r}"
            )

        try:
            parsed = schema.model_validate(tool_input)
        except ValidationError as exc:
            raise LLMStructuredOutputError(
                f"Tool output failed Pydantic validation: {exc}"
            ) from exc

        response = self._build_response(raw, content=json.dumps(tool_input))
        return parsed, response

    def _chat_structured_json_mode(
        self,
        system: str,
        messages: list[Message],
        schema: type[T],
        max_tokens: int,
    ) -> tuple[T, LLMResponse]:
        """Structured output via schema-in-system-prompt + JSON parsing.

        Used for Claude 4 thinking models (e.g. claude-opus-4-7) that do not
        handle forced tool use reliably.  temperature is intentionally omitted
        (deprecated for these models).
        """
        schema_json = json.dumps(schema.model_json_schema(), indent=2)
        enhanced_system = (
            f"{system}\n\n"
            f"CRITICAL INSTRUCTION: Your response MUST be a valid JSON object "
            f"that exactly matches this schema. Output ONLY the JSON — no "
            f"markdown fences, no explanation, no extra keys.\n\n"
            f"Required JSON schema:\n{schema_json}"
        )

        raw = self._call(
            system=enhanced_system,
            messages=messages,
            temperature=0.0,   # will be stripped by _call for _NO_TEMPERATURE_MODELS
            max_tokens=max_tokens,
            tools=None,
        )

        text = _extract_json(self._extract_text(raw))

        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMStructuredOutputError(
                f"Claude/{self._model} returned invalid JSON: {text[:300]!r}"
            ) from exc

        try:
            parsed = schema.model_validate(payload)
        except ValidationError as exc:
            raise LLMStructuredOutputError(
                f"Claude/{self._model} JSON output failed Pydantic validation: {exc}\n"
                f"Raw JSON: {text[:300]}"
            ) from exc

        return parsed, self._build_response(raw, content=text)

    # ------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------

    @retry(
        retry=retry_if_exception_type(_TRANSIENT_ERRORS),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        reraise=True,
    )
    def _call(
        self,
        system: str,
        messages: list[Message],
        temperature: float,
        max_tokens: int,
        tools: list[dict[str, Any]] | None,
        tool_choice: dict[str, Any] | None = None,
    ):
        """Single call to Anthropic's Messages API, with retry on transients."""
        # Anthropic accepts system as a list of blocks for cache_control.
        system_param: Any
        if self._cache_system and system:
            system_param = [
                {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
            ]
        else:
            system_param = system

        # Extended thinking models require `thinking` to produce a text block.
        # Without it they return ONLY thinking blocks → _extract_text → "".
        # We also ensure max_tokens > budget_tokens (API requirement).
        effective_max_tokens = max_tokens
        if self._model in _EXTENDED_THINKING_MODELS:
            effective_max_tokens = max(max_tokens, THINKING_BUDGET_TOKENS + 512)

        kwargs: dict[str, Any] = {
            "model": self._model,
            "system": system_param,
            "messages": [m.model_dump() for m in messages],
            "max_tokens": effective_max_tokens,
        }
        # Some models (e.g. claude-opus-4-7) have deprecated the `temperature`
        # parameter and return HTTP 400 if it is included.  Only add it when
        # the model is known to accept it.
        if self._model not in _NO_TEMPERATURE_MODELS:
            kwargs["temperature"] = temperature
        # Enable extended thinking for models that require it.
        if self._model in _EXTENDED_THINKING_MODELS:
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": THINKING_BUDGET_TOKENS,
            }
        if tools is not None:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice

        return self._client.messages.create(**kwargs)

    @staticmethod
    def _extract_text(raw) -> str:
        """Concatenate all `text` blocks from the response content."""
        chunks: list[str] = []
        for block in raw.content:
            if getattr(block, "type", None) == "text":
                chunks.append(block.text)
        return "".join(chunks)

    def _build_response(self, raw, content: str) -> LLMResponse:
        """Wrap an Anthropic message into our `LLMResponse`."""
        usage = raw.usage
        input_tokens = getattr(usage, "input_tokens", 0) or 0
        output_tokens = getattr(usage, "output_tokens", 0) or 0
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_creation = getattr(usage, "cache_creation_input_tokens", 0) or 0

        cost = estimate_cost_usd(
            model=self._model,
            input_tokens=input_tokens + cache_read + cache_creation,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read,
            cache_creation_tokens=cache_creation,
        )

        return LLMResponse(
            content=content,
            model=self._model,
            provider="anthropic",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read,
            cache_creation_tokens=cache_creation,
            cost_usd=cost,
        )


# ============================================================
# Module-level helper
# ============================================================


def _extract_json(text: str) -> str:
    """Strip markdown code fences and isolate the JSON object.

    Handles:
        - ```json { ... } ```
        - ``` { ... } ```
        - Plain { ... }
    """
    for fence in ("```json", "```"):
        if fence in text:
            start = text.find(fence) + len(fence)
            end = text.rfind("```")
            if end > start:
                text = text[start:end].strip()
                break

    # Find the first { and last } to isolate the JSON object
    start_idx = text.find("{")
    end_idx = text.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        text = text[start_idx: end_idx + 1]

    return text.strip()
