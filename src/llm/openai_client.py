"""OpenAI (GPT) LLM client.

Wraps the official `openai` SDK and conforms to the `LLMClient` Protocol.

Structured outputs use the `response_format={"type": "json_schema", ...}`
mechanism with `strict=True`, which constrains the model to a JSON Schema
matching our Pydantic schema.

OpenAI does not have prompt caching exposed via SDK in the same way as
Anthropic (caching is automatic but not surfaced to the user); we just leave
those fields at 0.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Literal, TypeVar

import openai
from pydantic import BaseModel, ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import OPENAI_API_KEY
from src.llm.base import (
    LLMResponse,
    LLMStructuredOutputError,
    Message,
    estimate_cost_usd,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


_TRANSIENT_ERRORS = (
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.RateLimitError,
    openai.InternalServerError,
)


class OpenAIClient:
    """LLMClient implementation backed by OpenAI's API."""

    def __init__(self, model: str, api_key: str | None = None):
        """Init.

        Args:
            model: e.g. "gpt-4o-mini", "gpt-4o".
            api_key: optional override; defaults to env OPENAI_API_KEY.
        """
        key = api_key or OPENAI_API_KEY
        if not key:
            raise ValueError("OPENAI_API_KEY is missing — set it in .env")
        self._client = openai.OpenAI(api_key=key)
        self._model = model

    @property
    def provider(self) -> Literal["anthropic", "openai"]:
        return "openai"

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
        raw = self._call(
            system=system,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=None,
        )
        text = raw.choices[0].message.content or ""
        return self._build_response(raw, content=text)

    def chat_structured(
        self,
        system: str,
        messages: list[Message],
        schema: type[T],
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> tuple[T, LLMResponse]:
        """Structured completion via JSON Schema with strict mode."""
        json_schema = _to_strict_json_schema(schema)
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": schema.__name__,
                "schema": json_schema,
                "strict": True,
            },
        }

        raw = self._call(
            system=system,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )

        text = raw.choices[0].message.content or ""
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMStructuredOutputError(
                f"OpenAI returned invalid JSON: {text[:200]!r}"
            ) from exc

        try:
            parsed = schema.model_validate(payload)
        except ValidationError as exc:
            raise LLMStructuredOutputError(
                f"OpenAI output failed Pydantic validation: {exc}"
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
        response_format: dict[str, Any] | None,
    ):
        api_messages = [{"role": "system", "content": system}] if system else []
        api_messages.extend(m.model_dump() for m in messages)

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": api_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format

        return self._client.chat.completions.create(**kwargs)

    def _build_response(self, raw, content: str) -> LLMResponse:
        usage = raw.usage
        input_tokens = getattr(usage, "prompt_tokens", 0) or 0
        output_tokens = getattr(usage, "completion_tokens", 0) or 0

        cost = estimate_cost_usd(
            model=self._model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        return LLMResponse(
            content=content,
            model=self._model,
            provider="openai",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )


# ============================================================
# JSON Schema utilities for OpenAI strict mode
# ============================================================


def _to_strict_json_schema(schema: type[BaseModel]) -> dict[str, Any]:
    """Convert a Pydantic schema to the JSON Schema flavor OpenAI expects.

    OpenAI strict mode requires:
        - additionalProperties: false on every object
        - all properties must be in `required`
    Pydantic emits standard JSON Schema; we tighten it here.
    """
    base = schema.model_json_schema()
    return _tighten(base)


def _tighten(node: Any) -> Any:
    if isinstance(node, dict):
        node = {k: _tighten(v) for k, v in node.items()}
        if node.get("type") == "object" and "properties" in node:
            node["additionalProperties"] = False
            node["required"] = list(node["properties"].keys())
        return node
    if isinstance(node, list):
        return [_tighten(item) for item in node]
    return node
