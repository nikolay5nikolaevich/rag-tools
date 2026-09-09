"""Бэкенд `api`: прямой вызов Anthropic SDK с prompt caching. Опционален — включается
только при заданном EV_ANTHROPIC_API_KEY (docs/CONTRACTS.md §9): для чистых замеров
стоимости в T-30 и для запуска рецензентом без подписки. Отсутствие ключа — не ошибка
этого модуля, переключение на cli делает фабрика в ergovault/llm/__init__.py.
"""

from __future__ import annotations

import time

import anthropic

from ergovault.llm.base import LLMResponse
from ergovault.llm.pricing import billed_cost_usd


class ApiLLMProvider:
    """Провайдер через Anthropic SDK. cache_system ставит cache_control на системный промпт."""

    def __init__(self, *, api_key: str) -> None:
        self._client = anthropic.Anthropic(api_key=api_key)

    def complete(
        self,
        *,
        system: str,
        user: str,
        model: str,
        cache_system: bool = False,
        max_tokens: int = 4000,
    ) -> LLMResponse:
        system_param: str | list[dict] = (
            [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
            if cache_system
            else system
        )

        start = time.monotonic()
        message = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_param,
            messages=[{"role": "user", "content": user}],
        )
        latency_ms = int((time.monotonic() - start) * 1000)

        text = "".join(block.text for block in message.content if block.type == "text")
        usage = message.usage
        input_tokens = usage.input_tokens
        output_tokens = usage.output_tokens
        cache_read = usage.cache_read_input_tokens or 0
        cache_write = usage.cache_creation_input_tokens or 0

        return LLMResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
            latency_ms=latency_ms,
            backend="api",
            model=model,
            raw_cost_usd=None,  # Anthropic SDK не сообщает total_cost_usd по запросу
            billed_cost_usd=billed_cost_usd(
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_tokens=cache_read,
                cache_write_tokens=cache_write,
            ),
        )
