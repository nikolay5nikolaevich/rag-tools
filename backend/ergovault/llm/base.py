"""Интерфейс LLM-провайдера. Нормативно — docs/CONTRACTS.md §9, docs/tickets/T-15-llm-provider.md.

Весь код, которому нужен вызов модели, ходит только через `LLMProvider.complete()`.
Прямые обращения к Anthropic SDK или к `claude` CLI из других пакетов запрещены —
это единственная точка входа в модель для T-21 (генерация) и T-23 (судья).
"""

from __future__ import annotations

from typing import Literal, Protocol

from pydantic import BaseModel


class LLMResponse(BaseModel):
    text: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    latency_ms: int
    backend: Literal["cli", "api"]
    model: str
    raw_cost_usd: float | None
    billed_cost_usd: float


class LLMProvider(Protocol):
    def complete(
        self,
        *,
        system: str,
        user: str,
        model: str,
        cache_system: bool = False,
        max_tokens: int = 4000,
    ) -> LLMResponse: ...
