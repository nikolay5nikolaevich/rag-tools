"""Фабрика LLM-провайдера. Единственная точка входа в модель для остальных пакетов
(T-21, T-23) — они импортируют `get_provider()`, не бэкенды напрямую (docs/CONTRACTS.md §9).

`EV_LLM_BACKEND=cli|api`, по умолчанию `cli` — работает по подписке, без ключа.
Бэкенд `api` включается, только если явно выбран И задан EV_ANTHROPIC_API_KEY;
отсутствие ключа — не ошибка, а переключение на cli.
"""

from __future__ import annotations

from ergovault.llm.base import LLMProvider, LLMResponse
from ergovault.llm.cli_backend import CliLLMProvider, LLMError, NotLoggedInError
from ergovault.settings import get_settings

__all__ = ["LLMError", "LLMProvider", "LLMResponse", "NotLoggedInError", "get_provider"]


def get_provider() -> LLMProvider:
    settings = get_settings()
    if settings.llm_backend == "api" and settings.anthropic_api_key:
        from ergovault.llm.api_backend import ApiLLMProvider

        return ApiLLMProvider(api_key=settings.anthropic_api_key)
    return CliLLMProvider()
