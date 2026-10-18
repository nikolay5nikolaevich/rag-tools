"""Прайс на модели Claude, $/1M токенов. Нормативный расчёт `billed_cost_usd` — по
токенам НАШЕГО промпта и ответа (docs/CONTRACTS.md §9), не по `total_cost_usd` из
cli-бэкенда: в тот входит служебный оверхед headless-режима.

Ставки кэша — стандартное соотношение Anthropic для TTL 5 минут (умолчание
`cache_control: {"type": "ephemeral"}`, без явного `ttl: "1h"`): запись = 1.25× цены
входа, чтение = 0.1× цены входа.
"""

from __future__ import annotations

from pydantic import BaseModel

PRICE_DATE = "2026-06-24"  # дата, на которую сверена таблица (сверка — скилл claude-api)

_CACHE_WRITE_MULT = 1.25
_CACHE_READ_MULT = 0.1


class ModelPrice(BaseModel):
    input_per_mtok: float
    output_per_mtok: float
    cache_write_per_mtok: float
    cache_read_per_mtok: float


def _price(input_per_mtok: float, output_per_mtok: float) -> ModelPrice:
    return ModelPrice(
        input_per_mtok=input_per_mtok,
        output_per_mtok=output_per_mtok,
        cache_write_per_mtok=input_per_mtok * _CACHE_WRITE_MULT,
        cache_read_per_mtok=input_per_mtok * _CACHE_READ_MULT,
    )


# Модели, реально используемые проектом (EV_GEN_MODEL, EV_JUDGE_MODEL, CONTRACTS.md §9)
# плюс haiku — дешёвая модель для замеров оверхеда и тестов.
PRICES: dict[str, ModelPrice] = {
    "claude-opus-5": _price(5.00, 25.00),
    "claude-sonnet-5": _price(2.00, 10.00),
    "claude-haiku-4-5-20251001": _price(1.00, 5.00),
}


class UnknownModelPriceError(KeyError):
    pass


def billed_cost_usd(
    *,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float:
    """Стоимость по прайсу от собственных токенов — не то же самое, что raw_cost_usd бэкенда."""
    try:
        price = PRICES[model]
    except KeyError as e:
        raise UnknownModelPriceError(
            f"нет цены для модели {model!r} в pricing.py — добавь строку в таблицу PRICES"
        ) from e
    return (
        input_tokens * price.input_per_mtok
        + output_tokens * price.output_per_mtok
        + cache_read_tokens * price.cache_read_per_mtok
        + cache_write_tokens * price.cache_write_per_mtok
    ) / 1_000_000
