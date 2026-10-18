"""Четыре способа ответить на один и тот же вопрос — предмет сравнения T-30.

| вариант       | что получает модель                        | recall применим |
|---------------|--------------------------------------------|-----------------|
| `no_search`   | только вопрос                              | нет: поиска нет |
| `fts`         | top-k полнотекстового поиска               | да              |
| `rag`         | top-k выбранного режима (по умолчанию RRF) | да              |
| `full_corpus` | все видимые чанки корпуса                  | нет: поиск не участвует, выдача тривиально полна |

Второго конвейера нет: ищет `search.py`, собирает и проверяет ответ `answering.py`,
в модель ходит `llm/`. Варианты отличаются только тем, какие фрагменты попадают
в промпт, и один из них не получает их вовсе.

Вариант `no_search` работает по отдельному промпту `{prompt_version}-noctx`: правила
совпадают с основным всюду, кроме фрагментов и ссылок [n], которых у него нет. Так
сравнивается доступ к корпусу, а не удачность формулировок промпта.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence

import psycopg
from pydantic import BaseModel, ConfigDict

from ergovault import answering, db
from ergovault.answering import MAX_OUTPUT_TOKENS, AnswerValidationError
from ergovault.llm import LLMProvider, get_provider
from ergovault.models import AnswerPayload, Hit, Profile, Source
from ergovault.search import search
from ergovault.settings import Settings, get_settings

Variant = str
VARIANTS: tuple[Variant, ...] = ("no_search", "fts", "rag", "full_corpus")
# Recall и MRR определены только там, где выдачу формировал поиск. У `full_corpus`
# в контексте лежит весь корпус: «нашлось всё» — свойство конструкции, а не измерение.
SEARCH_VARIANTS: tuple[Variant, ...] = ("fts", "rag")
NO_CONTEXT_SUFFIX = "-noctx"

VARIANT_TITLES = {
    "no_search": "модель без поиска",
    "fts": "полнотекстовый поиск",
    "rag": "выбранный RAG",
    "full_corpus": "весь корпус в контексте",
}

MISSING_NO_SEARCH = "вопрос задан без доступа к корпусу"


class VariantAnswer(BaseModel):
    """Ответ варианта вместе с тем, что было ему передано."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    variant: Variant
    payload: AnswerPayload
    hits: list[Hit]
    retrieved_source_ids: list[str]
    search_latency_ms: float | None  # None — поиск не выполнялся


def retrieval_mode_of(variant: Variant, settings: Settings) -> str:
    """Что записать в `retrieval_mode` прогона (CONTRACTS.md §10). Для вариантов без
    поиска это не режим `search.py`, а честное имя способа получить контекст."""
    if variant == "rag":
        return settings.retrieval_mode
    if variant == "fts":
        return "fts"
    return variant


def _dedup_source_ids(hits: Sequence[Hit]) -> list[str]:
    seen: list[str] = []
    for hit in hits:
        if hit.source.id not in seen:
            seen.append(hit.source.id)
    return seen


def corpus_hits(conn: psycopg.Connection, profile: Profile) -> list[Hit]:
    """Весь видимый корпус как список Hit — контекст варианта `full_corpus`.

    Порядок фиксирован (источник, затем ordinal), score и components пусты: ранжирования
    здесь нет, номер фрагмента ничего не значит кроме позиции в промпте.
    """
    return [
        Hit(chunk=chunk, source=source, score=0.0, rank=rank, components={})
        for rank, (chunk, source) in enumerate(db.list_chunk_views(conn, profile), start=1)
    ]


def retrieve(
    conn: psycopg.Connection,
    variant: Variant,
    question: str,
    *,
    profile: Profile,
    k: int,
    settings: Settings,
    corpus: Sequence[Hit] | None = None,
) -> tuple[list[Hit], float | None]:
    """Фрагменты варианта и время их получения (None — поиска не было).

    `corpus` — уже вычитанный корпус для `full_corpus`: читать 709 чанков заново на каждый
    вопрос незачем, а время такого чтения не относится к сравнению подходов.
    """
    if variant == "no_search":
        return [], None
    if variant == "full_corpus":
        return list(corpus if corpus is not None else corpus_hits(conn, profile)), None

    mode = "fts" if variant == "fts" else settings.retrieval_mode
    started = time.perf_counter()
    hits = search(conn, question, k, profile=profile, mode=mode, settings=settings)
    return hits, (time.perf_counter() - started) * 1000


def answer_without_search(
    question: str,
    *,
    settings: Settings | None = None,
    provider: LLMProvider | None = None,
) -> AnswerPayload:
    """Ответ модели без единого фрагмента корпуса — базовый вариант сравнения.

    Проверка та же, что у остальных вариантов, за вычетом невозможного: ссылок [n] быть
    не может, цитаты сверять не с чем, ограничения из метаданных не вычисляются, потому
    что источников нет. Отказ остаётся отказом: `refused=True` и `no_data`.
    """
    settings = settings or get_settings()
    started = time.monotonic()
    version = f"{settings.prompt_version}{NO_CONTEXT_SUFFIX}"
    system, user_template, prompt_hash = answering.load_prompt(version)
    user = user_template.substitute(question=question.strip())

    response = (provider or get_provider()).complete(
        system=system,
        user=user,
        model=settings.gen_model,
        cache_system=True,
        max_tokens=MAX_OUTPUT_TOKENS,
    )
    meta = {
        "corpus_version": settings.corpus_version,
        "retrieval_mode": "no_search",
        "prompt_chars": len(system) + len(user),
        "prompt_version": version,
        "prompt_hash": prompt_hash,
        "model": response.model,
        "llm_backend": response.backend,
        "llm_latency_ms": response.latency_ms,
        "cost_usd": response.billed_cost_usd,
        "raw_cost_usd": response.raw_cost_usd,
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "cache_read_tokens": response.cache_read_tokens,
        "cache_write_tokens": response.cache_write_tokens,
    }

    try:
        parsed = answering.parse_model_output(response.text)
        text = parsed.answer_markdown.strip()
        if not text:
            raise AnswerValidationError("пустой answer_markdown: отказ тоже объясняет, чего нет")
        if parsed.cited or answering.REF.search(text):
            raise AnswerValidationError(
                "ссылки [n] в варианте без поиска: фрагментов не передавалось"
            )
    except AnswerValidationError as exc:
        exc.raw_text = response.text
        exc.meta = {**meta, "latency_ms": int((time.monotonic() - started) * 1000)}
        raise

    limitations = answering.compute_limitations(
        [], refused=parsed.refused, sources={}, missing=MISSING_NO_SEARCH
    )
    meta["latency_ms"] = int((time.monotonic() - started) * 1000)
    return AnswerPayload(
        answer_markdown=text, limitations=limitations, refused=parsed.refused, meta=meta
    )


def answer_variant(
    conn: psycopg.Connection,
    variant: Variant,
    question: str,
    *,
    sources: Mapping[str, Source],
    profile: Profile,
    k: int,
    settings: Settings | None = None,
    provider: LLMProvider | None = None,
    corpus: Sequence[Hit] | None = None,
    retrieved: tuple[list[Hit], float | None] | None = None,
) -> VariantAnswer:
    """Один вариант на одном вопросе. Исключения те же, что у `answering.answer_hits`.

    `retrieved` — уже полученная выдача (`retrieve`): оценка сохраняет её до вызова
    модели, чтобы recall не зависел от того, состоялась ли генерация.
    """
    if variant not in VARIANTS:
        raise ValueError(f"неизвестный вариант {variant!r}; допустимы {VARIANTS}")
    settings = settings or get_settings()

    hits, search_latency_ms = retrieved or retrieve(
        conn, variant, question, profile=profile, k=k, settings=settings, corpus=corpus
    )
    if variant == "no_search":
        payload = answer_without_search(question, settings=settings, provider=provider)
    else:
        payload = answering.answer_hits(
            question,
            hits,
            sources=sources,
            retrieval_mode=retrieval_mode_of(variant, settings),
            settings=settings,
            provider=provider,
        )
    payload.meta["variant"] = variant
    if search_latency_ms is not None:
        payload.meta["search_latency_ms"] = search_latency_ms
    return VariantAnswer(
        variant=variant,
        payload=payload,
        hits=list(hits),
        retrieved_source_ids=_dedup_source_ids(hits),
        search_latency_ms=search_latency_ms,
    )
