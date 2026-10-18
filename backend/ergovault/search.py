"""Поиск фрагментов: текст, векторы или объединение рангов (RRF)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

import psycopg

from ergovault import db
from ergovault.models import Hit, Profile
from ergovault.settings import Settings, get_settings

RetrievalMode = Literal["fts", "vector", "hybrid"]
MODES = ("fts", "vector", "hybrid")
CANDIDATE_MULTIPLIER = 4
MIN_CANDIDATES = 20
RANK_NOT_FOUND = 0.0
# Сколько чанков одного источника попадает в top-k. Без ограничения один источник
# занимал до четырёх из пяти мест; лимит 2 на проверочном наборе не отличался от
# отсутствия лимита (eval/reports/retrieval-comparison.md).
MAX_CHUNKS_PER_SOURCE = 1


def search(
    conn: psycopg.Connection,
    query: str,
    k: int = 10,
    *,
    profile: Profile,
    mode: RetrievalMode | None = None,
    settings: Settings | None = None,
    encode_query: Callable[[str], Sequence[float]] | None = None,
) -> list[Hit]:
    """Один вход для приложения и eval. Модель нужна только векторной ветви."""
    settings = settings or get_settings()
    mode = mode or settings.retrieval_mode
    if mode not in MODES:
        raise ValueError(f"неизвестный режим поиска {mode!r}; допустимы {MODES}")
    if k < 1:
        raise ValueError("k должен быть положительным")
    if profile not in ("public", "pilot"):
        raise ValueError("profile должен быть public или pilot")
    # Запас кандидатов нужен во всех режимах: после ограничения чанков на источник
    # top-k заполняется следующими источниками, а не остаётся коротким.
    depth = max(k * CANDIDATE_MULTIPLIER, MIN_CANDIDATES)
    fts_hits = []
    if mode in ("fts", "hybrid"):
        fts_hits = db.search_fts(
            conn,
            query,
            depth,
            profile=profile,
            weight_ru=settings.weight_ru,
            weight_simple=settings.weight_simple,
        )
        if mode == "fts":
            return limit_per_source(fts_hits, k)

    if encode_query is None:
        from ergovault.embeddings import get_model

        model = get_model(settings.embed_model, settings.embed_dim)
        embedding = model.encode_queries([query])[0].tolist()
    else:
        embedding = encode_query(query)
    vector_hits = db.search_vector(conn, embedding, depth, profile=profile)
    if mode == "vector":
        return limit_per_source(vector_hits, k)
    fused = fuse(fts_hits, vector_hits, k=len(fts_hits) + len(vector_hits), k_rrf=settings.rrf_k)
    return limit_per_source(fused, k)


def limit_per_source(
    hits: list[Hit], k: int, *, per_source: int = MAX_CHUNKS_PER_SOURCE
) -> list[Hit]:
    """Оставить не больше `per_source` чанков источника и первые k; ранги пересчитать.

    Остаётся лучший по рангу чанк источника. Оригинал видео и его пересказ — разные
    источники и здесь не схлопываются: на проверочном наборе это теряло строки, нужные
    для цитат, а независимость подтверждений проверяет ответ (CONTRACTS.md §7).
    """
    taken: dict[str, int] = {}
    top: list[Hit] = []
    for hit in hits:
        count = taken.get(hit.source.id, 0)
        if count >= per_source:
            continue
        taken[hit.source.id] = count + 1
        hit.rank = len(top) + 1
        top.append(hit)
        if len(top) == k:
            break
    return top


def fuse(
    fts_hits: list[Hit],
    vector_hits: list[Hit],
    *,
    k: int,
    k_rrf: int = 60,
    weight_fts: float = 1.0,
    weight_vector: float = 1.0,
) -> list[Hit]:
    """Сложить ранги ветвей, сохранить вклады и стабильно отсортировать top-k."""

    def _contribution(rank: int | None, weight: float) -> float:
        return 0.0 if rank is None else weight / (k_rrf + rank)

    by_fts = {hit.chunk.id: hit for hit in fts_hits}
    by_vector = {hit.chunk.id: hit for hit in vector_hits}

    fused: list[Hit] = []
    for chunk_id in list(by_fts) + [cid for cid in by_vector if cid not in by_fts]:
        fts_hit = by_fts.get(chunk_id)
        vector_hit = by_vector.get(chunk_id)
        carrier = fts_hit or vector_hit  # обе ветви несут один и тот же Chunk/Source

        rrf_fts = _contribution(fts_hit.rank if fts_hit else None, weight_fts)
        rrf_vector = _contribution(vector_hit.rank if vector_hit else None, weight_vector)

        # Сырые оценки обеих ветвей остаются рядом с вкладами RRF: по ним видно не
        # только «кто поднял документ», но и почему — совпало имя собственное в
        # simple-ветви или сработала морфология либо семантика.
        components = {
            "rrf_fts": rrf_fts,
            "rrf_vector": rrf_vector,
            "rank_fts": float(fts_hit.rank) if fts_hit else RANK_NOT_FOUND,
            "rank_vector": float(vector_hit.rank) if vector_hit else RANK_NOT_FOUND,
            "fts_ru": fts_hit.components.get("fts_ru", 0.0) if fts_hit else 0.0,
            "fts_simple": fts_hit.components.get("fts_simple", 0.0) if fts_hit else 0.0,
            "vector": vector_hit.components.get("vector", 0.0) if vector_hit else 0.0,
        }
        fused.append(
            Hit(
                chunk=carrier.chunk,
                source=carrier.source,
                score=rrf_fts + rrf_vector,
                rank=1,  # настоящий ранг проставляется ниже, после сортировки
                components=components,
            )
        )

    # Тай-брейк по chunk_id: одинаковые суммы вкладов у RRF — обычное дело (документ,
    # найденный только одной ветвью и на том же ранге, что и соседний), а порядок
    # выдачи обязан быть воспроизводимым между прогонами.
    fused.sort(key=lambda hit: (-hit.score, hit.chunk.id))
    top = fused[:k]
    for rank, hit in enumerate(top, start=1):
        hit.rank = rank
    return top
