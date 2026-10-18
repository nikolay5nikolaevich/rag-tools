"""Запустить одну функцию поиска на проверочных вопросах и записать результаты."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from pathlib import Path

from pydantic import BaseModel

from ergovault.eval.dataset import GoldenQuestion
from ergovault.models import AnswerPayload, Hit, Profile
from ergovault.settings import Settings, get_settings

# Генератор — необязательный шаг «полного пайплайна» (answering.answer_hits). Раннер
# не создаёт и не имитирует его сам: если вызывающая сторона не передала generate,
# метрики отказа честно помечаются как недоступные, а не подгоняются под прокси-эвристику
# по retrieval-сигналу (CLAUDE.md: "не подгонять метрику под красивую цифру").
GenerateFn = Callable[[GoldenQuestion, list[Hit]], AnswerPayload]


class RetrievedItem(BaseModel):
    rank: int
    chunk_id: str
    source_id: str
    score: float


class RunRow(BaseModel):
    question_id: str
    category: str
    question: str
    answerable: bool
    expected_source_ids: list[str]
    retrieved: list[RetrievedItem]
    retrieved_source_ids: list[str]  # без повторов, в порядке первого (лучшего) появления
    empty: bool
    latency_ms: float | None = None  # только поиск, без генерации; None — не замерялось
    refused: bool | None = None
    # Версионирование — CONTRACTS.md §10: без этих четырёх полей прогон недействителен.
    corpus_version: str
    prompt_version: str
    retrieval_mode: str
    model: str | None = None  # None: генератор не запускался, модель не вызывалась


def _dedup_source_ids(hits: Sequence[Hit]) -> list[str]:
    seen: list[str] = []
    for hit in hits:
        if hit.source.id not in seen:
            seen.append(hit.source.id)
    return seen


def run(
    questions: Sequence[GoldenQuestion],
    search: Callable[..., list[Hit]],
    *,
    mode: str,
    k: int = 5,
    profile: Profile = "public",
    settings: Settings | None = None,
    generate: GenerateFn | None = None,
) -> list[RunRow]:
    settings = settings or get_settings()
    rows: list[RunRow] = []
    for q in questions:
        started = time.perf_counter()
        hits = search(q.question, k, profile=profile)
        latency_ms = (time.perf_counter() - started) * 1000

        refused: bool | None = None
        model: str | None = None
        if generate is not None:
            answer = generate(q, hits)
            refused = answer.refused
            model = answer.meta.get("model") or settings.gen_model

        rows.append(
            RunRow(
                question_id=q.id,
                category=q.category,
                question=q.question,
                answerable=q.answerable,
                expected_source_ids=q.expected_source_ids,
                retrieved=[
                    RetrievedItem(
                        rank=hit.rank,
                        chunk_id=hit.chunk.id,
                        source_id=hit.source.id,
                        score=hit.score,
                    )
                    for hit in hits
                ],
                retrieved_source_ids=_dedup_source_ids(hits),
                empty=len(hits) == 0,
                latency_ms=latency_ms,
                refused=refused,
                corpus_version=settings.corpus_version,
                prompt_version=settings.prompt_version,
                retrieval_mode=mode,
                model=model,
            )
        )
    return rows


def save_run_jsonl(rows: Sequence[RunRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(row.model_dump_json() + "\n")
