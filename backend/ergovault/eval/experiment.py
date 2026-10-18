"""Сравнение четырёх подходов на одном наборе вопросов (T-30, шаги 2–4).

Прогон пишет сырые строки в `eval/runs/*.jsonl` по мере готовности: один вопрос —
один вызов модели, и терять часы работы из-за обрыва на середине нельзя. Метрики
и отчёт считаются из этих строк отдельной функцией, поэтому пересчёт и правка
формулировок отчёта не требуют новых вызовов модели.

Что здесь измеряется без судьи и потому публикуется сразу: попадание обязательных
формул `must_contain`, появление запрещённых `must_not_contain`, корректность отказов,
recall и MRR там, где был поиск, независимость процитированных источников (по
метаданным, а не по мнению модели), время, токены и расчётная стоимость.

Подтверждённость (доля утверждений, опирающихся на процитированный фрагмент) считает
`judge.py` и публикуется только вместе с согласием судьи с ручной разметкой. Пока
разметки нет, поле остаётся `None` и называется неизмеренным, а не нулём.
"""

from __future__ import annotations

import datetime
import json
import statistics
import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

import psycopg
from pydantic import BaseModel, Field

from ergovault import answering, db
from ergovault.answering import AnswerValidationError
from ergovault.eval.dataset import GoldenQuestion, load_golden_set
from ergovault.eval.judge import Calibration, JudgeConfig, judge_answer
from ergovault.eval.metrics import SAMPLE_SIZE_CAVEAT
from ergovault.eval.variants import (
    SEARCH_VARIANTS,
    VARIANT_TITLES,
    VARIANTS,
    Variant,
    answer_variant,
    corpus_hits,
    retrieval_mode_of,
    retrieve,
)
from ergovault.llm import LLMError
from ergovault.llm.cli_backend import CliLLMProvider
from ergovault.models import Profile
from ergovault.settings import REPO_ROOT, Settings, get_settings

RUNS_DIR = REPO_ROOT / "eval" / "runs"
REPORTS_DIR = REPO_ROOT / "eval" / "reports"
FINAL_REPORT_PATH = REPORTS_DIR / "final-comparison.md"
SUMMARY_PATH = REPORTS_DIR / "summary.json"
# Замер кэша и служебной части хранится отдельно: отчёт пересобирается из строк прогона
# без вызовов модели и должен подхватывать последний замер, а не терять его.
PROBE_PATH = REPORTS_DIR / "llm-probe.json"

# Весь корпус в контексте — это ~126 тыс. токенов на вопрос: стандартных 120 секунд
# CLI-бэкенду на такой вызов не хватает.
LONG_CALL_TIMEOUT_S = 900.0
# CLI отказывает волнами по несколько минут (замечено в финальном прогоне 12.09.2026):
# без паузы прогон за секунды проскакивает весь оставшийся список, собирая одни сбои.
LLM_FAILURE_COOLDOWN_S = 90.0


def save_probe(probe: CacheProbe, path: Path | None = None) -> None:
    path = path or PROBE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(probe.model_dump_json(indent=2) + "\n", encoding="utf-8")


def load_probe(path: Path | None = None) -> CacheProbe | None:
    path = path or PROBE_PATH
    if not path.is_file():
        return None
    return CacheProbe.model_validate_json(path.read_text(encoding="utf-8"))


def utc_stamp() -> str:
    return datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")


# --- сырые строки прогона ----------------------------------------------------------------


class VariantRow(BaseModel):
    """Один вопрос × один вариант × один повтор. Версии — CONTRACTS.md §10."""

    question_id: str
    category: str
    question: str
    answerable: bool
    variant: Variant
    repeat: int

    expected_source_ids: list[str] = Field(default_factory=list)
    retrieved_source_ids: list[str] = Field(default_factory=list)
    fragments: int = 0
    # Что лежало в промпте под номерами [1..n]: без этого нельзя ни собрать пары для
    # калибровки судьи, ни перепроверить цитату задним числом.
    fragment_chunk_ids: list[str] = Field(default_factory=list)
    recall_applicable: bool = False

    answer_markdown: str = ""
    refused: bool | None = None
    citation_source_ids: list[str] = Field(default_factory=list)
    limitation_kinds: list[str] = Field(default_factory=list)
    distinct_orgs: int = 0
    must_contain_missing: list[str] = Field(default_factory=list)
    must_not_contain_present: list[str] = Field(default_factory=list)
    claim_verdicts: list[dict] = Field(default_factory=list)  # пусто — судья не запускался

    latency_ms: float | None = None
    search_latency_ms: float | None = None
    llm_latency_ms: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    cost_usd: float | None = None
    raw_cost_usd: float | None = None
    # Размер нашего промпта: единственная величина в этой строке, посчитанная нами,
    # а не бэкендом. По ней сравниваются варианты между собой (см. отчёт, раздел стоимости).
    prompt_chars: int | None = None

    corpus_version: str
    prompt_version: str
    prompt_hash: str | None = None
    retrieval_mode: str
    model: str | None = None
    llm_backend: str | None = None
    error: str | None = None  # сбой модели или отклонённый проверкой ответ


def save_rows(rows: Sequence[VariantRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(row.model_dump_json() + "\n")


def load_rows(path: Path) -> list[VariantRow]:
    return [
        VariantRow.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _key(row: VariantRow) -> tuple[str, str, int]:
    return (row.question_id, row.variant, row.repeat)


def is_infrastructure_failure(row: VariantRow) -> bool:
    """Вызов не состоялся (CLI не ответил) — в отличие от ответа, отклонённого проверкой.

    Разница принципиальная. Отклонённый ответ — результат системы: модель ответила, и
    ответ не прошёл гарантии §7. Повторять его до «удачного» значило бы подбирать
    результат. Несостоявшийся вызов результата не несёт, и его повтор честен.
    """
    return row.error is not None and row.error.startswith("LLMError")


def dedupe_rows(rows: Sequence[VariantRow]) -> list[VariantRow]:
    """По одной строке на «вопрос × вариант × повтор».

    Повторный проход дописывается в тот же файл: строку со сбоем инфраструктуры
    вытесняет более поздняя строка. Состоявшийся результат (ответ или отклонённый
    проверкой ответ) не вытесняется ничем — переигрывания здесь нет. Если вызов так
    и не состоялся, остаётся последний сбой, и отчёт показывает его как ошибку.
    """
    best: dict[tuple[str, str, int], VariantRow] = {}
    for row in rows:
        previous = best.get(_key(row))
        if previous is None or is_infrastructure_failure(previous):
            best[_key(row)] = row
    return list(best.values())


def failed_cells(rows: Sequence[VariantRow]) -> set[tuple[str, str, int]]:
    """Ячейки, где вызов модели так и не состоялся: только их повторяет второй проход."""
    return {_key(row) for row in dedupe_rows(rows) if is_infrastructure_failure(row)}


def _append_row(row: VariantRow, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(row.model_dump_json() + "\n")


# --- проверки, не требующие судьи ---------------------------------------------------------


def _missing_required(answer: str, required: Iterable[str]) -> list[str]:
    """Какие обязательные формулировки не найдены в ответе (регистр и ё/е не важны)."""
    text = answering.normalize_for_match(answer)
    return [item for item in required if answering.normalize_for_match(item) not in text]


def _present_forbidden(answer: str, forbidden: Iterable[str]) -> list[str]:
    text = answering.normalize_for_match(answer)
    return [item for item in forbidden if answering.normalize_for_match(item) in text]


# --- прогон ------------------------------------------------------------------------------


def run_comparison(
    conn: psycopg.Connection,
    questions: Sequence[GoldenQuestion],
    *,
    variants: Sequence[Variant],
    profile: Profile,
    k: int,
    repeats: int,
    settings: Settings,
    run_path: Path,
    judge_config: JudgeConfig | None = None,
    progress: Callable[[str], None] | None = None,
    only_cells: set[tuple[str, str, int]] | None = None,
    cooldown_s: float = LLM_FAILURE_COOLDOWN_S,
) -> list[VariantRow]:
    """Прогнать выбранные варианты по вопросам, записывая строки по мере готовности.

    `only_cells` ограничивает прогон конкретными «вопрос × вариант × повтор» — это
    повторный проход по сорвавшимся вызовам, а не новый эксперимент.
    """
    sources = {source.id: source for source in db.list_sources(conn, profile)}
    corpus = corpus_hits(conn, profile) if "full_corpus" in variants else None
    # Один провайдер на весь прогон: вариант «весь корпус» упирается в таймаут CLI,
    # остальным длинный таймаут не мешает.
    provider = (
        CliLLMProvider(timeout_s=LONG_CALL_TIMEOUT_S) if settings.llm_backend == "cli" else None
    )

    rows: list[VariantRow] = []
    for repeat in range(1, repeats + 1):
        for question in questions:
            for variant in variants:
                if only_cells is not None and (question.id, variant, repeat) not in only_cells:
                    continue
                row = _run_one(
                    conn,
                    question,
                    variant,
                    repeat=repeat,
                    sources=sources,
                    profile=profile,
                    k=k,
                    settings=settings,
                    provider=provider,
                    corpus=corpus,
                    judge_config=judge_config,
                )
                rows.append(row)
                _append_row(row, run_path)
                if progress is not None:
                    state = row.error or ("отказ" if row.refused else "ответ")
                    progress(
                        f"{question.id} {variant} #{repeat}: {state} "
                        f"({(row.latency_ms or 0) / 1000:.1f} с)"
                    )
                if is_infrastructure_failure(row) and cooldown_s > 0:
                    if progress is not None:
                        progress(f"CLI не отвечает — пауза {cooldown_s:.0f} с")
                    time.sleep(cooldown_s)
    return rows


def _run_one(
    conn: psycopg.Connection,
    question: GoldenQuestion,
    variant: Variant,
    *,
    repeat: int,
    sources,
    profile: Profile,
    k: int,
    settings: Settings,
    provider,
    corpus,
    judge_config: JudgeConfig | None,
) -> VariantRow:
    base = VariantRow(
        question_id=question.id,
        category=question.category,
        question=question.question,
        answerable=question.answerable,
        variant=variant,
        repeat=repeat,
        expected_source_ids=list(question.expected_source_ids),
        recall_applicable=variant in SEARCH_VARIANTS,
        corpus_version=settings.corpus_version,
        prompt_version=settings.prompt_version,
        retrieval_mode=retrieval_mode_of(variant, settings),
        model=settings.gen_model,
        llm_backend=settings.llm_backend,
    )
    # Выдача фиксируется до вызова модели: полнота поиска — свойство поиска, и сбой
    # генерации не должен выбрасывать вопрос из знаменателя recall.
    hits, search_latency_ms = retrieve(
        conn, variant, question.question, profile=profile, k=k, settings=settings, corpus=corpus
    )
    base = base.model_copy(
        update={
            "retrieved_source_ids": list(dict.fromkeys(hit.source.id for hit in hits)),
            "fragments": len(hits),
            "fragment_chunk_ids": [hit.chunk.id for hit in hits],
            "search_latency_ms": search_latency_ms,
        }
    )
    try:
        result = answer_variant(
            conn,
            variant,
            question.question,
            sources=sources,
            profile=profile,
            k=k,
            settings=settings,
            provider=provider,
            corpus=corpus,
            retrieved=(hits, search_latency_ms),
        )
    except (LLMError, AnswerValidationError) as exc:
        meta = getattr(exc, "meta", {}) or {}
        return base.model_copy(
            update={
                "error": f"{type(exc).__name__}: {exc}",
                "latency_ms": meta.get("latency_ms"),
                "input_tokens": meta.get("input_tokens"),
                "output_tokens": meta.get("output_tokens"),
                "cost_usd": meta.get("cost_usd"),
                "raw_cost_usd": meta.get("raw_cost_usd"),
                "prompt_chars": meta.get("prompt_chars"),
                "prompt_hash": meta.get("prompt_hash"),
            }
        )

    payload, meta = result.payload, result.payload.meta
    cited_sources = [c.source_id for c in payload.citations]
    orgs = {
        sources[sid].speaker_org
        for sid in cited_sources
        if sid in sources and sources[sid].speaker_org
    }
    claim_verdicts = (
        [v.model_dump() for v in judge_answer(payload, result.hits, config=judge_config)]
        if judge_config is not None and payload.citations
        else []
    )
    return base.model_copy(
        update={
            "retrieved_source_ids": result.retrieved_source_ids,
            "fragments": len(result.hits),
            "fragment_chunk_ids": [hit.chunk.id for hit in result.hits],
            "answer_markdown": payload.answer_markdown,
            "refused": payload.refused,
            "citation_source_ids": list(dict.fromkeys(cited_sources)),
            "limitation_kinds": [limitation.kind for limitation in payload.limitations],
            "distinct_orgs": len(orgs),
            "must_contain_missing": _missing_required(
                payload.answer_markdown, question.must_contain
            ),
            "must_not_contain_present": _present_forbidden(
                payload.answer_markdown, question.must_not_contain
            ),
            "claim_verdicts": claim_verdicts,
            "latency_ms": meta.get("latency_ms"),
            "search_latency_ms": meta.get("search_latency_ms"),
            "llm_latency_ms": meta.get("llm_latency_ms"),
            "input_tokens": meta.get("input_tokens"),
            "output_tokens": meta.get("output_tokens"),
            "cache_read_tokens": meta.get("cache_read_tokens"),
            "cache_write_tokens": meta.get("cache_write_tokens"),
            "cost_usd": meta.get("cost_usd"),
            "raw_cost_usd": meta.get("raw_cost_usd"),
            "prompt_chars": meta.get("prompt_chars"),
            "prompt_hash": meta.get("prompt_hash"),
            "model": meta.get("model") or settings.gen_model,
            "llm_backend": meta.get("llm_backend") or settings.llm_backend,
        }
    )


# --- метрики -------------------------------------------------------------------------------


class VariantMetrics(BaseModel):
    """Метрики одного варианта. None везде означает «не измерено», а не ноль."""

    variant: Variant
    title: str
    n_rows: int
    n_questions: int
    n_errors: int
    n_infra_failures: int = 0  # из них вызов не состоялся; остальное — отклонено проверкой

    recall_at_k: float | None
    mrr: float | None
    recall_n: int  # вопросов с ожидаемыми источниками и записанной выдачей
    recall_note: str  # почему метрики нет, если её нет

    must_contain_rate: float | None
    must_contain_n: int
    must_not_contain_violations: int
    refusal_precision: float | None
    refusal_recall: float | None
    refusal_n: int

    answers_with_citations: float | None
    median_citations: float | None
    single_org_share: float | None
    derived_only_share: float | None
    median_distinct_orgs: float | None

    median_latency_ms: float | None
    median_llm_latency_ms: float | None
    median_input_tokens: float | None
    median_output_tokens: float | None
    median_prompt_chars: float | None  # размер нашего промпта, посчитанный кодом
    total_cost_usd: float | None
    cost_usd_per_question: float | None
    total_raw_cost_usd: float | None
    cache_read_tokens: int
    cache_write_tokens: int

    grounded_rate: float | None  # публикуется только вместе с калибровкой судьи
    grounded_n: int
    per_repeat_must_contain: list[float] = Field(default_factory=list)


def _median(values: Sequence[float | int | None]) -> float | None:
    clean = [float(v) for v in values if v is not None]
    return statistics.median(clean) if clean else None


def _share(rows: Sequence[VariantRow], predicate: Callable[[VariantRow], bool]) -> float | None:
    return (sum(1 for r in rows if predicate(r)) / len(rows)) if rows else None


def _recall_and_mrr(rows: Sequence[VariantRow]) -> tuple[float | None, float | None]:
    """Recall и MRR по источникам. MRR — по позиции первого ожидаемого источника."""
    scored = [r for r in rows if r.expected_source_ids]
    if not scored:
        return None, None
    recalls, ranks = [], []
    for row in scored:
        expected = set(row.expected_source_ids)
        recalls.append(len(expected & set(row.retrieved_source_ids)) / len(expected))
        positions = [
            i for i, sid in enumerate(row.retrieved_source_ids, start=1) if sid in expected
        ]
        ranks.append(1.0 / min(positions) if positions else 0.0)
    return sum(recalls) / len(recalls), sum(ranks) / len(ranks)


def _refusal(rows: Sequence[VariantRow]) -> tuple[float | None, float | None, int]:
    scored = [r for r in rows if r.refused is not None]
    if not scored:
        return None, None, 0
    predicted = [r for r in scored if r.refused]
    should = [r for r in scored if not r.answerable]
    precision = (
        sum(1 for r in predicted if not r.answerable) / len(predicted) if predicted else None
    )
    recall = sum(1 for r in should if r.refused) / len(should) if should else None
    return precision, recall, len(scored)


def _grounded(rows: Sequence[VariantRow]) -> tuple[float | None, int]:
    verdicts = [v for row in rows for v in row.claim_verdicts]
    if not verdicts:
        return None, 0
    supported = sum(1 for v in verdicts if v.get("verdict") == "supported")
    return supported / len(verdicts), len(verdicts)


def compute_variant_metrics(
    rows: Sequence[VariantRow], questions_by_id: dict[str, GoldenQuestion]
) -> VariantMetrics:
    variant = rows[0].variant
    ok = [r for r in rows if r.error is None]
    with_required = [
        r
        for r in ok
        if questions_by_id.get(r.question_id) and questions_by_id[r.question_id].must_contain
    ]
    must_rate = (
        sum(1 for r in with_required if not r.must_contain_missing) / len(with_required)
        if with_required
        else None
    )
    per_repeat = []
    for repeat in sorted({r.repeat for r in with_required}):
        group = [r for r in with_required if r.repeat == repeat]
        per_repeat.append(sum(1 for r in group if not r.must_contain_missing) / len(group))

    applicable = variant in SEARCH_VARIANTS
    # Recall считается и по строкам, где сорвалась генерация, если выдача записана:
    # иначе знаменатель состоял бы из «выживших» вопросов, а они не случайная выборка.
    searched = [r for r in rows if r.error is None or r.fragments > 0]
    recall, mrr = _recall_and_mrr(searched) if applicable else (None, None)
    recall_note = (
        "измерено по source_id в top-k"
        if applicable
        else (
            "неприменимо: поиска нет"
            if variant == "no_search"
            else "неприменимо: в контекст передан весь корпус, полнота тривиально равна 1"
        )
    )
    precision, refusal_recall, refusal_n = _refusal(ok)
    grounded, grounded_n = _grounded(ok)
    costs = [r.cost_usd for r in rows if r.cost_usd is not None]
    raw_costs = [r.raw_cost_usd for r in rows if r.raw_cost_usd is not None]

    return VariantMetrics(
        variant=variant,
        title=VARIANT_TITLES.get(variant, variant),
        n_rows=len(rows),
        n_questions=len({r.question_id for r in rows}),
        n_errors=sum(1 for r in rows if r.error is not None),
        n_infra_failures=sum(1 for r in rows if is_infrastructure_failure(r)),
        recall_at_k=recall,
        mrr=mrr,
        recall_n=sum(1 for r in searched if r.expected_source_ids) if applicable else 0,
        recall_note=recall_note,
        must_contain_rate=must_rate,
        must_contain_n=len(with_required),
        must_not_contain_violations=sum(1 for r in ok if r.must_not_contain_present),
        refusal_precision=precision,
        refusal_recall=refusal_recall,
        refusal_n=refusal_n,
        answers_with_citations=_share(ok, lambda r: bool(r.citation_source_ids)),
        median_citations=_median([len(r.citation_source_ids) for r in ok]),
        single_org_share=_share(ok, lambda r: "single_org" in r.limitation_kinds),
        derived_only_share=_share(ok, lambda r: "derived_only" in r.limitation_kinds),
        median_distinct_orgs=_median([r.distinct_orgs for r in ok]),
        median_latency_ms=_median([r.latency_ms for r in rows]),
        median_llm_latency_ms=_median([r.llm_latency_ms for r in rows]),
        median_input_tokens=_median([r.input_tokens for r in rows]),
        median_output_tokens=_median([r.output_tokens for r in rows]),
        median_prompt_chars=_median([r.prompt_chars for r in rows]),
        total_cost_usd=sum(costs) if costs else None,
        cost_usd_per_question=(sum(costs) / len(costs)) if costs else None,
        total_raw_cost_usd=sum(raw_costs) if raw_costs else None,
        cache_read_tokens=sum(r.cache_read_tokens or 0 for r in rows),
        cache_write_tokens=sum(r.cache_write_tokens or 0 for r in rows),
        grounded_rate=grounded,
        grounded_n=grounded_n,
        per_repeat_must_contain=per_repeat,
    )


class ProbeCall(BaseModel):
    n: int
    prompt_chars: int
    latency_ms: int
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    cost_usd: float
    raw_cost_usd: float | None

    @property
    def total_prompt_tokens(self) -> int:
        """Весь промпт запроса: некэшированный остаток плюс прочитанное и записанное в кэш.

        Смотреть на один `input_tokens` нельзя: при работающем кэше там остаются единицы
        токенов, а не размер запроса.
        """
        return self.input_tokens + self.cache_read_tokens + self.cache_write_tokens


class CacheProbe(BaseModel):
    """Замер вызовов: работает ли кэш и как токены зависят от размера нашего промпта.

    `cache_confirmed` — не флаг запроса, а факт из ответа: в usage повторного вызова
    есть ненулевой `cache_read`. Флага `cache_system` для этого недостаточно —
    cli-бэкенд его игнорирует (CONTRACTS.md §9).

    `tokens_per_char` и `service_tokens` — наклон и свободный член прямой, проведённой
    по точкам разного размера промпта. Наклон переводит наши символы в токены, свободный
    член — то, что приходит в счёт не от нашего текста. Оба измерены, а не предположены.
    """

    model: str
    backend: str
    calls: list[ProbeCall]
    minimal_call: ProbeCall | None  # почти пустой запрос: другая точка на той же прямой
    size_points: list[ProbeCall] = Field(default_factory=list)
    tokens_per_char: float | None = None
    service_tokens: float | None = None
    cache_confirmed: bool
    note: str

    def own_tokens(self, prompt_chars: float | None) -> float | None:
        """Сколько токенов приходится на промпт такого размера без служебной части."""
        if self.tokens_per_char is None or prompt_chars is None:
            return None
        return self.tokens_per_char * prompt_chars


def _fit_line(points: Sequence[tuple[float, float]]) -> tuple[float, float] | None:
    """Наклон и свободный член методом наименьших квадратов; None — точек мало."""
    if len(points) < 2:
        return None
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    mean_x, mean_y = sum(xs) / len(xs), sum(ys) / len(ys)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator == 0:
        return None
    slope = sum((x - mean_x) * (y - mean_y) for x, y in points) / denominator
    return slope, mean_y - slope * mean_x


class Comparison(BaseModel):
    """Всё, что нужно отчёту и экрану «Проверка качества»."""

    generated_at: str
    corpus_version: str
    prompt_version: str
    prompt_hash: str | None
    gen_model: str
    llm_backend: str
    profile: str
    k: int
    repeats: int
    include_holdout: bool
    n_questions: int
    run_path: str
    variants: list[VariantMetrics]
    judge: Calibration | None = None
    probe: CacheProbe | None = None


def compare(
    rows: Sequence[VariantRow],
    questions: Sequence[GoldenQuestion],
    *,
    settings: Settings,
    k: int,
    include_holdout: bool,
    run_path: str,
    profile: str,
    judge: Calibration | None = None,
    probe: CacheProbe | None = None,
    generated_at: str | None = None,
) -> Comparison:
    by_id = {q.id: q for q in questions}
    order = {name: i for i, name in enumerate(VARIANTS)}
    groups = sorted({r.variant for r in rows}, key=lambda v: order.get(v, 99))
    metrics = [compute_variant_metrics([r for r in rows if r.variant == v], by_id) for v in groups]
    hashes = {r.prompt_hash for r in rows if r.prompt_hash}
    return Comparison(
        generated_at=generated_at or utc_stamp(),
        corpus_version=settings.corpus_version,
        prompt_version=settings.prompt_version,
        prompt_hash=min(hashes) if len(hashes) == 1 else None,
        gen_model=settings.gen_model,
        llm_backend=settings.llm_backend,
        profile=profile,
        k=k,
        repeats=max((r.repeat for r in rows), default=1),
        include_holdout=include_holdout,
        n_questions=len({r.question_id for r in rows}),
        run_path=run_path,
        variants=metrics,
        judge=judge,
        probe=probe,
    )


# --- холодный/тёплый вызов и фактический кэш ------------------------------------------------


PROBE_SIZES = (0, 2, 4, 8)  # сколько фрагментов класть в промпт точки замера


def probe_cache(
    conn: psycopg.Connection,
    question: str,
    *,
    model: str,
    repeats: int = 3,
    profile: Profile = "public",
    k: int = 8,
    provider=None,
    settings: Settings | None = None,
) -> CacheProbe:
    """Замерить кэш и зависимость токенов от размера промпта на настоящем промпте.

    Три части: один и тот же запрос несколько раз подряд (холодный и тёплые), промпты
    разного размера (по ним считается, сколько токенов даёт наш символ и сколько
    приходит помимо нашего текста) и почти пустой запрос.
    """
    from ergovault.llm import get_provider
    from ergovault.search import search

    settings = settings or get_settings()
    provider = provider or get_provider()
    system, user_template, _ = answering.load_prompt(settings.prompt_version)
    hits = search(conn, question, k, profile=profile, settings=settings)

    def render(count: int) -> str:
        chosen = hits[:count]
        return user_template.substitute(
            question=question, count=len(chosen), fragments=answering.render_fragments(chosen)
        )

    backend = settings.llm_backend

    def call(n: int, sys_text: str, user_text: str) -> ProbeCall:
        nonlocal backend
        response = provider.complete(
            system=sys_text, user=user_text, model=model, cache_system=True, max_tokens=200
        )
        backend = response.backend
        return ProbeCall(
            n=n,
            prompt_chars=len(sys_text) + len(user_text),
            latency_ms=response.latency_ms,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cache_read_tokens=response.cache_read_tokens,
            cache_write_tokens=response.cache_write_tokens,
            cost_usd=response.billed_cost_usd,
            raw_cost_usd=response.raw_cost_usd,
        )

    full_user = render(min(k, len(hits)))
    calls = [call(n, system, full_user) for n in range(1, repeats + 1)]
    size_points = [call(-count, system, render(count)) for count in PROBE_SIZES if count != k]
    size_points.append(calls[0])
    minimal = call(0, "Отвечай одним словом.", "Скажи слово: тест")

    fit = _fit_line([(float(c.prompt_chars), float(c.total_prompt_tokens)) for c in size_points])
    confirmed = any(c.cache_read_tokens > 0 for c in calls[1:])
    note = (
        "Кэш подтверждён: в повторном вызове ненулевой cache_read."
        if confirmed
        else (
            "Кэш НЕ подтверждён: во всех повторных вызовах cache_read = 0. "
            "Стоимость повторных вызовов считается без скидки на кэш."
        )
    )
    return CacheProbe(
        model=model,
        backend=backend,
        calls=calls,
        minimal_call=minimal,
        size_points=sorted(size_points, key=lambda c: c.prompt_chars),
        tokens_per_char=fit[0] if fit else None,
        service_tokens=fit[1] if fit else None,
        cache_confirmed=confirmed,
        note=note,
    )


# --- отчёт ------------------------------------------------------------------------------------


def _pct(value: float | None) -> str:
    return "не измерено" if value is None else f"{value * 100:.0f}%"


def _num(value: float | None, digits: int = 2) -> str:
    return "не измерено" if value is None else f"{value:.{digits}f}"


def _ms(value: float | None) -> str:
    return "не измерено" if value is None else f"{value / 1000:.1f} с"


def _usd(value: float | None) -> str:
    return "не измерено" if value is None else f"${value:.4f}"


def _na(value: float | None, applicable: bool) -> str:
    return "неприменимо" if not applicable else _num(value)


def render_report(comparison: Comparison, examples: str = "") -> str:
    c = comparison
    calibrated = c.judge is not None and c.judge.labeled_enough
    lines = [
        "# Сравнение четырёх подходов к ответу по корпусу",
        "",
        f"Прогон {c.generated_at}. Сырые строки: `{c.run_path}`.",
        "Настройки, метрики и правила чтения результата зафиксированы до открытия holdout —",
        "`eval/preregistration.md`. Этот отчёт собран из строк прогона кодом, а не написан руками.",
        "",
        *([f"> **{incomplete}**", ""] if (incomplete := incompleteness_of(c)) else []),
        "## Что и на чём измерялось",
        "",
        f"- вопросов: {c.n_questions}, повторов каждого: {c.repeats}",
        f"- holdout: {'включён' if c.include_holdout else 'выключен'}",
        (
            f"- corpus_version: `{c.corpus_version}`, prompt_version: `{c.prompt_version}`, "
            f"prompt_hash: `{c.prompt_hash or 'разный в строках прогона'}`"
        ),
        f"- модель ответа: `{c.gen_model}`, доступ: `{c.llm_backend}`, профиль: `{c.profile}`, k={c.k}",
        "",
        "Варианты отличаются только тем, что попадает в промпт; ищет один и тот же `search.py`,",
        "ответ собирает и проверяет один и тот же `answering.py`.",
        "",
        "| Вариант | Что получает модель |",
        "|---|---|",
    ]
    for m in c.variants:
        lines.append(f"| `{m.variant}` | {m.title} |")

    lines += [
        "",
        "## Качество",
        "",
        (
            "`must_contain` — доля вопросов, где в ответе нашлись все обязательные формулировки "
            "ручной разметки; знаменатель — только вопросы, у которых такая разметка есть."
        ),
        "",
        (
            "| Вариант | must_contain | n | нарушений must_not_contain | "
            "отклонено проверкой | вызов не состоялся |"
        ),
        "|---|---|---|---|---|---|",
    ]
    for m in c.variants:
        lines.append(
            f"| `{m.variant}` | {_pct(m.must_contain_rate)} | {m.must_contain_n} | "
            f"{m.must_not_contain_violations} | {m.n_errors - m.n_infra_failures} | "
            f"{m.n_infra_failures} |"
        )

    lines += [
        "",
        "## Полнота поиска",
        "",
        "| Вариант | recall@k | MRR | n | примечание |",
        "|---|---|---|---|---|",
    ]
    for m in c.variants:
        applicable = m.variant in SEARCH_VARIANTS
        lines.append(
            f"| `{m.variant}` | {_na(m.recall_at_k, applicable)} | "
            f"{_na(m.mrr, applicable)} | {m.recall_n if applicable else '—'} | {m.recall_note} |"
        )

    lines += [
        "",
        "## Корректные отказы",
        "",
        (
            "Точность отказа — доля отказов, пришедшихся на вопросы без ответа в корпусе; "
            "полнота — доля таких вопросов, на которых система отказалась."
        ),
        "",
        "| Вариант | точность отказа | полнота отказа | n |",
        "|---|---|---|---|",
    ]
    for m in c.variants:
        lines.append(
            f"| `{m.variant}` | {_num(m.refusal_precision)} | "
            f"{_num(m.refusal_recall)} | {m.refusal_n} |"
        )

    lines += [
        "",
        "## Независимость источников",
        "",
        (
            "Считается кодом по метаданным (CONTRACTS.md §7), а не мнением модели. "
            "Оригинал видео и его пересказ — одно подтверждение."
        ),
        "",
        (
            "| Вариант | ответов с цитатами | медиана цитат | одна организация | только пересказы | "
            "медиана организаций |"
        ),
        "|---|---|---|---|---|---|",
    ]
    for m in c.variants:
        lines.append(
            f"| `{m.variant}` | {_pct(m.answers_with_citations)} | {_num(m.median_citations, 1)} | "
            f"{_pct(m.single_org_share)} | {_pct(m.derived_only_share)} | "
            f"{_num(m.median_distinct_orgs, 1)} |"
        )

    lines += [
        "",
        "## Подтверждённость утверждений",
        "",
    ]
    if calibrated:
        lines += [
            "| Вариант | доля утверждений с опорой | утверждений |",
            "|---|---|---|",
        ]
        for m in c.variants:
            lines.append(f"| `{m.variant}` | {_pct(m.grounded_rate)} | {m.grounded_n} |")
    else:
        reason = c.judge.note if c.judge else "калибровка судьи не проводилась"
        lines += [
            f"**Не измерена.** {reason}",
            "",
            (
                "Судья — измерительный прибор, и без сверки с ручной разметкой его показания "
                "не публикуются. Ноль здесь не подразумевается: неизмеренное не равно нулю."
            ),
        ]

    lines += ["", "## Время и стоимость", ""]
    if c.probe is not None:
        lines += [f"{c.probe.note}", ""]
    lines += [
        (
            "| Вариант | медиана времени | из них модель | символов промпта | наши токены (оценка) "
            "| токенов ответа | расчёт по прайсу за вопрос | значение бэкенда всего |"
        ),
        "|---|---|---|---|---|---|---|---|",
    ]
    probe = c.probe
    for m in c.variants:
        own = probe.own_tokens(m.median_prompt_chars) if probe else None
        lines.append(
            f"| `{m.variant}` | {_ms(m.median_latency_ms)} | {_ms(m.median_llm_latency_ms)} | "
            f"{_num(m.median_prompt_chars, 0)} | {_num(own, 0)} | "
            f"{_num(m.median_output_tokens, 0)} | {_usd(m.cost_usd_per_question)} | "
            f"{_usd(m.total_raw_cost_usd)} |"
        )
    lines += [
        "",
        (
            "«Символов промпта» — единственная колонка, посчитанная нами и проверяемая по коду. "
            "«Наши токены» — она же, переведённая в токены по измеренному ниже курсу; служебная "
            "часть вызова в неё не входит. «Расчёт по прайсу» (`llm/pricing.py`) сделан по usage "
            "бэкенда, то есть вместе со служебной частью, а «значение бэкенда» — его собственное "
            "`total_cost_usd`. Это три разных числа, и ни одно не является фактическим списанием: "
            "генерация идёт по подписке через Claude Code CLI. Токены ответа включают токены "
            "рассуждения модели, отделить которые usage не позволяет."
        ),
        "",
    ]
    if probe is not None:
        lines += [
            "### Холодный и тёплый вызов, размер промпта и служебная часть",
            "",
            "| Вызов | символов промпта | всего токенов запроса | cache_read | cache_write | время |",
            "|---|---|---|---|---|---|",
        ]
        seen: set[int] = set()
        labels = {0: "почти пустой запрос", 1: "холодный (полный промпт)"}
        for call in [*probe.calls, *probe.size_points, *filter(None, [probe.minimal_call])]:
            if id(call) in seen:
                continue
            seen.add(id(call))
            label = labels.get(call.n) or (
                f"тёплый #{call.n - 1}" if call.n > 1 else "промпт меньшего размера"
            )
            lines.append(
                f"| {label} | {call.prompt_chars} | {call.total_prompt_tokens} | "
                f"{call.cache_read_tokens} | {call.cache_write_tokens} | "
                f"{call.latency_ms / 1000:.1f} с |"
            )
        if probe.tokens_per_char is not None:
            lines += [
                "",
                (
                    f"По точкам разного размера: {probe.tokens_per_char:.2f} токена на символ "
                    f"нашего промпта, и ещё около {probe.service_tokens:,.0f} токенов в запросе "
                    "не приходится на наш текст. Это служебная часть headless-режима; она "
                    "измерена, а не предположена, и из «расчёта по прайсу» не исключена. "
                    "Равенство этой части между вариантами тоже не предполагается — у вариантов "
                    "разные промпты, и проверить постоянство слагаемого можно только замером."
                ),
                "",
            ]

    if c.judge is not None:
        lines += ["## Проверка судьи", "", f"{c.judge.note}", ""]
        lines += [
            (
                "| Конфигурация | пар | размечено человеком | повторов | устойчивость | "
                "согласие с человеком | Cohen's kappa |"
            ),
            "|---|---|---|---|---|---|---|",
        ]
        for cfg in c.judge.judge_configs:
            vs = cfg.vs_human
            lines.append(
                f"| `{cfg.config}` | {cfg.n_pairs} | {cfg.n_labeled} | {cfg.repeats} | "
                f"{_pct(cfg.self_consistency)} | "
                f"{_pct(vs.observed_agreement) if vs else 'не измерено'} | "
                f"{_num(vs.cohen_kappa) if vs and vs.cohen_kappa is not None else 'не измерено'} |"
            )
        if c.judge.between_configs is not None:
            between = c.judge.between_configs
            lines += [
                "",
                (
                    f"Согласие двух конфигураций судьи между собой: "
                    f"{_pct(between.observed_agreement)} на {between.n} парах, "
                    f"kappa {_num(between.cohen_kappa) if between.cohen_kappa is not None else 'не определена'}."
                ),
            ]
        lines.append("")

    if examples:
        lines += ["## Разобранные примеры", "", examples, ""]

    lines += [
        "## Ограничения",
        "",
        *(f"- {item}" for item in limitations_of(comparison)),
        "",
    ]
    return "\n".join(lines) + "\n"


# --- пары для калибровки судьи ------------------------------------------------------------


DISTRACTOR_SHARE = 0.4  # доля пар «утверждение и НЕ тот фрагмент»


class JudgePairDraft(BaseModel):
    """Заготовка пары до присвоения id. Поле `kind` — только след происхождения."""

    id: str = ""
    claim: str
    chunk_id: str
    origin: str
    kind: str


def build_judge_pairs(
    rows: Sequence[VariantRow], *, target: int = 48, seed: int = 30
) -> list[JudgePairDraft]:
    """Собрать пары «утверждение — фрагмент» из готового прогона для ручной разметки.

    Пары строятся двух видов: утверждение со своим процитированным фрагментом и то же
    утверждение с чужим фрагментом из той же выдачи. Без второго вида набор состоял бы
    почти из одних `supported`, и согласие судьи с человеком оказалось бы завышено
    просто потому, что угадать преобладающий класс легко.

    Вердиктов здесь не проставляется: разметка — работа человека, и модель не заполняет
    её за него (T-30, шаг 1).
    """
    import random

    from ergovault.eval.judge import split_claims

    rng = random.Random(seed)
    cited_pool: list[JudgePairDraft] = []
    distractor_pool: list[JudgePairDraft] = []
    for row in sorted(rows, key=lambda r: (r.question_id, r.variant)):
        if row.error or not row.fragment_chunk_ids:
            continue
        for claim in split_claims(row.answer_markdown):
            own = [
                row.fragment_chunk_ids[n - 1]
                for n in claim.refs
                if 1 <= n <= len(row.fragment_chunk_ids)
            ]
            if not own:
                continue
            origin = f"{row.question_id}/{row.variant}"
            cited_pool.append(
                JudgePairDraft(claim=claim.text, chunk_id=own[0], origin=origin, kind="cited")
            )
            others = [c for c in row.fragment_chunk_ids if c not in own]
            if others:
                distractor_pool.append(
                    JudgePairDraft(
                        claim=claim.text,
                        chunk_id=rng.choice(others),
                        origin=origin,
                        kind="distractor",
                    )
                )

    want_distractor = round(target * DISTRACTOR_SHARE)
    rng.shuffle(cited_pool)
    rng.shuffle(distractor_pool)
    chosen = distractor_pool[:want_distractor] + cited_pool[: target - want_distractor]
    chosen.sort(key=lambda p: (p.origin, p.kind))
    for i, pair in enumerate(chosen, start=1):
        pair.id = f"JP-{i:02d}"
    return chosen


def refresh_judge_pairs(
    rows: Sequence[VariantRow], *, target: int = 48, path: Path | None = None
) -> tuple[int, int]:
    """Пересобрать файл пар из прогона, сохранив уже проставленную человеком разметку.

    Возвращает (всего пар, из них размеченных). Разметка переносится по совпадению
    утверждения и чанка: пересборка набора не должна стирать работу человека.
    """
    from ergovault.eval.judge import PAIRS_PATH, JudgePair, load_judge_pairs, save_judge_pairs

    path = path or PAIRS_PATH
    known = {}
    if path.is_file():
        known = {(p.claim, p.chunk_id): p for p in load_judge_pairs(path)}

    pairs = []
    for draft in build_judge_pairs(rows, target=target):
        previous = known.get((draft.claim, draft.chunk_id))
        pairs.append(
            JudgePair(
                id=draft.id,
                claim=draft.claim,
                chunk_id=draft.chunk_id,
                origin=draft.origin,
                human_verdict=previous.human_verdict if previous else None,
                note=previous.note if previous else f"источник пары: {draft.kind}",
            )
        )
    save_judge_pairs(pairs, path)
    return len(pairs), sum(1 for p in pairs if p.human_verdict is not None)


MAX_EXAMPLE_CHARS = 700


def _example_block(row: VariantRow, verdict: str) -> str:
    body = row.answer_markdown.strip()
    if len(body) > MAX_EXAMPLE_CHARS:
        body = body[:MAX_EXAMPLE_CHARS].rstrip() + " …"
    head = f"**{row.question_id} ({row.variant}) — {verdict}.** {row.question}"
    tail = []
    if row.must_contain_missing:
        tail.append(f"не хватило: {', '.join(row.must_contain_missing)}")
    if row.must_not_contain_present:
        tail.append(f"появилось запрещённое: {', '.join(row.must_not_contain_present)}")
    if row.error:
        tail.append(f"ошибка: {row.error}")
    note = f"\n\n_{'; '.join(tail)}_" if tail else ""
    return f"{head}\n\n> {body}{note}"


def pick_examples(rows: Sequence[VariantRow], *, limit: int = 4) -> str:
    """Удачные и неудачные случаи наравне — по фактам строк, а не по впечатлению.

    Удачный: вариант с поиском выполнил разметку там, где вариант без поиска её провалил.
    Неудачный: разметка провалена, появилось запрещённое или прогон завершился ошибкой.
    """
    by_key = {(r.question_id, r.variant): r for r in rows if r.repeat == 1}
    good, bad = [], []
    for (qid, variant), row in by_key.items():
        if variant == "no_search":
            continue
        baseline = by_key.get((qid, "no_search"))
        if (
            row.error is None
            and not row.must_contain_missing
            and baseline is not None
            and (baseline.must_contain_missing or baseline.error)
        ):
            good.append(row)
        if row.error or row.must_contain_missing or row.must_not_contain_present:
            bad.append(row)

    parts = []
    if good:
        parts.append("### Где корпус помог\n")
        parts += [_example_block(r, "разметка выполнена") for r in good[:limit]]
    if bad:
        parts.append("### Где не сработало\n")
        parts += [_example_block(r, "разметка не выполнена") for r in bad[:limit]]
    return "\n\n".join(parts)


def incompleteness_of(c: Comparison) -> str | None:
    """Формулировка о несостоявшихся вызовах, если они есть; None — прогон полный.

    Ячейки без результата выпадают из знаменателей неслучайно: сорвавшиеся вызовы идут
    подряд, волной, и выжившие вопросы — не случайная выборка набора. Такие числа
    результатом не являются, и об этом говорится первым делом, а не в сноске.
    """
    failed = sum(m.n_infra_failures for m in c.variants)
    total = sum(m.n_rows for m in c.variants)
    if not failed:
        return None
    return (
        f"Прогон неполный: {failed} из {total} вызовов модели не состоялись (CLI не ответил). "
        "Метрики посчитаны по выжившим вопросам, а они не случайная выборка — "
        "это промежуточные числа, а не результат сравнения."
    )


def limitations_of(c: Comparison) -> list[str]:
    """Оговорки отчёта. Собираются из фактов прогона, а не пишутся один раз вручную."""
    items = [SAMPLE_SIZE_CAVEAT]
    if incomplete := incompleteness_of(c):
        items.insert(0, incomplete)
    spread = [m for m in c.variants if len(m.per_repeat_must_contain) > 1]
    if spread:
        worst = max(
            spread, key=lambda m: max(m.per_repeat_must_contain) - min(m.per_repeat_must_contain)
        )
        delta = max(worst.per_repeat_must_contain) - min(worst.per_repeat_must_contain)
        items.append(
            f"Разброс между повторами измерен: у варианта `{worst.variant}` доля must_contain "
            f"колеблется на {delta * 100:.0f} п.п. между прогонами. Разница между вариантами "
            "меньше этого разброса результатом не считается."
        )
    else:
        items.append(
            "Повторов больше одного не делалось: разброс между прогонами не измерен, "
            "и различия между вариантами нельзя отделить от случайности одного прогона."
        )
    if c.judge is None or not c.judge.labeled_enough:
        items.append(
            "Подтверждённость утверждений не измерена: судья не откалиброван по ручной "
            "разметке. Это не ноль ошибок, а отсутствие измерения."
        )
    if c.probe is None:
        items.append(
            "Замер кэша и служебной части вызова не выполнен: стоимость в отчёте — только "
            "расчёт по usage бэкенда, без разделения на наш промпт и служебные токены."
        )
    if c.probe is not None and not c.probe.cache_confirmed:
        items.append(
            "Prompt caching не подтверждён: cache_read во всех повторных вызовах равен нулю. "
            "Сравнение стоимости между вариантами сделано без скидки на кэш; вариант с большим "
            "контекстом в конфигурации с работающим кэшем стоил бы меньше показанного."
        )
    items.append(
        "Стоимость расчётная. Генерация идёт по подписке через Claude Code CLI: ни расчёт по "
        "прайсу, ни значение бэкенда не равны фактическому списанию, а служебная часть вызова "
        "входит в `input_tokens` и отделена только нижней оценкой."
    )
    items.append(
        "Для cli-бэкенда не заявляется управление temperature: соответствующего флага у "
        "headless-режима нет, детерминизм ответов не предполагается."
    )
    best = [m for m in c.variants if m.must_contain_rate is not None]
    if len(best) >= 2:
        top, second = sorted(best, key=lambda m: -(m.must_contain_rate or 0))[:2]
        gap = (top.must_contain_rate or 0) - (second.must_contain_rate or 0)
        if gap < 0.15:
            items.append(
                f"Лучший вариант (`{top.variant}`) опережает следующий (`{second.variant}`) "
                f"на {gap * 100:.0f} п.п. — меньше порога различимости при таком n. "
                "Выигрыш RAG на этом наборе не доказан."
            )
    return items


# --- summary.json для /api/eval/summary --------------------------------------------------------


def build_summary(c: Comparison, *, message: str = "") -> dict:
    """Данные экрана «Проверка качества» (CONTRACTS.md §8)."""
    calibrated = c.judge is not None and c.judge.labeled_enough
    runs = []
    for m in c.variants:
        runs.append(
            {
                "label": m.variant,
                "title": m.title,
                "mode": m.variant,
                "k": c.k,
                "n_questions": m.n_questions,
                "recall_at_k": m.recall_at_k,
                "mrr": m.mrr,
                "recall_note": m.recall_note,
                "refusal_precision": m.refusal_precision,
                "refusal_recall": m.refusal_recall,
                "must_contain_rate": m.must_contain_rate,
                "grounded_rate": m.grounded_rate if calibrated else None,
                "median_latency_ms": m.median_latency_ms,
                "cost_usd_per_question": m.cost_usd_per_question,
            }
        )

    judge_block = None
    if c.judge is not None:
        primary = c.judge.judge_configs[0] if c.judge.judge_configs else None
        vs = primary.vs_human if primary else None
        judge_block = {
            "available": c.judge.labeled_enough,
            "message": c.judge.note,
            "n_labeled": primary.n_labeled if primary else 0,
            "agreement": vs.observed_agreement if vs else None,
            "cohen_kappa": vs.cohen_kappa if vs else None,
            "self_consistency": primary.self_consistency if primary else None,
        }

    default_message = f"Сравнение четырёх подходов на {c.n_questions} вопросах проверочного набора."
    if incomplete := incompleteness_of(c):
        default_message = incomplete
    return {
        "message": message or default_message,
        "generated_at": c.generated_at,
        "corpus_version": c.corpus_version,
        "prompt_version": c.prompt_version,
        "sample_size_caveat": SAMPLE_SIZE_CAVEAT,
        "runs": runs,
        "judge": judge_block,
        "limitations": limitations_of(c),
        "report_path": str(FINAL_REPORT_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
    }


def write_summary(summary: dict, path: Path = SUMMARY_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )


# --- сборка результата --------------------------------------------------------------------------


def publish(
    rows: Sequence[VariantRow],
    questions: Sequence[GoldenQuestion],
    *,
    settings: Settings,
    k: int,
    include_holdout: bool,
    profile: str,
    run_path: Path,
    probe: CacheProbe | None = None,
    generated_at: str | None = None,
) -> Comparison:
    """Из сырых строк — отчёт и `summary.json`. Модель здесь не вызывается."""
    from ergovault.eval.judge import load_calibration

    rows = dedupe_rows(rows)
    probe = probe or load_probe()
    comparison = compare(
        rows,
        questions,
        settings=settings,
        k=k,
        include_holdout=include_holdout,
        profile=profile,
        run_path=str(run_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        judge=load_calibration(),
        probe=probe,
        generated_at=generated_at,
    )
    FINAL_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    FINAL_REPORT_PATH.write_text(
        render_report(comparison, examples=pick_examples(rows)), encoding="utf-8"
    )
    write_summary(build_summary(comparison))
    return comparison


def render_stdout(c: Comparison) -> str:
    lines = [
        (
            f"вопросов={c.n_questions} повторов={c.repeats} k={c.k} "
            f"holdout={'включён' if c.include_holdout else 'выключен'} модель={c.gen_model}"
        ),
        "",
        (
            f"{'вариант':<14}{'must_contain':>14}{'recall@k':>12}{'отказ P/R':>14}"
            f"{'время':>10}{'$/вопрос':>12}{'ошибок':>9}"
        ),
    ]
    # В терминале «не измерено» разъезжается по колонкам: там «н/д», а расшифровка —
    # в отчёте, где для неё есть место.
    def short(value: float | None, kind: str = "num") -> str:
        if value is None:
            return "н/д"
        return {"pct": _pct, "num": _num, "ms": _ms, "usd": _usd}[kind](value)

    for m in c.variants:
        refusal = f"{short(m.refusal_precision)}/{short(m.refusal_recall)}"
        lines.append(
            f"{m.variant:<14}{short(m.must_contain_rate, 'pct'):>14}"
            f"{(short(m.recall_at_k) if m.variant in SEARCH_VARIANTS else 'н/п'):>12}"
            f"{refusal:>14}{short(m.median_latency_ms, 'ms'):>10}"
            f"{short(m.cost_usd_per_question, 'usd'):>12}{m.n_errors:>9}"
        )
    lines += ["", "ограничения:"]
    lines += [f"  - {item}" for item in limitations_of(c)]
    return "\n".join(lines)


def run_experiment(
    *,
    variants: Sequence[Variant] = VARIANTS,
    k: int | None = None,
    repeats: int = 1,
    include_holdout: bool = False,
    profile: Profile = "public",
    only: Sequence[str] = (),
    limit: int | None = None,
    label: str = "compare",
    with_probe: bool = True,
    with_judge: bool = False,
    retry: Path | None = None,
    settings: Settings | None = None,
    echo: Callable[[str], None] = print,
) -> Comparison:
    """Прогнать сравнение целиком и записать отчёт, summary.json и сырые строки.

    `retry` — путь к прогону, в котором часть вызовов сорвалась: повторяются только эти
    ячейки, результат дописывается в тот же файл, отчёт считается по всему файлу.
    Это доделка прогона, а не новый эксперимент: настройки берутся те же.
    """
    from ergovault.eval.judge import JudgeConfig

    settings = settings or get_settings()
    k = k or settings.top_k
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        raise ValueError(f"неизвестные варианты: {', '.join(unknown)}; допустимы {VARIANTS}")
    questions = select_questions(include_holdout=include_holdout, only=only, limit=limit)
    if not questions:
        raise ValueError("после отбора не осталось ни одного вопроса")

    stamp = utc_stamp()
    previous: list[VariantRow] = []
    only_cells: set[tuple[str, str, int]] | None = None
    if retry is not None:
        retry = retry.resolve()  # путь из командной строки бывает относительным
        previous = load_rows(retry)
        only_cells = failed_cells(previous)
        run_path = retry
        if not only_cells:
            echo("сорвавшихся вызовов нет — пересобираю отчёт по имеющимся строкам")
        else:
            echo(f"повтор {len(only_cells)} сорвавшихся вызовов; строки дописываются в {retry}")
    else:
        run_path = RUNS_DIR / f"{stamp}-{label}.jsonl"
        echo(
            f"вопросов {len(questions)} × вариантов {len(variants)} × повторов {repeats} = "
            f"{len(questions) * len(variants) * repeats} вызовов модели; строки: "
            f"{run_path.relative_to(REPO_ROOT)}"
        )

    judge_config = JudgeConfig(model=settings.judge_model) if with_judge else None
    probe = None
    with psycopg.connect(settings.database_url) as conn:
        rows = previous + run_comparison(
            conn,
            questions,
            variants=variants,
            profile=profile,
            k=k,
            repeats=repeats,
            settings=settings,
            run_path=run_path,
            judge_config=judge_config,
            progress=echo,
            only_cells=only_cells,
        )
        if with_probe:
            echo("замер холодного и тёплого вызова…")
            try:
                probe = probe_cache(
                    conn,
                    questions[0].question,
                    model=settings.gen_model,
                    profile=profile,
                    k=k,
                    settings=settings,
                )
            except LLMError as exc:
                # Сорвавшийся замер не отменяет прогон: отчёт соберётся с прежним замером
                # (если он есть) или скажет, что замера нет.
                echo(f"замер не выполнен: {exc}")
            else:
                save_probe(probe)
                echo(probe.note)

    comparison = publish(
        rows,
        questions,
        settings=settings,
        k=k,
        include_holdout=include_holdout,
        profile=profile,
        run_path=run_path,
        probe=probe,
        generated_at=stamp,
    )
    echo("")
    echo(render_stdout(comparison))
    echo("")
    echo(f"отчёт: {FINAL_REPORT_PATH.relative_to(REPO_ROOT)}")
    echo(f"экран: {SUMMARY_PATH.relative_to(REPO_ROOT)}")
    return comparison


def rebuild(
    run_path: Path, *, include_holdout: bool = False, settings: Settings | None = None
) -> Comparison:
    """Пересчитать отчёт и summary.json из готового прогона, не вызывая модель."""
    settings = settings or get_settings()
    run_path = run_path.resolve()  # путь из командной строки бывает относительным
    rows = load_rows(run_path)
    if not rows:
        raise ValueError(f"в {run_path} нет строк прогона")
    questions = select_questions(include_holdout=include_holdout)
    known = {q.id for q in questions}
    if missing := {r.question_id for r in rows} - known:
        # Строки holdout в прогоне, а набор открыт без флага: пересчитать нечем.
        raise ValueError(
            "в прогоне есть вопросы вне текущего набора "
            f"({', '.join(sorted(missing))}) — нужен --include-holdout"
        )
    # k восстанавливается по строке варианта с поиском: у `full_corpus` в контексте весь
    # корпус, у `no_search` фрагментов нет — их число к k отношения не имеет.
    searched = [r.fragments for r in rows if r.variant in SEARCH_VARIANTS and r.fragments]
    return publish(
        rows,
        questions,
        settings=settings,
        k=max(searched) if searched else settings.top_k,
        include_holdout=include_holdout,
        profile=settings.profile,
        run_path=run_path,
    )


def select_questions(
    *, include_holdout: bool, only: Sequence[str] = (), limit: int | None = None
) -> list[GoldenQuestion]:
    questions = load_golden_set(include_holdout=include_holdout)
    if only:
        wanted = {q.strip() for q in only if q.strip()}
        questions = [q for q in questions if q.id in wanted]
        if missing := wanted - {q.id for q in questions}:
            raise ValueError(f"нет таких вопросов в наборе: {', '.join(sorted(missing))}")
    return questions[:limit] if limit else questions
