"""Метрики поиска и отказа без LLM-судьи. Recall/MRR считаются по source_id в top-k чанках."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from ergovault.eval.runner import RunRow

# Спека (CONTRACTS.md, eval/golden/README.md): при n≈35 разница между вариантами меньше
# ~15 п.п. неотличима от шума. Оговорка обязательна в каждом отчёте вне зависимости от
# фактического n конкретного прогона.
SAMPLE_SIZE_CAVEAT = (
    "При n≈35 разница между вариантами меньше ~15 п.п. неотличима от шума "
    "(golden set рассчитан на порядок величины, не на статистическую значимость)."
)


def _expected_rows(rows: list[RunRow]) -> list[RunRow]:
    """Вопросы, для которых recall/MRR вообще определены — с непустым expected_source_ids.

    Вопросы с answerable=False намеренно имеют expected_source_ids=[] (спека holdout/golden
    README) — считать для них recall бессмысленно, они не входят в знаменатель."""
    return [r for r in rows if r.expected_source_ids]


def _row_recall(row: RunRow) -> float:
    expected = set(row.expected_source_ids)
    found = expected & set(row.retrieved_source_ids)
    return len(found) / len(expected)


def _row_reciprocal_rank(row: RunRow) -> float:
    expected = set(row.expected_source_ids)
    ranks = [item.rank for item in row.retrieved if item.source_id in expected]
    return 1.0 / min(ranks) if ranks else 0.0


@dataclass
class CategoryMetrics:
    category: str
    n: int
    recall_at_k: float | None
    mrr: float | None


@dataclass
class RefusalMetrics:
    n: int
    precision: float | None  # None — знаменатель 0 (генератор ни разу не отказал)
    recall: float | None  # None — знаменатель 0 (среди вопросов набора нет answerable=False)


@dataclass
class MetricsReport:
    k: int
    n_total: int
    n_with_expected: int
    recall_at_k: float | None
    mrr: float | None
    empty_rate_all: float
    empty_rate_answerable: float
    empty_rate_unanswerable: float
    by_category: list[CategoryMetrics] = field(default_factory=list)
    refusal: RefusalMetrics | None = None  # None — генератор не запускался ни разу
    refusal_unavailable_reason: str | None = None
    latency_median_ms: float | None = None  # None — в прогоне нет замеров задержки


def _latency_median(rows: list[RunRow]) -> float | None:
    measured = [r.latency_ms for r in rows if r.latency_ms is not None]
    return statistics.median(measured) if measured else None


def _empty_rate(rows: list[RunRow]) -> float:
    if not rows:
        return 0.0
    return sum(1 for r in rows if r.empty) / len(rows)


def _recall_and_mrr(rows: list[RunRow]) -> tuple[float | None, float | None]:
    expected_rows = _expected_rows(rows)
    if not expected_rows:
        return None, None
    recall = sum(_row_recall(r) for r in expected_rows) / len(expected_rows)
    mrr = sum(_row_reciprocal_rank(r) for r in expected_rows) / len(expected_rows)
    return recall, mrr


def _refusal_metrics(rows: list[RunRow]) -> tuple[RefusalMetrics | None, str | None]:
    scored = [r for r in rows if r.refused is not None]
    if not scored:
        return None, "генератор не запускался — это прогон только ретривера"

    predicted_refuse = [r for r in scored if r.refused]
    should_refuse = [r for r in scored if not r.answerable]

    tp_precision = sum(1 for r in predicted_refuse if not r.answerable)
    precision = tp_precision / len(predicted_refuse) if predicted_refuse else None

    tp_recall = sum(1 for r in should_refuse if r.refused)
    recall = tp_recall / len(should_refuse) if should_refuse else None

    return RefusalMetrics(n=len(scored), precision=precision, recall=recall), None


def _by_category(rows: list[RunRow], k: int) -> list[CategoryMetrics]:
    categories = sorted({r.category for r in rows})
    result = []
    for cat in categories:
        cat_rows = [r for r in rows if r.category == cat]
        recall, mrr = _recall_and_mrr(cat_rows)
        n = len(_expected_rows(cat_rows))
        result.append(CategoryMetrics(category=cat, n=n, recall_at_k=recall, mrr=mrr))
    return result


def compute_metrics(rows: list[RunRow], *, k: int) -> MetricsReport:
    recall, mrr = _recall_and_mrr(rows)
    answerable_rows = [r for r in rows if r.answerable]
    unanswerable_rows = [r for r in rows if not r.answerable]
    refusal, refusal_reason = _refusal_metrics(rows)

    return MetricsReport(
        k=k,
        n_total=len(rows),
        n_with_expected=len(_expected_rows(rows)),
        recall_at_k=recall,
        mrr=mrr,
        empty_rate_all=_empty_rate(rows),
        empty_rate_answerable=_empty_rate(answerable_rows),
        empty_rate_unanswerable=_empty_rate(unanswerable_rows),
        by_category=_by_category(rows, k),
        refusal=refusal,
        refusal_unavailable_reason=refusal_reason,
        latency_median_ms=_latency_median(rows),
    )


def _fmt(value: float | None, *, pct: bool = False) -> str:
    if value is None:
        return "н/д"
    return f"{value * 100:.0f}%" if pct else f"{value:.2f}"


def _fmt_ms(value: float | None) -> str:
    return "н/д" if value is None else f"{value:.0f} мс"


def render_stdout_table(report: MetricsReport, *, mode: str, include_holdout: bool) -> str:
    holdout_state = "включён" if include_holdout else "выключен"
    lines = [
        f"ретривер={mode}  k={report.k}  вопросов={report.n_total}  (holdout={holdout_state})",
        "",
        f"recall@{report.k}        {_fmt(report.recall_at_k)}  (n={report.n_with_expected})",
        f"MRR@{report.k}           {_fmt(report.mrr)}  (n={report.n_with_expected})",
        f"пустая выдача, всего   {_fmt(report.empty_rate_all, pct=True)}",
        f"пустая выдача, answerable=true   {_fmt(report.empty_rate_answerable, pct=True)}",
        f"пустая выдача, answerable=false  {_fmt(report.empty_rate_unanswerable, pct=True)}",
        f"медианная задержка поиска  {_fmt_ms(report.latency_median_ms)}",
    ]
    if report.refusal is not None:
        lines.append(
            f"отказ: precision={_fmt(report.refusal.precision)}  "
            f"recall={_fmt(report.refusal.recall)}  (n={report.refusal.n})"
        )
    else:
        lines.append(f"отказ: н/д — {report.refusal_unavailable_reason}")

    lines.append("")
    lines.append("по категориям:")
    for cat in report.by_category:
        lines.append(
            f"  {cat.category}  recall@{report.k}={_fmt(cat.recall_at_k)}  "
            f"mrr={_fmt(cat.mrr)}  n={cat.n}"
        )

    lines.append("")
    lines.append(f"ВНИМАНИЕ: {SAMPLE_SIZE_CAVEAT}")
    return "\n".join(lines)


def render_report_markdown(
    report: MetricsReport,
    *,
    mode: str,
    include_holdout: bool,
    corpus_version: str,
    prompt_version: str,
    label: str,
    run_path: str,
) -> str:
    lines = [
        f"# Отчёт eval retrieval — {label}",
        "",
        f"- retrieval_mode: `{mode}`",
        f"- corpus_version: `{corpus_version}`",
        f"- prompt_version: `{prompt_version}`",
        f"- k: {report.k}",
        f"- holdout: {'включён (--include-holdout)' if include_holdout else 'выключен'}",
        f"- вопросов в прогоне: {report.n_total}",
        f"- сырые результаты: `{run_path}`",
        "",
        "## Метрики",
        "",
        "| Метрика | Значение | n |",
        "|---|---|---|",
        f"| recall@{report.k} | {_fmt(report.recall_at_k)} | {report.n_with_expected} |",
        f"| MRR@{report.k} | {_fmt(report.mrr)} | {report.n_with_expected} |",
        f"| пустая выдача (все) | {_fmt(report.empty_rate_all, pct=True)} | {report.n_total} |",
        f"| пустая выдача (answerable=true) | {_fmt(report.empty_rate_answerable, pct=True)} | — |",
        f"| пустая выдача (answerable=false) | {_fmt(report.empty_rate_unanswerable, pct=True)} | — |",
        f"| медианная задержка поиска | {_fmt_ms(report.latency_median_ms)} | {report.n_total} |",
    ]
    if report.refusal is not None:
        lines.append(
            f"| отказ: precision | {_fmt(report.refusal.precision)} | {report.refusal.n} |"
        )
        lines.append(f"| отказ: recall | {_fmt(report.refusal.recall)} | {report.refusal.n} |")
    else:
        lines.append(f"| отказ: precision/recall | н/д — {report.refusal_unavailable_reason} | — |")

    lines += ["", "## По категориям", "", "| Категория | recall@k | MRR | n |", "|---|---|---|---|"]
    for cat in report.by_category:
        lines.append(f"| {cat.category} | {_fmt(cat.recall_at_k)} | {_fmt(cat.mrr)} | {cat.n} |")

    lines += ["", "## Оговорка о размере выборки", "", SAMPLE_SIZE_CAVEAT]
    return "\n".join(lines) + "\n"
