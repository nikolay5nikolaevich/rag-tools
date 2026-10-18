"""Проверки загрузки вопросов, прогона поиска и метрик на данных с известными ответами."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ergovault.eval.dataset import (
    DuplicateGoldenIdError,
    GoldenQuestion,
    load_golden_set,
)
from ergovault.eval.metrics import compute_metrics
from ergovault.eval.runner import RunRow, run
from ergovault.models import Chunk, Hit, Source

GOLDEN_DIR = Path(__file__).parents[2] / "eval" / "golden"

# --- dataset.py: holdout — самообман, которого не должно случиться -------------------


def test_holdout_excluded_by_default() -> None:
    questions = load_golden_set(GOLDEN_DIR)
    assert len(questions) == 23  # eval/golden/README.md, таблица состава: 33 - 10 holdout
    assert all(not q.holdout for q in questions)


def test_include_holdout_flag_adds_holdout_questions() -> None:
    questions = load_golden_set(GOLDEN_DIR, include_holdout=True)
    assert len(questions) == 33
    assert sum(1 for q in questions if q.holdout) == 10


def test_default_call_matches_real_golden_dir() -> None:
    # DEFAULT_GOLDEN_DIR указывает на репозиторный eval/golden без явного параметра.
    assert len(load_golden_set()) == 23


def test_category_counts_match_readme() -> None:
    questions = load_golden_set(GOLDEN_DIR, include_holdout=True)
    counts = {cat: sum(1 for q in questions if q.category == cat) for cat in "ABCDEF"}
    assert counts == {"A": 10, "B": 6, "C": 5, "D": 4, "E": 6, "F": 2}


def test_holdout_file_not_read_without_flag(tmp_path, monkeypatch) -> None:
    # Файл holdout.yml с заведомо битым YAML: если бы loader его открывал без флага,
    # прогон упал бы ещё до фильтрации по полю `holdout`.
    golden_dir = tmp_path
    (golden_dir / "A-facts.yml").write_text(
        "- id: A-01\n  category: A\n  question: q\n  answerable: true\n", encoding="utf-8"
    )
    (golden_dir / "holdout.yml").write_text("{ этот yaml специально сломан", encoding="utf-8")

    questions = load_golden_set(golden_dir)  # не должно упасть
    assert [q.id for q in questions] == ["A-01"]

    with pytest.raises(yaml.YAMLError):
        load_golden_set(golden_dir, include_holdout=True)


def test_duplicate_id_raises(tmp_path) -> None:
    golden_dir = tmp_path
    (golden_dir / "A.yml").write_text(
        "- id: DUP\n  category: A\n  question: q1\n  answerable: true\n"
        "- id: DUP\n  category: A\n  question: q2\n  answerable: true\n",
        encoding="utf-8",
    )
    with pytest.raises(DuplicateGoldenIdError):
        load_golden_set(golden_dir)


# --- runner.py: прогон через фейковый ретривер ----------------------------------------


def _chunk(chunk_id: str, source_id: str, text: str = "текст чанка") -> Chunk:
    return Chunk(
        id=chunk_id, source_id=source_id, ordinal=0, heading_path=["_intro"],
        text=text, char_len=len(text), token_estimate=len(text) // 3,
        atomic=False, quote_anchor=text[:120],
    )


def _source(source_id: str) -> Source:
    return Source(id=source_id, path=f"{source_id}.md", type="article", title=source_id, content_hash="h")


def _hit(source_id: str, rank: int, score: float = 1.0) -> Hit:
    chunk_id = f"{source_id}#000"
    return Hit(chunk=_chunk(chunk_id, source_id), source=_source(source_id), score=score, rank=rank)


class _FakeRetriever:
    name = "fake"

    def __init__(self, hits_by_query: dict[str, list[Hit]]):
        self._hits_by_query = hits_by_query

    def search(self, query: str, k: int = 10, *, profile: str = "public") -> list[Hit]:
        return self._hits_by_query.get(query, [])[:k]


def test_runner_produces_rows_with_versions_and_dedup() -> None:
    q = GoldenQuestion(
        id="Q1", category="A", question="вопрос про s1 и s3",
        answerable=True, expected_source_ids=["s1"],
    )
    hits = [_hit("s3", 1), _hit("s1", 2), _hit("s3", 3)]  # s3 дважды разными чанками
    retriever = _FakeRetriever({"вопрос про s1 и s3": hits})

    class _S:
        corpus_version = "v1"
        prompt_version = "p1"
        gen_model = "claude-opus-5"

    rows = run([q], retriever.search, mode=retriever.name, k=5, settings=_S())
    assert len(rows) == 1
    row = rows[0]
    assert row.corpus_version == "v1"
    assert row.prompt_version == "p1"
    assert row.retrieval_mode == "fake"
    assert row.model is None  # генератор не передан
    assert row.refused is None
    assert row.retrieved_source_ids == ["s3", "s1"]  # без повторов, порядок первого появления
    assert row.empty is False


def test_runner_marks_empty_result() -> None:
    q = GoldenQuestion(id="Q2", category="E", question="нет данных", answerable=False)
    retriever = _FakeRetriever({})

    class _S:
        corpus_version = "v1"
        prompt_version = "p1"
        gen_model = "claude-opus-5"

    rows = run([q], retriever.search, mode=retriever.name, k=5, settings=_S())
    assert rows[0].empty is True
    assert rows[0].retrieved_source_ids == []


def test_runner_with_generate_hook_fills_refused_and_model() -> None:
    from ergovault.models import AnswerPayload

    q = GoldenQuestion(id="Q3", category="E", question="q", answerable=False)
    retriever = _FakeRetriever({})

    def fake_generate(question, hits):
        return AnswerPayload(answer_markdown="данных нет", refused=True, meta={"model": "test-model"})

    class _S:
        corpus_version = "v1"
        prompt_version = "p1"
        gen_model = "claude-opus-5"

    rows = run([q], retriever.search, mode=retriever.name, k=5, settings=_S(), generate=fake_generate)
    assert rows[0].refused is True
    assert rows[0].model == "test-model"


# --- metrics.py: recall@k / MRR / отказ — посчитано вручную ---------------------------


def _row(
    qid: str, category: str, answerable: bool, expected: list[str],
    retrieved: list[tuple[str, int]], empty: bool = False, refused: bool | None = None,
) -> RunRow:
    from ergovault.eval.runner import RetrievedItem

    return RunRow(
        question_id=qid, category=category, question="q", answerable=answerable,
        expected_source_ids=expected,
        retrieved=[RetrievedItem(rank=r, chunk_id=f"{sid}#000", source_id=sid, score=1.0) for sid, r in retrieved],
        retrieved_source_ids=[sid for sid, _ in sorted(retrieved, key=lambda t: t[1])],
        empty=empty, refused=refused,
        corpus_version="v1", prompt_version="p1", retrieval_mode="fake",
    )


def test_recall_and_mrr_hand_computed() -> None:
    rows = [
        # expected {s1,s2}; найдены s3(1), s1(2), s2(3) -> recall=1.0, RR=1/2=0.5
        _row("Q1", "A", True, ["s1", "s2"], [("s3", 1), ("s1", 2), ("s2", 3)]),
        # expected {s1}; найдены s2(1), s3(2) -> recall=0.0, RR=0
        _row("Q2", "B", True, ["s1"], [("s2", 1), ("s3", 2)]),
        # answerable=False, expected=[] -> вне знаменателя recall/MRR
        _row("Q3", "E", False, [], [], empty=True),
    ]
    report = compute_metrics(rows, k=3)
    assert report.n_with_expected == 2
    assert report.recall_at_k == pytest.approx((1.0 + 0.0) / 2)
    assert report.mrr == pytest.approx((0.5 + 0.0) / 2)

    assert report.empty_rate_all == pytest.approx(1 / 3)
    assert report.empty_rate_answerable == pytest.approx(0 / 2)
    assert report.empty_rate_unanswerable == pytest.approx(1 / 1)

    by_cat = {c.category: c for c in report.by_category}
    assert by_cat["A"].recall_at_k == pytest.approx(1.0)
    assert by_cat["A"].n == 1
    assert by_cat["B"].recall_at_k == pytest.approx(0.0)
    assert by_cat["E"].n == 0  # у E нет expected_source_ids — вне recall/MRR


def test_recall_partial_match_is_fractional() -> None:
    # expected три источника, найден только один -> recall = 1/3, не 0 и не 1 (не hit-rate).
    rows = [_row("Q1", "B", True, ["s1", "s2", "s3"], [("s1", 1)])]
    report = compute_metrics(rows, k=5)
    assert report.recall_at_k == pytest.approx(1 / 3)


def test_refusal_metrics_hand_computed() -> None:
    rows = [
        _row("R1", "A", True, ["s1"], [("s1", 1)], refused=False),  # answerable, не отказал — ок
        _row("R2", "A", True, ["s1"], [], refused=True, empty=True),  # answerable, но отказал — ложный отказ
        _row("R3", "E", False, [], [], refused=True, empty=True),  # not answerable, отказал — верно
        _row("R4", "E", False, [], [("s9", 1)], refused=False),  # not answerable, не отказал — опасная ошибка
    ]
    report = compute_metrics(rows, k=5)
    assert report.refusal is not None
    assert report.refusal.n == 4
    # predicted_refuse = {R2, R3}; TP (answerable=False среди них) = {R3} -> precision 1/2
    assert report.refusal.precision == pytest.approx(0.5)
    # should_refuse = {R3, R4}; отказал среди них только R3 -> recall 1/2
    assert report.refusal.recall == pytest.approx(0.5)


def test_refusal_metrics_unavailable_without_generator() -> None:
    rows = [_row("Q1", "A", True, ["s1"], [("s1", 1)])]  # refused=None по умолчанию
    report = compute_metrics(rows, k=5)
    assert report.refusal is None
    assert "генератор не запускался" in report.refusal_unavailable_reason


def test_runner_records_search_latency() -> None:
    q = GoldenQuestion(
        id="Q1", category="A", question="q", answerable=True, expected_source_ids=["s1"]
    )

    class _S:
        corpus_version = "v1"
        prompt_version = "p1"
        gen_model = "claude-opus-5"

    rows = run([q], _FakeRetriever({"q": [_hit("s1", 1)]}).search, mode="fake", k=5, settings=_S())
    assert rows[0].latency_ms is not None and rows[0].latency_ms >= 0


def test_latency_median_is_none_without_measurements() -> None:
    # Старые прогоны без замеров: «н/д», а не нулевая задержка.
    rows = [_row("Q1", "A", True, ["s1"], [("s1", 1)])]
    assert compute_metrics(rows, k=5).latency_median_ms is None

    rows = [_row(f"Q{i}", "A", True, ["s1"], [("s1", 1)]) for i in range(3)]
    timed = [
        row.model_copy(update={"latency_ms": ms}) for row, ms in zip(rows, [10.0, 30.0, 20.0])
    ]
    assert compute_metrics(timed, k=5).latency_median_ms == pytest.approx(20.0)


def test_no_expected_rows_gives_none_not_zero() -> None:
    # Ни один вопрос не имеет expected_source_ids -> recall/MRR неопределены (None),
    # а не ложно посчитаны как 0.0 — "пустое значение лучше выдуманного".
    rows = [_row("Q1", "E", False, [], [], empty=True)]
    report = compute_metrics(rows, k=5)
    assert report.recall_at_k is None
    assert report.mrr is None
