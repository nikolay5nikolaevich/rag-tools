"""Проверки сравнения четырёх подходов и судьи (T-30).

Метрики считаются на примерах с известным ответом: если расчёт поедет, тест упадёт
раньше, чем цифра попадёт в отчёт. Модель здесь не вызывается — вместо неё подставной
провайдер с заранее заданным текстом.
"""

from __future__ import annotations

import json

import pytest

from ergovault.eval.dataset import GoldenQuestion
from ergovault.eval.experiment import (
    VariantRow,
    _fit_line,
    build_judge_pairs,
    build_summary,
    compare,
    compute_variant_metrics,
    limitations_of,
    load_rows,
    render_report,
    save_rows,
)
from ergovault.eval.judge import (
    JudgeCall,
    JudgePair,
    agreement,
    calibrate,
    cohen_kappa,
    split_claims,
)
from ergovault.eval.variants import SEARCH_VARIANTS, VARIANTS, answer_without_search
from ergovault.llm.base import LLMResponse
from ergovault.settings import Settings


class FakeProvider:
    """Провайдер с заранее известным ответом: тест проверяет наш код, а не модель."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[dict] = []

    def complete(self, *, system, user, model, cache_system=False, max_tokens=4000):
        self.calls.append({"system": system, "user": user, "model": model})
        return LLMResponse(
            text=self.text,
            input_tokens=100,
            output_tokens=20,
            latency_ms=10,
            backend="cli",
            model=model,
            raw_cost_usd=0.01,
            billed_cost_usd=0.001,
        )


def make_row(**kwargs) -> VariantRow:
    base = {
        "question_id": "A-01",
        "category": "A",
        "question": "вопрос",
        "answerable": True,
        "variant": "rag",
        "repeat": 1,
        "corpus_version": "v1",
        "prompt_version": "p1",
        "retrieval_mode": "hybrid",
    }
    return VariantRow(**{**base, **kwargs})


def question(**kwargs) -> GoldenQuestion:
    base = {"id": "A-01", "category": "A", "question": "вопрос", "answerable": True}
    return GoldenQuestion(**{**base, **kwargs})


# --- метрики на примерах с известным ответом --------------------------------------------


def test_recall_and_mrr_on_known_example() -> None:
    rows = [
        # ожидали два источника, нашли один из них на второй позиции → recall 0.5, RR 0.5
        make_row(
            question_id="A-01",
            expected_source_ids=["s1", "s2"],
            retrieved_source_ids=["s9", "s1"],
        ),
        # ожидали один, нашли его первым → recall 1.0, RR 1.0
        make_row(question_id="A-02", expected_source_ids=["s3"], retrieved_source_ids=["s3"]),
    ]
    metrics = compute_variant_metrics(rows, {})
    assert metrics.recall_at_k == pytest.approx(0.75)
    assert metrics.mrr == pytest.approx(0.75)


def test_recall_not_computed_for_variants_without_search() -> None:
    """У `full_corpus` полнота тривиально равна 1 — это не измерение, и её не публикуют."""
    rows = [
        make_row(
            variant="full_corpus",
            expected_source_ids=["s1"],
            retrieved_source_ids=["s1", "s2", "s3"],
        )
    ]
    metrics = compute_variant_metrics(rows, {})
    assert metrics.recall_at_k is None
    assert metrics.mrr is None
    assert "неприменимо" in metrics.recall_note
    assert "full_corpus" not in SEARCH_VARIANTS


def test_must_contain_denominator_only_marked_questions() -> None:
    """Вопрос без must_contain в знаменатель не идёт: иначе метрика разбавляется единицами."""
    questions = {
        "A-01": question(id="A-01", must_contain=["8", "12 недель"]),
        "A-02": question(id="A-02"),  # разметки нет
    }
    rows = [
        make_row(question_id="A-01", must_contain_missing=["12 недель"]),
        make_row(question_id="A-02"),
    ]
    metrics = compute_variant_metrics(rows, questions)
    assert metrics.must_contain_n == 1
    assert metrics.must_contain_rate == 0.0


def test_refusal_precision_and_recall() -> None:
    rows = [
        make_row(question_id="E-01", answerable=False, refused=True),  # верный отказ
        make_row(question_id="E-02", answerable=False, refused=False),  # пропущенный отказ
        make_row(question_id="A-01", answerable=True, refused=True),  # лишний отказ
        make_row(question_id="A-02", answerable=True, refused=False),
    ]
    metrics = compute_variant_metrics(rows, {})
    assert metrics.refusal_precision == pytest.approx(0.5)  # из двух отказов верен один
    assert metrics.refusal_recall == pytest.approx(0.5)  # из двух нужных сделан один


def test_error_rows_do_not_count_as_answers() -> None:
    """Сбой генерации — не отказ и не ответ: он виден отдельно и не улучшает метрики."""
    rows = [make_row(error="LLMError: таймаут"), make_row(refused=False)]
    metrics = compute_variant_metrics(rows, {})
    assert metrics.n_errors == 1
    assert metrics.refusal_n == 1


def test_unmeasured_is_none_not_zero() -> None:
    metrics = compute_variant_metrics([make_row()], {})
    assert metrics.grounded_rate is None  # судья не запускался
    assert metrics.must_contain_rate is None  # размеченных вопросов нет
    assert metrics.refusal_precision is None  # ни одного отказа


# --- независимость источников -------------------------------------------------------------


def test_single_org_and_derived_shares_come_from_limitations() -> None:
    rows = [
        make_row(limitation_kinds=["single_org"], distinct_orgs=1),
        make_row(question_id="A-02", limitation_kinds=["derived_only"], distinct_orgs=2),
    ]
    metrics = compute_variant_metrics(rows, {})
    assert metrics.single_org_share == pytest.approx(0.5)
    assert metrics.derived_only_share == pytest.approx(0.5)
    assert metrics.median_distinct_orgs == pytest.approx(1.5)


# --- отчёт и summary.json --------------------------------------------------------------------


def _comparison(rows, questions):
    return compare(
        rows,
        questions,
        settings=Settings(),
        k=8,
        include_holdout=False,
        profile="public",
        run_path="eval/runs/test.jsonl",
    )


def test_report_marks_groundedness_unmeasured_without_calibration() -> None:
    comparison = _comparison([make_row(refused=False)], [question()])
    report = render_report(comparison)
    assert "## Подтверждённость утверждений" in report
    assert "**Не измерена.**" in report
    assert any("не измерена" in item for item in limitations_of(comparison))


def test_summary_hides_groundedness_until_judge_is_calibrated() -> None:
    rows = [make_row(refused=False, claim_verdicts=[{"verdict": "supported"}])]
    comparison = _comparison(rows, [question()])
    assert comparison.variants[0].grounded_rate == 1.0  # посчитано
    summary = build_summary(comparison)
    assert summary["runs"][0]["grounded_rate"] is None  # но не опубликовано
    assert summary["judge"] is None
    assert summary["sample_size_caveat"]
    assert summary["limitations"]


def test_summary_is_json_serialisable() -> None:
    comparison = _comparison([make_row(refused=False)], [question()])
    json.loads(json.dumps(build_summary(comparison), ensure_ascii=False))


def test_small_difference_is_not_called_a_win() -> None:
    questions = [question(id="A-01", must_contain=["8"]), question(id="A-02", must_contain=["8"])]
    rows = [
        make_row(question_id="A-01", variant="rag"),
        make_row(question_id="A-02", variant="rag", must_contain_missing=["8"]),
        make_row(question_id="A-01", variant="fts"),
        make_row(question_id="A-02", variant="fts", must_contain_missing=["8"]),
    ]
    items = limitations_of(_comparison(rows, questions))
    assert any("не доказан" in item for item in items)


def test_single_run_says_spread_not_measured() -> None:
    comparison = _comparison([make_row()], [question()])
    assert any("разброс между прогонами не измерен" in item for item in limitations_of(comparison))


def test_retry_pass_replaces_failed_cell_with_the_answer() -> None:
    """Повторный проход дописывается в тот же файл: удачная строка вытесняет ошибку."""
    from ergovault.eval.experiment import dedupe_rows, failed_cells

    failed = make_row(error="LLMError: claude CLI не ответил")
    retried = make_row(refused=False, answer_markdown="Ответ [1].")
    other_failed = make_row(question_id="A-02", error="LLMError: таймаут")

    rows = dedupe_rows([failed, other_failed, retried])
    assert len(rows) == 2
    assert next(r for r in rows if r.question_id == "A-01").error is None
    # Ячейка, где ответа так и нет, остаётся сбоем и в отчёте видна как ошибка.
    assert failed_cells(rows) == {("A-02", "rag", 1)}


def test_rejected_answer_is_a_result_and_is_never_replayed() -> None:
    """Ответ, отклонённый проверкой §7, — результат системы. Повторять его до «удачного»
    значило бы подбирать исход, поэтому ни повтор, ни дедупликация его не трогают."""
    from ergovault.eval.experiment import dedupe_rows, failed_cells

    rejected = make_row(error="AnswerValidationError: цитата не найдена")
    later_success = make_row(refused=False)

    assert failed_cells([rejected]) == set()
    kept = dedupe_rows([rejected, later_success])
    assert len(kept) == 1
    assert kept[0].error.startswith("AnswerValidationError")


def test_incomplete_run_is_announced_first_not_in_a_footnote() -> None:
    rows = [make_row(refused=False), make_row(question_id="A-02", error="LLMError: нет ответа")]
    comparison = _comparison(rows, [question(), question(id="A-02")])

    assert limitations_of(comparison)[0].startswith("Прогон неполный: 1 из 2")
    assert build_summary(comparison)["message"].startswith("Прогон неполный")
    assert "Прогон неполный" in render_report(comparison).split("## Что и на чём")[0]


def test_complete_run_has_no_incompleteness_notice() -> None:
    comparison = _comparison([make_row(refused=False)], [question()])
    assert not any(item.startswith("Прогон неполный") for item in limitations_of(comparison))


def test_rows_survive_save_and_load(tmp_path) -> None:
    rows = [make_row(fragment_chunk_ids=["s1#001"], prompt_chars=1234)]
    path = tmp_path / "run.jsonl"
    save_rows(rows, path)
    assert load_rows(path) == rows


# --- судья ------------------------------------------------------------------------------------


def test_split_claims_keeps_only_sentences_with_references() -> None:
    claims = split_claims(
        "Базовое правило — восемь касаний и пауза [1]. Это предложение без ссылки. "
        "Второе утверждение с двумя ссылками [2][3]."
    )
    assert [c.refs for c in claims] == [[1], [2, 3]]
    assert "[1]" not in claims[0].text


def test_cohen_kappa_known_values() -> None:
    same = ["supported", "partially", "not_supported", "supported"]
    assert cohen_kappa(same, same) == pytest.approx(1.0)
    # Полное расхождение при равных долях меток даёт отрицательную kappa.
    assert cohen_kappa(["supported", "not_supported"], ["not_supported", "supported"]) < 0
    # Обе разметки поставили одну метку везде: kappa не определена, и это не 1.0.
    assert cohen_kappa(["supported", "supported"], ["supported", "supported"]) is None


def test_agreement_counts_only_labelled_pairs() -> None:
    metrics = agreement(["supported", "partially"], ["supported", "supported"])
    assert metrics.n == 2
    assert metrics.observed_agreement == pytest.approx(0.5)
    assert metrics.confusion["partially→supported"] == 1


def _call(pair_id: str, verdict: str, repeat: int, config: str = "j1@m") -> JudgeCall:
    return JudgeCall(
        pair_id=pair_id,
        config=config,
        model="m",
        prompt_version="j1",
        prompt_hash="abc",
        repeat=repeat,
        verdict=verdict,
        reason="",
        latency_ms=1,
        input_tokens=1,
        output_tokens=1,
        cost_usd=0.0,
        raw_cost_usd=None,
    )


def test_calibration_without_human_labels_is_not_published() -> None:
    pairs = [JudgePair(id="JP-01", claim="утверждение", chunk_id="s1#001")]
    calls = [_call("JP-01", "supported", 1), _call("JP-01", "partially", 2)]
    calibration = calibrate(pairs, calls, generated_at="20260912T000000Z")

    assert calibration.labeled_enough is False
    assert "не измерено" in calibration.note or "не измерена" in calibration.note
    config = calibration.judge_configs[0]
    assert config.vs_human is None  # согласия с человеком нет, а не «ноль»
    assert config.self_consistency == 0.0  # два повтора разошлись
    assert config.unstable_pairs == ["JP-01"]


def test_calibration_uses_only_human_labelled_pairs() -> None:
    pairs = [
        JudgePair(id="JP-01", claim="a", chunk_id="s1#001", human_verdict="supported"),
        JudgePair(id="JP-02", claim="b", chunk_id="s1#002"),  # не размечена
    ]
    calls = [_call("JP-01", "supported", 1), _call("JP-02", "not_supported", 1)]
    config = calibrate(pairs, calls, generated_at="ts").judge_configs[0]

    assert config.n_pairs == 2
    assert config.n_labeled == 1
    assert config.vs_human.observed_agreement == pytest.approx(1.0)


def test_calibration_compares_two_judge_configurations() -> None:
    pairs = [JudgePair(id="JP-01", claim="a", chunk_id="s1#001")]
    calls = [_call("JP-01", "supported", 1), _call("JP-01", "partially", 1, config="j2@other")]
    calibration = calibrate(pairs, calls, generated_at="ts")
    assert calibration.between_configs.n == 1
    assert calibration.between_configs.observed_agreement == pytest.approx(0.0)


def test_judge_answer_outside_schema_is_not_a_verdict() -> None:
    """Судья, придумавший четвёртое значение, даёт ошибку, а не `not_supported`."""
    from ergovault.eval.judge import JudgeConfig, JudgeError, judge_pair

    provider = FakeProvider(json.dumps({"verdict": "недостаточно данных", "reason": "…"}))
    with pytest.raises(JudgeError):
        judge_pair("утверждение", "фрагмент", config=JudgeConfig(model="m"), provider=provider)


def test_unparsed_judge_answers_are_reported_not_hidden() -> None:
    pairs = [JudgePair(id="JP-01", claim="a", chunk_id="s1#001")]
    calibration = calibrate(
        pairs, [_call("JP-01", "supported", 1)], generated_at="ts", unparsed_calls=2
    )
    assert calibration.unparsed_calls == 2
    assert "не по схеме" in calibration.note


def test_judge_pairs_include_distractors_and_never_prefill_verdicts() -> None:
    rows = [
        make_row(
            question_id=f"A-{i:02d}",
            answer_markdown=f"Утверждение номер {i} про восемь касаний [1].",
            fragment_chunk_ids=["s1#001", "s2#001", "s3#001"],
        )
        for i in range(1, 11)
    ]
    pairs = build_judge_pairs(rows, target=10)
    assert len(pairs) == 10
    kinds = {p.kind for p in pairs}
    assert kinds == {"cited", "distractor"}
    assert all(p.chunk_id == "s1#001" for p in pairs if p.kind == "cited")
    assert all(p.chunk_id != "s1#001" for p in pairs if p.kind == "distractor")
    assert len({p.id for p in pairs}) == 10


# --- вариант без поиска -----------------------------------------------------------------------


def test_answer_without_search_rejects_invented_references() -> None:
    """Вариант без фрагментов не может ссылаться на них: такой ответ — ошибка, не отказ."""
    from ergovault.answering import AnswerValidationError

    provider = FakeProvider(
        json.dumps({"answer_markdown": "Восемь касаний [1].", "cited": [1], "refused": False})
    )
    with pytest.raises(AnswerValidationError):
        answer_without_search("вопрос", settings=Settings(), provider=provider)


def test_answer_without_search_keeps_refusal_and_meta() -> None:
    provider = FakeProvider(
        json.dumps({"answer_markdown": "Таких данных у меня нет.", "cited": [], "refused": True})
    )
    payload = answer_without_search("вопрос", settings=Settings(), provider=provider)

    assert payload.refused is True
    assert [limitation.kind for limitation in payload.limitations] == ["no_data"]
    assert payload.citations == []
    assert payload.meta["retrieval_mode"] == "no_search"
    assert payload.meta["prompt_version"].endswith("-noctx")
    assert payload.meta["prompt_chars"] > 0


def test_variant_names_are_the_four_of_the_ticket() -> None:
    assert VARIANTS == ("no_search", "fts", "rag", "full_corpus")


def test_judge_pairs_file_lives_outside_golden_dir() -> None:
    """`load_golden_set` читает из eval/golden все *.yml как вопросы: файл пар там сломал бы
    загрузку проверочного набора, поэтому он лежит рядом, а не внутри."""
    from ergovault.eval.dataset import DEFAULT_GOLDEN_DIR, load_golden_set
    from ergovault.eval.judge import PAIRS_PATH

    assert PAIRS_PATH.parent != DEFAULT_GOLDEN_DIR
    assert len(load_golden_set()) == 23


# --- служебная часть вызова --------------------------------------------------------------------


def test_line_fit_recovers_slope_and_intercept() -> None:
    """Оверхед вызова оценивается прямой по точкам разного размера промпта."""
    points = [(1000.0, 33_400.0), (5000.0, 35_000.0), (25000.0, 43_000.0)]
    slope, intercept = _fit_line(points)
    assert slope == pytest.approx(0.4, abs=0.02)
    assert intercept == pytest.approx(33_000, abs=300)
    assert _fit_line([(1.0, 2.0)]) is None
