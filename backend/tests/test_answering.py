"""Ответ по фрагментам: номера и цитаты, URL из БД, ограничения, отказ против ошибки."""

from __future__ import annotations

import json
import re
from pathlib import Path

import psycopg
import pytest

from ergovault import answering
from ergovault.answering import (
    AnswerValidationError,
    answer,
    answer_hits,
    compute_limitations,
    load_prompt,
    select_quote,
)
from ergovault.eval.dataset import load_golden_set
from ergovault.llm import LLMError, LLMResponse
from ergovault.models import MAX_QUOTE_CHARS, Chunk, Hit, Source
from ergovault.settings import Settings, get_settings

SETTINGS = Settings(
    corpus_version="v-test",
    prompt_version="p1",
    gen_model="claude-opus-5",
    retrieval_mode="hybrid",
    top_k=5,
)
PROVIDER = "организатор-провайдер"


def _source(
    source_id: str,
    *,
    type: str = "video-source",
    title: str | None = None,
    url: str | None = None,
    org: str | None = None,
    stance: str = "нейтральный",
    derived_of: str | None = None,
    note: str | None = None,
) -> Source:
    return Source(
        id=source_id,
        path=f"{source_id}.md",
        type=type,
        title=title or f"Заметка {source_id}",
        url=url,
        language="ru",
        speaker_org=org,
        stance=stance,
        derived=derived_of is not None,
        derived_of=derived_of,
        reliability_note=note,
        content_hash="0" * 64,
    )


SG = _source(
    "video--sales-gravy",
    title="Lead Follow Up Secrets",
    url="https://www.youtube.com/watch?v=EjGI-z-AA3c",
    org="Sales Gravy",
    note="Цифры — иллюстративные оценки спикера, не измеренные данные.",
)
SG_THESES = _source(
    "theses--sales-gravy",
    type="video-theses",
    title="Lead Follow Up Secrets (тезисы)",
    derived_of=SG.id,
    note="Эвристики спикера, в транскрипте не подтверждены.",
)
SG_THESES_NO_NOTE = _source("theses--sales-gravy-2", type="video-theses", derived_of=SG.id)
RGE_A = _source(
    "video--rge-a",
    title="Как работать на выставках",
    url="https://rutube.ru/video/rge-a/",
    org="RusGlobalExport",
    stance=PROVIDER,
)
RGE_B = _source(
    "video--rge-b", title="Подготовка к Organic & Natural", org="RusGlobalExport", stance=PROVIDER
)
RGE_THESES = _source(
    "theses--rge-a", type="video-theses", title="Тезисы о выставках", derived_of=RGE_A.id
)
HUB = _source("hub--posle", type="topic-hub", title="После мероприятия")
CATALOG = {s.id: s for s in [SG, SG_THESES, SG_THESES_NO_NOTE, RGE_A, RGE_B, RGE_THESES, HUB]}

SG_TEXT = (
    "Базовое правило follow-up: 8 касаний, затем пауза 12 недель, затем заход снова. "
    "Типовой плохой вариант — 3–4 хаотичных касания."
)
HUB_TEXT = (
    "Контакты разбираются в тот же вечер: карточка, пометка о договорённости и срок "
    "следующего касания. Без этого через неделю половина визиток не опознаётся."
)
RGE_TEXT = "Обработать 100% контактов через 1–3 дня после возвращения с выставки."


def _hit(source: Source, text: str, *, heading: tuple[str, ...] = ("Основные тезисы",)) -> Hit:
    chunk = Chunk(
        id=f"{source.id}#000",
        source_id=source.id,
        ordinal=0,
        heading_path=list(heading),
        text=text,
        char_len=len(text),
        token_estimate=len(text) // 3,
        atomic=False,
        quote_anchor=text[:120],
    )
    return Hit(chunk=chunk, source=source, score=1.0, rank=1)


class FakeProvider:
    """Провайдер с заранее заданным ответом; запоминает, что ему передали."""

    def __init__(self, reply: dict | str | None = None, *, error: Exception | None = None):
        self.reply = reply
        self.error = error
        self.calls: list[dict] = []

    def complete(self, *, system, user, model, cache_system=False, max_tokens=4000):
        self.calls.append({"system": system, "user": user, "model": model})
        if self.error is not None:
            raise self.error
        text = (
            self.reply
            if isinstance(self.reply, str)
            else json.dumps(self.reply, ensure_ascii=False)
        )
        return LLMResponse(
            text=text,
            input_tokens=1500,
            output_tokens=200,
            latency_ms=2100,
            backend="cli",
            model=model,
            raw_cost_usd=0.07,
            billed_cost_usd=0.0125,
        )


def _reply(text: str, cited: list[int], refused: bool = False) -> dict:
    return {"answer_markdown": text, "cited": cited, "refused": refused}


def _ask(reply, hits: list[Hit], question: str = "Как работать с контактами после выставки?"):
    provider = FakeProvider(reply)
    payload = answer_hits(
        question,
        hits,
        sources=CATALOG,
        retrieval_mode="hybrid",
        settings=SETTINGS,
        provider=provider,
    )
    return payload, provider


def _two_hits() -> list[Hit]:
    return [_hit(SG, SG_TEXT), _hit(HUB, HUB_TEXT, heading=("Обработка контактов",))]


def _kinds(limitations) -> list[str]:
    return [lim.kind for lim in limitations]


# --- ссылки и цитаты -------------------------------------------------------------------------


def test_citations_are_built_from_passed_chunks_and_urls_from_sources() -> None:
    hits = _two_hits()
    payload, _ = _ask(
        _reply(
            "Правило: 8 касаний, затем пауза 12 недель [1]. Контакты разбирают вечером [2].", [1, 2]
        ),
        hits,
    )
    assert payload.refused is False
    assert [c.n for c in payload.citations] == [1, 2]
    first, second = payload.citations
    assert (first.chunk_id, first.source_id) == (hits[0].chunk.id, SG.id)
    assert (first.source_title, first.source_type) == (SG.title, "video-source")
    assert first.heading_path == ["Основные тезисы"]
    assert first.url == SG.url  # из Source, модель URL не пишет
    assert second.url is None  # у источника URL нет — заглушку не подставляем
    for citation, hit in zip(payload.citations, hits):
        assert citation.quote and citation.quote in hit.chunk.text


@pytest.mark.parametrize("number", [0, 3])
def test_reference_to_missing_fragment_is_rejected(number: int) -> None:
    with pytest.raises(AnswerValidationError, match="несуществующие фрагменты"):
        _ask(_reply(f"Правило: 8 касаний [1]. Ещё факт [{number}].", [1, number]), _two_hits())


def test_number_in_text_but_not_in_cited_is_rejected() -> None:
    with pytest.raises(AnswerValidationError, match="есть в тексте"):
        _ask(_reply("8 касаний [1]. Разбор вечером [2].", [1]), _two_hits())


def test_extra_number_in_cited_is_rejected() -> None:
    with pytest.raises(AnswerValidationError, match="указаны в cited"):
        _ask(_reply("8 касаний [1].", [1, 2]), _two_hits())


def test_fabricated_quote_is_rejected() -> None:
    with pytest.raises(AnswerValidationError, match="не найдена"):
        _ask(_reply("Спикер советует «20 касаний за три месяца» [1].", [1]), _two_hits())


def test_quote_must_come_from_the_fragment_it_cites() -> None:
    quote = "Спикер пишет: «разбираются в тот же вечер» [{n}]."
    with pytest.raises(AnswerValidationError, match="не найдена во фрагментах \\[1\\]"):
        _ask(_reply(quote.format(n=1), [1]), _two_hits())
    payload, _ = _ask(_reply(quote.format(n=2), [2]), _two_hits())
    assert [c.n for c in payload.citations] == [2]


def test_verbatim_quote_tolerates_case_spacing_markup_dashes_and_ellipsis() -> None:
    text = "Правило: «**8 касаний**,  затем   пауза 12 недель…заход снова» [1], а не «3-4 хаотичных касания» [1]."
    payload, _ = _ask(_reply(text, [1]), _two_hits())
    assert payload.citations[0].quote in SG_TEXT


def test_quoted_source_title_counts_as_quote_of_that_fragment() -> None:
    payload, _ = _ask(
        _reply("В заметке «После мероприятия» [2] — разбор вечером.", [2]), _two_hits()
    )
    assert payload.citations[0].source_id == HUB.id


def test_quote_is_exact_substring_of_long_chunk_near_the_claim() -> None:
    filler = "Общие рассуждения о подготовке стенда и бюджете участия без конкретики. " * 8
    text = filler + "Инструменты: CRM HubSpot, Salesforce, Pipedrive; команда — Asana. " + filler
    payload, _ = _ask(
        _reply("Названы CRM HubSpot, Salesforce и Pipedrive [1].", [1]), [_hit(SG, text)]
    )
    quote = payload.citations[0].quote
    assert quote in text
    assert len(quote) <= MAX_QUOTE_CHARS
    assert quote.startswith("Инструменты: CRM HubSpot")


def test_select_quote_without_overlap_starts_at_chunk_beginning() -> None:
    text = "Первое предложение фрагмента. " * 20
    quote = select_quote(text, "совершенно другая тема")
    assert text.startswith(quote)
    assert 0 < len(quote) <= MAX_QUOTE_CHARS


def test_quote_does_not_start_with_closing_markdown_of_previous_sentence() -> None:
    # Живой ответ давал цитату «** Спикер называет…»: жирный абзац кончался точкой.
    text = "**Follow-up начинается сразу после выставки.** Спикер называет 30-45 контактов."
    quote = select_quote(text, "30-45 контактов на сотрудника")
    assert quote == "Спикер называет 30-45 контактов."
    assert quote in text


def test_model_written_url_is_rejected() -> None:
    with pytest.raises(AnswerValidationError, match="URL"):
        _ask(_reply("Подробности: https://example.com/video [1].", [1]), _two_hits())


def test_reference_group_is_split_into_single_references() -> None:
    payload, _ = _ask(_reply("Контакты обрабатывают быстро [1, 2].", [1, 2]), _two_hits())
    assert "[1][2]" in payload.answer_markdown
    assert [c.n for c in payload.citations] == [1, 2]


# --- промпт ----------------------------------------------------------------------------------


def test_prompt_carries_fragments_but_not_service_fields() -> None:
    hits = [_hit(SG, SG_TEXT), _hit(RGE_A, RGE_TEXT)]
    question = "Когда начинать follow-up?"
    _, provider = _ask(_reply("8 касаний [1]. Через 1–3 дня [2].", [1, 2]), hits, question)
    call = provider.calls[0]
    user, system = call["user"], call["system"]

    assert call["model"] == "claude-opus-5"
    for expected in [
        question,
        SG.title,
        "video-source",
        "Основные тезисы",
        SG_TEXT,
        RGE_TEXT,
        '<fragment n="2">',
    ]:
        assert expected in user
    for hidden in [
        SG.url,
        RGE_A.url,
        "Sales Gravy",
        "RusGlobalExport",
        PROVIDER,
        SG.reliability_note,
        "derived",
    ]:
        assert hidden not in user
        assert hidden not in system


def test_prompt_carries_source_metadata_for_questions_about_materials() -> None:
    """Язык, дата и длительность нужны в промпте: на вопросы о самих материалах отвечают
    поля Source, а не текст заметки. Отсутствующее значение остаётся отсутствующим."""
    english = _source("video--english", title="Lead Follow Up", url="https://example.com/v")
    english = english.model_copy(update={"language": "en", "published": "2024-05-01"})
    hits = [_hit(english, SG_TEXT), _hit(RGE_A, RGE_TEXT)]

    fragments = answering.render_fragments(hits)

    assert "Язык: английский" in fragments
    assert "Дата публикации: 2024-05-01" in fragments
    # У второго источника даты и длительности нет — подставлять нечего.
    assert "Дата публикации: не указана" in fragments
    assert "Длительность: не указана" in fragments
    assert english.url not in fragments


def test_unknown_prompt_version_is_an_error() -> None:
    with pytest.raises(ValueError, match="p999"):
        load_prompt("p999")


# --- ограничения -----------------------------------------------------------------------------


def test_one_known_organization_gives_single_org_and_provider_stance() -> None:
    limitations = compute_limitations([RGE_A, RGE_B], refused=False, sources=CATALOG)
    assert _kinds(limitations) == ["single_org", "provider_stance"]
    assert "RusGlobalExport" in limitations[0].text
    assert limitations[0].source_ids == [RGE_A.id, RGE_B.id]
    assert limitations[1].source_ids == [RGE_A.id, RGE_B.id]


def test_single_citation_of_one_organization_is_single_org() -> None:
    # A-04: один источник РусГлобалЭкспорта — подтверждение одно (§7: все цитаты с одним org).
    assert "single_org" in _kinds(compute_limitations([RGE_A], refused=False, sources=CATALOG))


def test_unknown_organization_is_not_a_common_organization() -> None:
    limitations = compute_limitations([RGE_A, HUB], refused=False, sources=CATALOG)
    assert _kinds(limitations) == ["provider_stance"]
    assert limitations[0].source_ids == [RGE_A.id]
    assert "single_org" not in _kinds(compute_limitations([HUB], refused=False, sources=CATALOG))


def test_different_organizations_are_not_single_org() -> None:
    kinds = _kinds(compute_limitations([SG, RGE_A], refused=False, sources=CATALOG))
    assert "single_org" not in kinds


def test_original_and_retelling_collapse_into_one_confirmation() -> None:
    # Пересказ без своего speaker_org получает организацию оригинала: Sales Gravy плюс
    # его тезисы — одно подтверждение, а не два независимых.
    limitations = compute_limitations([SG, SG_THESES], refused=False, sources=CATALOG)
    assert _kinds(limitations) == ["single_org", "reliability_note", "reliability_note"]
    assert "Sales Gravy" in limitations[0].text
    assert "derived_only" not in _kinds(limitations)  # оригинал тоже процитирован


def test_retelling_alone_takes_attribution_of_its_original_from_catalog() -> None:
    limitations = compute_limitations([RGE_THESES], refused=False, sources=CATALOG)
    assert _kinds(limitations) == ["single_org", "provider_stance", "derived_only"]
    assert "RusGlobalExport" in limitations[0].text


def test_reliability_note_is_extracted_per_source_and_inherited_by_retelling() -> None:
    limitations = compute_limitations([SG, HUB, SG_THESES_NO_NOTE], refused=False, sources=CATALOG)
    notes = [lim for lim in limitations if lim.kind == "reliability_note"]
    # Пересказ без своей оговорки получает оговорку оригинала. Одинаковый абзац
    # выводится один раз с id обоих источников — на живом E-03 он дублировался.
    assert [lim.source_ids for lim in notes] == [[SG.id, SG_THESES_NO_NOTE.id]]
    assert SG.reliability_note in notes[0].text  # дословно из метаданных


def test_refusal_gives_no_data() -> None:
    assert _kinds(compute_limitations([], refused=True, sources=CATALOG)) == ["no_data"]


def test_answer_limitations_are_computed_from_cited_sources() -> None:
    hits = [
        _hit(RGE_A, RGE_TEXT),
        _hit(RGE_THESES, "Контакты обрабатывают за 1–3 дня."),
        _hit(SG, SG_TEXT),
    ]
    payload, _ = _ask(_reply("Через 1–3 дня [1][2].", [1, 2]), hits)
    # SG передан, но не процитирован — его оговорка в ответ не попадает.
    assert _kinds(payload.limitations) == ["single_org", "provider_stance"]


# --- отказ против ошибки -----------------------------------------------------------------------


def test_lack_of_data_is_a_refusal_with_explanation() -> None:
    text = "В фрагментах нет цены квадратного метра на Hannover Messe."
    payload, _ = _ask(_reply(text, [], refused=True), [_hit(HUB, HUB_TEXT)])
    assert payload.refused is True
    assert payload.answer_markdown == text
    assert payload.citations == []
    assert _kinds(payload.limitations) == ["no_data"]


def test_refusal_may_point_to_adjacent_fragment() -> None:
    payload, _ = _ask(
        _reply(
            "Визовых требований нет; упомянута только проверка виз при подготовке [1].",
            [1],
            refused=True,
        ),
        [_hit(RGE_A, "Перед поездкой проверить визы и сроки оформления документов.")],
    )
    assert payload.refused is True
    assert _kinds(payload.limitations)[0] == "no_data"
    assert [c.n for c in payload.citations] == [1]


def test_refusal_without_explanation_is_rejected() -> None:
    with pytest.raises(AnswerValidationError, match="пустой"):
        _ask(_reply("  ", [], refused=True), _two_hits())


def test_answer_without_references_is_rejected() -> None:
    with pytest.raises(AnswerValidationError, match="без единой ссылки"):
        _ask(_reply("Контакты надо обрабатывать быстро.", []), _two_hits())


def test_model_failure_stays_an_error() -> None:
    provider = FakeProvider(error=LLMError("claude CLI не ответил после 3 попыток"))
    with pytest.raises(LLMError):
        answer_hits(
            "вопрос",
            _two_hits(),
            sources=CATALOG,
            retrieval_mode="hybrid",
            settings=SETTINGS,
            provider=provider,
        )


@pytest.mark.parametrize(
    "reply",
    [
        "Вот ответ: контакты обрабатывают быстро [1].",
        '{"answer_markdown": "8 касаний [1].", "cited": "1", "refused": false}',
        '{"answer_markdown": "8 касаний [1].", "cited": [1]}',
    ],
)
def test_output_outside_json_schema_is_rejected(reply: str) -> None:
    with pytest.raises(AnswerValidationError, match="JSON"):
        _ask(reply, _two_hits())


def test_json_wrapped_in_code_fence_is_accepted() -> None:
    reply = "```json\n" + json.dumps(_reply("8 касаний [1].", [1]), ensure_ascii=False) + "\n```"
    payload, _ = _ask(reply, _two_hits())
    assert [c.n for c in payload.citations] == [1]


def test_validation_error_keeps_raw_text_and_cost_for_the_log() -> None:
    reply = json.dumps(_reply("Выдумка [7].", [7]), ensure_ascii=False)
    with pytest.raises(AnswerValidationError) as exc_info:
        _ask(reply, _two_hits())
    error = exc_info.value
    assert error.raw_text == reply
    assert error.meta["cost_usd"] == 0.0125
    assert error.meta["model"] == "claude-opus-5"


def test_empty_search_is_refused_without_calling_the_model() -> None:
    payload, provider = _ask(_reply("не должно вызываться [1].", [1]), [])
    assert provider.calls == []
    assert payload.refused is True
    assert _kinds(payload.limitations) == ["no_data"]
    assert "поиск не вернул" in payload.limitations[0].text
    assert payload.meta["model"] is None  # модель не вызывалась
    assert payload.meta["cost_usd"] == 0.0


# --- meta и точки входа ----------------------------------------------------------------------


def test_meta_holds_versions_model_time_and_cost_from_provider() -> None:
    payload, _ = _ask(_reply("8 касаний [1].", [1]), _two_hits())
    meta = payload.meta
    assert meta["corpus_version"] == "v-test"
    assert meta["prompt_version"] == "p1"
    assert re.fullmatch(r"[0-9a-f]{12}", meta["prompt_hash"])
    assert meta["retrieval_mode"] == "hybrid"
    assert meta["model"] == "claude-opus-5"
    assert meta["llm_backend"] == "cli"
    assert meta["llm_latency_ms"] == 2100
    assert meta["cost_usd"] == 0.0125  # расчётная по токенам
    assert meta["raw_cost_usd"] == 0.07  # значение бэкенда, хранится отдельно
    assert meta["input_tokens"] == 1500 and meta["output_tokens"] == 200
    assert isinstance(meta["latency_ms"], int) and meta["latency_ms"] >= 0


def test_answer_searches_by_profile_and_uses_catalog_for_attribution(monkeypatch) -> None:
    calls: dict = {}

    def fake_search(conn, query, k, *, profile, mode, settings, encode_query):
        calls["search"] = (query, k, profile, mode)
        return [_hit(RGE_THESES, "Контакты обрабатывают за 1–3 дня после выставки.")]

    def fake_list_sources(conn, profile):
        calls["list_sources"] = profile
        return list(CATALOG.values())

    monkeypatch.setattr(answering, "search", fake_search)
    monkeypatch.setattr(answering.db, "list_sources", fake_list_sources)
    provider = FakeProvider(_reply("За 1–3 дня [1].", [1]))
    payload = answer(
        None, "Когда обрабатывать контакты?", profile="public", settings=SETTINGS, provider=provider
    )

    assert calls["search"] == ("Когда обрабатывать контакты?", 5, "public", "hybrid")
    assert calls["list_sources"] == "public"
    # оригинала нет в выдаче, но каталог даёт пересказу его организацию и позицию
    assert _kinds(payload.limitations) == ["single_org", "provider_stance", "derived_only"]
    assert payload.meta["profile"] == "public"
    assert payload.meta["search_latency_ms"] >= 0


def test_empty_question_is_rejected_before_search() -> None:
    with pytest.raises(ValueError, match="пустой вопрос"):
        answer(None, "   ", profile="public", settings=SETTINGS, provider=FakeProvider())


def test_answering_does_not_depend_on_http_or_cli_frameworks() -> None:
    source = Path(answering.__file__).read_text(encoding="utf-8")
    assert not re.search(r"^\s*(from|import)\s+(typer|fastapi)\b", source, re.MULTILINE)


# --- живая проверка на открытой части набора -------------------------------------------------

_OPEN_SET = load_golden_set()  # holdout не открывается
_REFUSAL_QUESTIONS = [q for q in _OPEN_SET if q.category == "E"]
_NOT_REFUSAL = [q for q in _OPEN_SET if q.id == "C-03"]  # «доказательств нет» — ответ, не отказ


@pytest.fixture(scope="module")
def live_conn():
    try:
        conn = psycopg.connect(get_settings().database_url)
    except psycopg.OperationalError as exc:
        pytest.skip(f"нужен живой Postgres с загруженным корпусом: {exc}")
    yield conn
    conn.close()


@pytest.mark.live
@pytest.mark.parametrize("question", _REFUSAL_QUESTIONS + _NOT_REFUSAL, ids=lambda q: q.id)
def test_live_open_set_refusals(live_conn, question) -> None:
    payload = answer(live_conn, question.question, profile="public")
    assert payload.refused is (not question.answerable), payload.answer_markdown
    if not question.answerable:
        assert "no_data" in _kinds(payload.limitations)
    # Повтор слов самого вопроса («нет дат GITEX в 2027 году») — не выдумка. Буквальный
    # must_not_contain считает и его; эта метрика остаётся для T-30 как есть.
    for forbidden in question.must_not_contain:
        if forbidden not in question.question:
            assert forbidden not in payload.answer_markdown
