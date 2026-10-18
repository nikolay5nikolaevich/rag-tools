"""Поиск: неизменность RRF, переиспользование модели, фильтр профиля в SQL."""

from __future__ import annotations

import json
from functools import partial
from itertools import pairwise
from pathlib import Path
from typing import ClassVar

import psycopg
import pytest

from ergovault.db import upgrade, upsert_chunks, upsert_embeddings, upsert_sources
from ergovault.models import Chunk, Hit, Source
from ergovault.search import (
    CANDIDATE_MULTIPLIER,
    MIN_CANDIDATES,
    RANK_NOT_FOUND,
    fuse,
    limit_per_source,
    search,
)
from ergovault.settings import Settings, get_settings

FIXTURES = Path(__file__).parent / "fixtures"
CORPUS_VERSION = "test-fixture-t20"
DIM = get_settings().embed_dim

# Термин, которого нет в корпусе v1 — только в фикстуре. Нужен, чтобы проверки на top-5
# не зависели от того, загружен ли рядом настоящий корпус в ту же таблицу.
FIXTURE_ONLY_QUERY = "Alpha Expo Group"
FTS_ONLY_CHUNK = "fixture--company-alpha#000"
VECTOR_ONLY_CHUNK = "fixture--topic-hub-crm#003"
PILOT_CHUNK = "fixture--raw-pilot-transcript#000"


# --- юнит-часть: RRF, конфиг, ленивая модель ----------------------------------------


def _source(source_id: str, *, visibility: str = "public") -> Source:
    return Source(
        id=source_id,
        path=f"{source_id}.md",
        type="article",
        title=source_id,
        language="ru",
        stance="нейтральный",
        visibility=visibility,
        content_hash="0" * 64,
    )


def _hit(chunk_id: str, rank: int, *, score: float, components: dict[str, float]) -> Hit:
    source_id = chunk_id.split("#")[0]
    text = f"текст {chunk_id}"
    return Hit(
        chunk=Chunk(
            id=chunk_id,
            source_id=source_id,
            ordinal=int(chunk_id.split("#")[1]),
            heading_path=["H"],
            text=text,
            char_len=len(text),
            token_estimate=len(text) // 3,
            atomic=False,
            quote_anchor=text,
        ),
        source=_source(source_id),
        score=score,
        rank=rank,
        components=components,
    )


class _StubBranch:
    """Ветвь с заранее заданным порядком чанков. Запоминает, с какими k и profile её звали."""

    def __init__(self, chunk_ids: list[str], *, components: dict[str, float] | None = None):
        self._chunk_ids = chunk_ids
        self._components = components or {}
        self.calls: list[tuple[int, str]] = []

    def search(self, query: str, k: int = 10, *, profile: str) -> list[Hit]:
        self.calls.append((k, profile))
        return [
            _hit(cid, rank, score=1.0 / rank, components=dict(self._components))
            for rank, cid in enumerate(self._chunk_ids[:k], start=1)
        ]


def _hybrid(fts_ids: list[str], vector_ids: list[str], **kwargs):
    fts = _StubBranch(fts_ids, components={"fts_ru": 0.5, "fts_simple": 0.25})
    vector = _StubBranch(vector_ids, components={"vector": 0.9})
    return partial(
        fuse,
        fts.search("q", 100, profile="public"),
        vector.search("q", 100, profile="public"),
        **kwargs,
    )


def test_config_line_switches_search_mode(monkeypatch):
    text_hit = _hit("text--t#000", 1, score=1.0, components={})
    vector_hit = _hit("vector--v#000", 1, score=1.0, components={})
    monkeypatch.setattr("ergovault.db.search_fts", lambda *a, **kw: [text_hit])
    monkeypatch.setattr("ergovault.db.search_vector", lambda *a, **kw: [vector_hit])
    for mode, expected in [("fts", ["text--t#000"]), ("vector", ["vector--v#000"])]:
        hits = search(
            None,
            "q",
            profile="public",
            settings=Settings(retrieval_mode=mode),
            encode_query=lambda q: [1.0],
        )
        assert [hit.chunk.id for hit in hits] == expected


def test_unknown_mode_raises_instead_of_falling_back() -> None:
    # Молчаливый откат на режим по умолчанию подписал бы прогон чужой меткой
    # retrieval_mode — результат, недействительный по §10.
    with pytest.raises(ValueError, match="hybrid"):
        search(None, "q", mode="bm25", profile="public")


def test_rrf_score_matches_contract_formula() -> None:
    hybrid = _hybrid(["s--a#000", "s--b#000"], ["s--b#000", "s--c#000"])
    k_rrf = 60
    hits = {hit.chunk.id: hit for hit in hybrid(k=5)}

    assert hits["s--a#000"].score == pytest.approx(1 / (k_rrf + 1))
    assert hits["s--b#000"].score == pytest.approx(1 / (k_rrf + 2) + 1 / (k_rrf + 1))
    assert hits["s--c#000"].score == pytest.approx(1 / (k_rrf + 2))


def test_document_confirmed_by_both_branches_outranks_single_branch_leader() -> None:
    # Смысл RRF и причина, по которой гибрид вообще существует: согласие двух ветвей
    # весит больше, чем первое место в одной из них.
    hybrid = _hybrid(["only--fts#000", "both--doc#000"], ["only--vec#000", "both--doc#000"])
    hits = hybrid(k=3)
    assert hits[0].chunk.id == "both--doc#000"
    assert hits[0].rank == 1


def test_components_carry_contribution_of_each_branch() -> None:
    # §6: без вкладов ветвей нечем отлаживать промахи по именам собственным.
    hybrid = _hybrid(["only--fts#000", "both--doc#000"], ["both--doc#000"])
    hits = {hit.chunk.id: hit for hit in hybrid(k=5)}
    k_rrf = 60

    both = hits["both--doc#000"]
    assert both.components["rank_fts"] == 2.0
    assert both.components["rank_vector"] == 1.0
    assert both.components["rrf_fts"] == pytest.approx(1 / (k_rrf + 2))
    assert both.components["rrf_vector"] == pytest.approx(1 / (k_rrf + 1))
    assert both.components["fts_ru"] == 0.5 and both.components["vector"] == 0.9

    fts_only = hits["only--fts#000"]
    assert fts_only.components["rank_vector"] == RANK_NOT_FOUND
    assert fts_only.components["rrf_vector"] == 0.0
    assert fts_only.components["vector"] == 0.0


def test_weights_scale_branch_contributions() -> None:
    hybrid = _hybrid(["a--x#000"], ["b--y#000"], weight_fts=2.0, weight_vector=0.5)
    hits = {hit.chunk.id: hit for hit in hybrid(k=5)}
    k_rrf = 60
    assert hits["a--x#000"].score == pytest.approx(2.0 / (k_rrf + 1))
    assert hits["b--y#000"].score == pytest.approx(0.5 / (k_rrf + 1))


@pytest.mark.parametrize("profile", ["public", "pilot"])
def test_hybrid_passes_depth_and_profile_to_both_branches(monkeypatch, profile):
    fts, vector = _StubBranch(["a--a#000"]), _StubBranch(["b--b#000"])
    monkeypatch.setattr(
        "ergovault.db.search_fts",
        lambda conn, query, k, *, profile, **kw: fts.search(query, k, profile=profile),
    )
    monkeypatch.setattr(
        "ergovault.db.search_vector",
        lambda conn, embedding, k, *, profile: vector.search("q", k, profile=profile),
    )
    hits = search(None, "q", 5, mode="hybrid", profile=profile, encode_query=lambda q: [1.0])
    assert {h.chunk.id for h in hits} == {"a--a#000", "b--b#000"}
    expected = max(5 * CANDIDATE_MULTIPLIER, MIN_CANDIDATES)
    assert fts.calls == [(expected, profile)]
    assert vector.calls == [(expected, profile)]


def test_hybrid_output_is_ranked_and_truncated_to_k() -> None:
    hybrid = _hybrid([f"s--{i}#000" for i in range(8)], [f"s--{i}#000" for i in range(4, 12)])
    hits = hybrid(k=5)
    assert len(hits) == 5
    assert [hit.rank for hit in hits] == [1, 2, 3, 4, 5]
    assert all(a.score >= b.score for a, b in pairwise(hits))


def test_hybrid_order_is_reproducible_on_equal_scores() -> None:
    # У RRF совпадающие суммы вкладов — обычное дело; порядок выдачи всё равно обязан
    # быть одинаковым между прогонами, иначе eval меряет шум порядка словаря.
    ids = ["z--z#000", "a--a#000", "m--m#000"]
    # Ветви отдают один набор во встречном порядке: у крайних документов суммы вкладов
    # совпадают до последнего знака, и решает только тай-брейк.
    first = _hybrid(ids, list(reversed(ids)))(k=3)
    second = _hybrid(list(reversed(ids)), ids)(k=3)
    assert [h.chunk.id for h in first] == [h.chunk.id for h in second]
    assert first[0].chunk.id == "m--m#000"  # тай-брейк по chunk_id, не по порядку словаря


def test_limit_per_source_keeps_best_chunk_of_each_source() -> None:
    ids = ["a--a#000", "a--a#001", "b--b#000", "a--a#002", "c--c#000"]
    hits = [_hit(cid, rank, score=1.0 / rank, components={}) for rank, cid in enumerate(ids, 1)]
    top = limit_per_source(hits, 2)
    assert [hit.chunk.id for hit in top] == ["a--a#000", "b--b#000"]
    assert [hit.rank for hit in top] == [1, 2]


@pytest.mark.parametrize("mode", ["fts", "vector", "hybrid"])
def test_top_k_is_not_spent_on_one_source(monkeypatch, mode) -> None:
    # Промах базового прогона: четыре чанка одного источника занимали top-5, и второй
    # нужный источник не помещался. Теперь следующий источник поднимается на освободившееся место.
    same = ["a--a#000", "a--a#001", "a--a#002", "b--b#000"]
    fts, vector = _StubBranch(same), _StubBranch(same)
    monkeypatch.setattr(
        "ergovault.db.search_fts",
        lambda conn, query, k, *, profile, **kw: fts.search(query, k, profile=profile),
    )
    monkeypatch.setattr(
        "ergovault.db.search_vector",
        lambda conn, embedding, k, *, profile: vector.search("q", k, profile=profile),
    )
    hits = search(None, "q", 2, mode=mode, profile="public", encode_query=lambda q: [1.0])
    assert [hit.chunk.id for hit in hits] == ["a--a#000", "b--b#000"]


def test_original_and_retelling_both_stay_in_top_k() -> None:
    # Пересказ — отдельный источник: ограничение дублей его не прячет, а независимость
    # подтверждений проверяет ответ по derived_of (CONTRACTS.md §7).
    original = _hit("video--orig#000", 1, score=1.0, components={})
    retelling = _hit("theses--orig#000", 2, score=0.5, components={})
    retelling.source.derived = True
    retelling.source.derived_of = "video--orig"
    top = limit_per_source([original, retelling], 5)
    assert [hit.chunk.id for hit in top] == ["video--orig#000", "theses--orig#000"]


class _FakeVector(list):
    def tolist(self) -> list[float]:
        return list(self)


class _FakeModel:
    instances: ClassVar[list[_FakeModel]] = []

    def __init__(self, model_name: str, expected_dim: int) -> None:
        self.model_name = model_name
        self.expected_dim = expected_dim
        self.queries: list[str] = []
        _FakeModel.instances.append(self)

    def encode_queries(self, texts: list[str], **kwargs) -> list[_FakeVector]:
        self.queries.extend(texts)
        return [_FakeVector([0.0] * self.expected_dim) for _ in texts]

    def encode_documents(self, texts: list[str], **kwargs):  # pragma: no cover
        raise AssertionError("сторона запроса обязана кодироваться encode_queries")


def test_vector_retriever_loads_model_lazily_and_encodes_query_side(monkeypatch) -> None:
    # Импорт и конструктор не должны тянуть torch и ~2 ГБ весов; а когда модель нужна —
    # вопрос кодируется encode_queries, а не encode_documents (у e5 это разные префиксы).
    import ergovault.embeddings as embed_model

    embed_model.get_model.cache_clear()
    _FakeModel.instances.clear()
    monkeypatch.setattr(embed_model, "EmbeddingModel", _FakeModel)
    captured: dict = {}

    def fake_search_vector(conn, embedding, k=10, *, profile):
        captured["embedding"] = list(embedding)
        captured["profile"] = profile
        return []

    monkeypatch.setattr("ergovault.db.search_vector", fake_search_vector)

    retriever = partial(search, None, mode="vector")
    assert _FakeModel.instances == [], "модель поднялась на конструкторе, а не при поиске"

    retriever("сколько касаний в follow-up?", 5, profile="public")
    assert len(_FakeModel.instances) == 1
    assert _FakeModel.instances[0].queries == ["сколько касаний в follow-up?"]
    assert len(captured["embedding"]) == DIM
    assert captured["profile"] == "public"

    retriever("другой вопрос", 5, profile="public")
    assert len(_FakeModel.instances) == 1, "модель переиспользуется между вопросами"
    embed_model.get_model.cache_clear()


def test_injected_encoder_replaces_model_entirely(monkeypatch) -> None:
    monkeypatch.setattr("ergovault.db.search_vector", lambda *a, **kw: [])
    retriever = partial(search, None, mode="vector", encode_query=lambda text: [0.5] * DIM)
    retriever("вопрос", 5, profile="public")  # не должно требовать sentence-transformers


# --- часть на живой базе: слияние реальных выдач и фильтр профиля --------------------


def _load_jsonl(path: Path, model: type[Source | Chunk]) -> list:
    return [
        model(**json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@pytest.fixture(scope="module")
def conn():
    database_url = get_settings().database_url
    try:
        upgrade(database_url)
        connection = psycopg.connect(database_url)
    except psycopg.OperationalError as exc:
        pytest.skip(f"нужен живой Postgres — docker compose up -d db && ev db upgrade: {exc}")
        return
    yield connection
    connection.close()


def _clear_fixture(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM sources WHERE corpus_version = %s", (CORPUS_VERSION,))
    conn.commit()


@pytest.fixture(scope="module")
def positions(conn):
    """id фикстурного чанка -> позиция единицы в его one-hot векторе."""
    sources = _load_jsonl(FIXTURES / "sources.sample.jsonl", Source)
    chunks = _load_jsonl(FIXTURES / "chunks.sample.jsonl", Chunk)
    _clear_fixture(conn)
    upsert_sources(conn, sources, corpus_version=CORPUS_VERSION)
    upsert_chunks(conn, chunks, corpus_version=CORPUS_VERSION)
    by_id = {chunk.id: index for index, chunk in enumerate(chunks)}
    upsert_embeddings(conn, {cid: _one_hot(pos) for cid, pos in by_id.items()}, dim=DIM)
    yield by_id
    _clear_fixture(conn)


def _one_hot(position: int) -> list[float]:
    vector = [0.0] * DIM
    vector[position] = 1.0
    return vector


def _encoder_for(positions: dict[str, int], chunk_id: str):
    """Энкодер запроса, всегда возвращающий вектор конкретного чанка: kNN становится
    детерминированным, а настоящая модель в тестах не поднимается."""
    return lambda _query: _one_hot(positions[chunk_id])


def _hybrid_over_db(conn, positions, chunk_id: str):
    return partial(search, conn, mode="hybrid", encode_query=_encoder_for(positions, chunk_id))


def test_hybrid_unites_what_only_one_branch_found(conn, positions) -> None:
    hybrid = _hybrid_over_db(conn, positions, VECTOR_ONLY_CHUNK)
    hits = {hit.chunk.id: hit for hit in hybrid(FIXTURE_ONLY_QUERY, 5, profile="public")}

    assert FTS_ONLY_CHUNK in hits, "документ, найденный только полнотекстовой ветвью, потерян"
    assert VECTOR_ONLY_CHUNK in hits, "документ, найденный только векторной ветвью, потерян"

    from_fts = hits[FTS_ONLY_CHUNK].components
    assert from_fts["rank_fts"] >= 1.0 and from_fts["rrf_fts"] > 0
    assert from_fts["rank_vector"] == RANK_NOT_FOUND

    from_vector = hits[VECTOR_ONLY_CHUNK].components
    assert from_vector["rank_vector"] == 1.0 and from_vector["rrf_vector"] > 0
    assert from_vector["rank_fts"] == RANK_NOT_FOUND


@pytest.mark.parametrize("mode", ["fts", "vector", "hybrid"])
def test_every_hit_carries_components(conn, positions, mode: str) -> None:
    # Приёмка T-20: components заполнен у каждого Hit во всех трёх режимах.
    hits = search(
        conn,
        FIXTURE_ONLY_QUERY,
        5,
        mode=mode,
        profile="public",
        encode_query=_encoder_for(positions, VECTOR_ONLY_CHUNK),
    )
    assert hits, "фикстурный запрос обязан что-то находить во всех режимах"
    assert all(hit.components for hit in hits)
    assert [hit.rank for hit in hits] == list(range(1, len(hits) + 1))


def test_pilot_chunk_invisible_to_public_profile_in_every_mode(conn, positions) -> None:
    # Запрос — точный вектор pilot-чанка и его же дословный текст: под public он не должен
    # появиться ни в одной ветви. raw/ — расшифровки чужих видео, утечка это «стоп-релиз».
    pilot_query = "расшифровка внутреннего разговора на стенде HubSpot"
    encoder = _encoder_for(positions, PILOT_CHUNK)
    for mode in ("fts", "vector", "hybrid"):
        hits = search(conn, pilot_query, 10, mode=mode, profile="public", encode_query=encoder)
        assert PILOT_CHUNK not in {hit.chunk.id for hit in hits}, mode
        assert all(hit.source.visibility == "public" for hit in hits), mode


def test_pilot_profile_sees_pilot_chunk_in_hybrid(conn, positions) -> None:
    hybrid = _hybrid_over_db(conn, positions, PILOT_CHUNK)
    hits = hybrid("расшифровка внутреннего разговора на стенде HubSpot", 5, profile="pilot")
    assert PILOT_CHUNK in {hit.chunk.id for hit in hits}


def test_full_question_finds_chunk_without_all_its_words(conn, positions) -> None:
    # Исходный промах: plainto_tsquery требовал все слова вопроса, а в чанке нет
    # «занимается» — ветка была пустой. fts_all_terms=0 фиксирует именно этот случай.
    question = "Чем занимается подрядчик Alpha Expo Group на выставках?"
    hits = search(conn, question, 5, mode="fts", profile="public")
    by_id = {hit.chunk.id: hit for hit in hits}
    assert FTS_ONLY_CHUNK in by_id
    assert by_id[FTS_ONLY_CHUNK].components["fts_all_terms"] == 0.0


@pytest.mark.parametrize(
    ("question", "name"),
    [
        ("Какие CRM вроде HubSpot названы в источниках для работы с лидами?", "HubSpot"),
        ("Что источники говорят про специфику работы на GITEX в Дубае?", "GITEX"),
    ],
)
def test_latin_name_inside_full_question_is_found(conn, positions, question, name) -> None:
    hits = search(conn, question, 5, mode="fts", profile="public")
    named = [hit for hit in hits if name in hit.chunk.text]
    assert named, f"{name} не найден в top-5 по полному вопросу"
    assert all(hit.components["fts_simple"] > 0 for hit in named)
    assert all(hit.source.visibility == "public" for hit in hits)


@pytest.mark.parametrize(
    "question",
    ["", "и в на", "O'Reilly & (GITEX) | !HubSpot:* <-> 'x'", "Organic & Natural"],
)
def test_question_text_is_not_parsed_as_tsquery(conn, positions, question) -> None:
    # Операторы tsquery в вопросе — обычный текст; вопрос из одних стоп-слов — пустая
    # выдача, а не ошибка SQL.
    hits = search(conn, question, 5, mode="fts", profile="public")
    if question in ("", "и в на"):
        assert hits == []
    else:
        assert hits
