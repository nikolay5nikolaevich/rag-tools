"""Тесты на живой базе (docker compose up -d db; ev db upgrade).

У всех источников фикстуры backend/tests/fixtures/{sources,chunks}.sample.jsonl
id под префиксом `fixture--`, чтобы upsert по PK `id` никогда не задел
реально загруженный корпус в той же базе. Изоляция от прогона к прогону — фильтром
по corpus_version, не отдельной БД: используется тот же db-сервис.
"""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path

import psycopg
import pytest

from ergovault.db import (
    embedding_coverage,
    get_chunk,
    list_sources,
    search_fts,
    search_vector,
    upgrade,
    upsert_chunks,
    upsert_embeddings,
    upsert_sources,
)
from ergovault.models import Chunk, Source
from ergovault.settings import get_settings

FIXTURES = Path(__file__).parent / "fixtures"
CORPUS_VERSION = "test-fixture-t11"
# search_fts не фильтрует по corpus_version (это не часть контракта §6) — на общей
# dev-базе там может уже лежать настоящий корпус v1 (`ev db load`). Большой k здесь —
# не про объём выдачи в проде (там TOP_K=8), а про то, чтобы редкий термин из фикстуры
# не вытеснили совпадения из совсем другого набора данных в той же таблице.
SEARCH_K = 1000


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


@pytest.fixture(scope="module", autouse=True)
def loaded_fixture(conn):
    sources = _load_jsonl(FIXTURES / "sources.sample.jsonl", Source)
    chunks = _load_jsonl(FIXTURES / "chunks.sample.jsonl", Chunk)
    _clear_fixture(conn)
    upsert_sources(conn, sources, corpus_version=CORPUS_VERSION)
    upsert_chunks(conn, chunks, corpus_version=CORPUS_VERSION)
    yield sources, chunks
    _clear_fixture(conn)


def test_upgrade_is_idempotent(conn) -> None:
    # `conn`-фикстура уже накатила схему один раз (module scope) — повторный upgrade()
    # не должен ни упасть, ни применить что-то ещё раз.
    database_url = get_settings().database_url
    first = upgrade(database_url)
    second = upgrade(database_url)
    assert first == []
    assert second == []


def test_load_counts_match_fixture(conn, loaded_fixture) -> None:
    sources, chunks = loaded_fixture
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM sources WHERE corpus_version = %s", (CORPUS_VERSION,))
        source_count = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM chunks WHERE corpus_version = %s", (CORPUS_VERSION,))
        chunk_count = cur.fetchone()[0]
    assert source_count == len(sources)
    assert chunk_count == len(chunks)


def test_upsert_is_idempotent_by_pk(conn, loaded_fixture) -> None:
    sources, chunks = loaded_fixture
    upsert_sources(conn, sources, corpus_version=CORPUS_VERSION)
    upsert_chunks(conn, chunks, corpus_version=CORPUS_VERSION)
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM sources WHERE corpus_version = %s", (CORPUS_VERSION,))
        assert cur.fetchone()[0] == len(sources)


def test_derived_of_resolved_after_batch_insert(conn, loaded_fixture) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT derived_of FROM sources WHERE id = 'fixture--theses-trade-show-hubspot'"
        )
        row = cur.fetchone()
    assert row is not None
    assert row[0] == "fixture--video-trade-show-hubspot"


def test_search_gitex_found_via_simple_branch(conn, loaded_fixture) -> None:
    # Эмпирически на pgvector/pgvector:pg16 russian_stem пропускает ASCII-леммы без
    # стемминга, так что «GITEX» здесь совпадает и через tsv_ru, и через tsv_simple —
    # ключевое по контракту (§5/§0) не «russian не находит», а то, что simple всегда
    # рабочий канал для англоязычных имён собственных и components считает его отдельно.
    hits = search_fts(conn, "GITEX", k=SEARCH_K, profile="public")
    ids = {hit.source.id for hit in hits}
    assert "fixture--video-gitex-2023-iglobal" in ids

    hit = next(h for h in hits if h.source.id == "fixture--video-gitex-2023-iglobal")
    assert hit.components["fts_simple"] > 0


def test_search_baselworld_proper_noun_found(conn, loaded_fixture) -> None:
    hits = search_fts(conn, "Baselworld", k=SEARCH_K, profile="public")
    ids = {hit.source.id for hit in hits}
    assert "fixture--article-analitika-statey" in ids


def test_search_hubspot_proper_noun_found(conn, loaded_fixture) -> None:
    hits = search_fts(conn, "HubSpot", k=SEARCH_K, profile="public")
    ids = {hit.source.id for hit in hits}
    assert {"fixture--video-trade-show-hubspot", "fixture--theses-trade-show-hubspot"} & ids


def test_search_morphology_finds_inflected_query(conn, loaded_fixture) -> None:
    hits = search_fts(conn, "обработка контактов после выставки", k=SEARCH_K, profile="public")
    ids = {hit.source.id for hit in hits}
    assert "fixture--article-analitika-statey" in ids

    hit = next(h for h in hits if h.source.id == "fixture--article-analitika-statey")
    assert hit.components["fts_ru"] > 0


def test_hit_components_carry_both_branches_separately(conn, loaded_fixture) -> None:
    hits = search_fts(conn, "выставка контакты", k=SEARCH_K, profile="public")
    assert hits, "фикстура должна давать хотя бы одно совпадение"
    for hit in hits:
        assert {"fts_ru", "fts_simple"} <= set(hit.components)
        assert hit.components["fts_all_terms"] in (0.0, 1.0)
        assert hit.score == pytest.approx(hit.components["fts_ru"] + hit.components["fts_simple"])


def test_search_fts_hides_pilot_from_public_profile(conn, loaded_fixture) -> None:
    hits = search_fts(conn, "HubSpot", k=SEARCH_K, profile="public")
    assert all(hit.source.visibility == "public" for hit in hits)
    assert "fixture--raw-pilot-transcript" not in {hit.source.id for hit in hits}


def test_search_fts_pilot_profile_can_see_pilot_rows(conn, loaded_fixture) -> None:
    hits = search_fts(conn, "HubSpot", k=SEARCH_K, profile="pilot")
    ids = {hit.source.id for hit in hits}
    assert "fixture--raw-pilot-transcript" in ids


def test_list_sources_public_excludes_pilot(conn, loaded_fixture) -> None:
    sources = list_sources(conn, "public")
    fixture_ids = {s.id for s in sources if s.id.startswith("fixture--")}
    assert "fixture--raw-pilot-transcript" not in fixture_ids
    assert "fixture--article-analitika-statey" in fixture_ids


def test_list_sources_pilot_includes_both_visibilities(conn, loaded_fixture) -> None:
    sources = list_sources(conn, "pilot")
    fixture_ids = {s.id for s in sources if s.id.startswith("fixture--")}
    assert "fixture--raw-pilot-transcript" in fixture_ids
    assert "fixture--article-analitika-statey" in fixture_ids


def test_get_chunk_roundtrip(conn, loaded_fixture) -> None:
    _, chunks = loaded_fixture
    original = chunks[0]
    fetched = get_chunk(conn, original.id)
    assert fetched is not None
    assert fetched.model_dump() == original.model_dump()


def test_get_chunk_missing_returns_none(conn, loaded_fixture) -> None:
    assert get_chunk(conn, "fixture--does-not-exist#000") is None


# --- эмбеддинги: интерфейс записи и kNN ---
#
# Векторы фикстуры синтетические и попарно ортогональные (one-hot): косинус с реальными
# BGE-векторами корпуса, если он лежит в той же dev-базе, около нуля, а самосовпадение
# ровно 1.0 — поэтому проверки не зависят от того, загружен ли рядом настоящий корпус.
# Один чанк намеренно остаётся без вектора: «нет эмбеддинга» — отдельный проверяемый
# случай, а не край выдачи.
DIM = get_settings().embed_dim
NO_EMBEDDING_CHUNK = "fixture--wiki-final-report#000"
PILOT_CHUNK = "fixture--raw-pilot-transcript#000"


def _one_hot(position: int) -> list[float]:
    vector = [0.0] * DIM
    vector[position] = 1.0
    return vector


@pytest.fixture(scope="module")
def embedded(conn, loaded_fixture):
    """id фикстурного чанка -> позиция его единицы в one-hot векторе."""
    _, chunks = loaded_fixture
    positions = {
        chunk.id: index
        for index, chunk in enumerate(chunks)
        if chunk.id != NO_EMBEDDING_CHUNK
    }
    upsert_embeddings(
        conn, {cid: _one_hot(pos) for cid, pos in positions.items()}, dim=DIM
    )
    return positions


def test_upsert_embeddings_reports_rows_it_actually_updated(conn, embedded) -> None:
    known = next(iter(embedded))
    updated = upsert_embeddings(
        conn,
        {known: _one_hot(embedded[known]), "fixture--does-not-exist#000": _one_hot(0)},
        dim=DIM,
    )
    assert updated == 1, "несуществующий chunk_id не должен считаться записанным"
    assert get_chunk(conn, "fixture--does-not-exist#000") is None, (
        "репозиторий не создаёт чанки под сироту-эмбеддинг"
    )


def test_upsert_embeddings_rejects_wrong_dimension_without_writing(conn, embedded) -> None:
    with pytest.raises(ValueError, match=str(DIM)):
        upsert_embeddings(conn, {NO_EMBEDDING_CHUNK: [0.1, 0.2, 0.3]}, dim=DIM)
    with conn.cursor() as cur:
        cur.execute("SELECT embedding FROM chunks WHERE id = %s", (NO_EMBEDDING_CHUNK,))
        assert cur.fetchone()[0] is None, "проверка размерности идёт до записи, а не после"


def test_upsert_chunks_does_not_wipe_embedding(conn, loaded_fixture, embedded) -> None:
    _, chunks = loaded_fixture
    upsert_chunks(conn, chunks, corpus_version=CORPUS_VERSION)
    target, position = next(iter(embedded.items()))
    hits = search_vector(conn, _one_hot(position), k=1, profile="pilot")
    assert hits and hits[0].chunk.id == target


def test_search_vector_returns_exact_match_first(conn, embedded) -> None:
    target = "fixture--topic-hub-crm#002"
    hits = search_vector(conn, _one_hot(embedded[target]), k=5, profile="public")
    assert hits[0].chunk.id == target
    assert hits[0].score == pytest.approx(1.0, abs=1e-6)


def test_search_vector_components_carry_distance_and_similarity(conn, embedded) -> None:
    target = "fixture--company-alpha#000"
    hits = search_vector(conn, _one_hot(embedded[target]), k=5, profile="public")
    assert hits, "фикстура должна давать хотя бы одно совпадение"
    for hit in hits:
        assert set(hit.components) == {"vector", "cosine_distance"}
        assert hit.score == pytest.approx(1.0 - hit.components["cosine_distance"])
    assert [hit.rank for hit in hits] == list(range(1, len(hits) + 1))
    assert all(a.score >= b.score for a, b in pairwise(hits))


def test_search_vector_hides_pilot_from_public_profile(conn, embedded) -> None:
    # Запрос — точный вектор pilot-чанка: при profile="public" он был бы первым,
    # если бы фильтр visibility стоял в Python, а не в SQL.
    hits = search_vector(conn, _one_hot(embedded[PILOT_CHUNK]), k=5, profile="public")
    assert PILOT_CHUNK not in {hit.chunk.id for hit in hits}
    assert all(hit.source.visibility == "public" for hit in hits)


def test_search_vector_pilot_profile_sees_pilot_rows(conn, embedded) -> None:
    hits = search_vector(conn, _one_hot(embedded[PILOT_CHUNK]), k=5, profile="pilot")
    assert hits[0].chunk.id == PILOT_CHUNK


def test_search_vector_skips_chunks_without_embedding(conn, loaded_fixture, embedded) -> None:
    _, chunks = loaded_fixture
    position = next(i for i, c in enumerate(chunks) if c.id == NO_EMBEDDING_CHUNK)
    hits = search_vector(conn, _one_hot(position), k=5, profile="pilot")
    assert NO_EMBEDDING_CHUNK not in {hit.chunk.id for hit in hits}


def test_embedding_coverage_counts_only_filled_vectors(conn, loaded_fixture, embedded) -> None:
    _, chunks = loaded_fixture
    total, covered = embedding_coverage(conn, corpus_version=CORPUS_VERSION)
    assert total == len(chunks)
    assert covered == len(chunks) - 1  # ровно один чанк оставлен без вектора
