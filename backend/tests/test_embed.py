"""Проверки embeddings.py и indexing.py: модель, кэш, повторные прогоны и запись в БД."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from ergovault import indexing
from ergovault.embeddings import EmbedDimensionMismatch, EmbeddingModel

FIXTURES = Path(__file__).parent / "fixtures"


# --- фейковый sentence_transformers: детерминированные векторы без сети/весов ---------


class _FakeSentenceTransformer:
    instances: ClassVar[list[_FakeSentenceTransformer]] = []

    def __init__(self, model_name: str, dim: int = 8) -> None:
        self.model_name = model_name
        self.device = "cpu"
        self.dim = dim
        self.encode_calls: list[dict] = []
        _FakeSentenceTransformer.instances.append(self)

    def get_sentence_embedding_dimension(self) -> int:
        return self.dim

    def encode(
        self, inputs, *, batch_size, show_progress_bar, convert_to_numpy, normalize_embeddings
    ):
        self.encode_calls.append({"inputs": list(inputs), "batch_size": batch_size})
        rng = np.random.default_rng(abs(hash(tuple(inputs))) % (2**32))
        return rng.random((len(inputs), self.dim)).astype(np.float32)


@pytest.fixture
def fake_sentence_transformers(monkeypatch: pytest.MonkeyPatch):
    _FakeSentenceTransformer.instances.clear()
    fake_module = types.SimpleNamespace(SentenceTransformer=_FakeSentenceTransformer)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    yield _FakeSentenceTransformer


# --- model.py: префиксы query:/passage:, проверка размерности -------------------------


def test_bge_m3_has_no_prefix(fake_sentence_transformers) -> None:
    model = EmbeddingModel("BAAI/bge-m3", expected_dim=8)
    model.encode_documents(["текст"])
    calls = fake_sentence_transformers.instances[0].encode_calls
    assert calls[0]["inputs"] == ["текст"]


def test_e5_uses_query_passage_prefixes(fake_sentence_transformers) -> None:
    model = EmbeddingModel("intfloat/multilingual-e5-large", expected_dim=8)
    model.encode_documents(["текст документа"])
    model.encode_queries(["текст запроса"])
    calls = fake_sentence_transformers.instances[0].encode_calls
    assert calls[0]["inputs"] == ["passage: текст документа"]
    assert calls[1]["inputs"] == ["query: текст запроса"]


def test_dimension_mismatch_raises_clear_error(fake_sentence_transformers) -> None:
    model = EmbeddingModel("BAAI/bge-m3", expected_dim=1024)  # фейк отдаёт 8
    with pytest.raises(EmbedDimensionMismatch, match="1024"):
        model.encode_documents(["текст"])


def test_batch_size_passed_through(fake_sentence_transformers) -> None:
    model = EmbeddingModel("BAAI/bge-m3", expected_dim=8)
    model.encode_documents(["a", "b", "c"], batch_size=2)
    assert fake_sentence_transformers.instances[0].encode_calls[0]["batch_size"] == 2


def test_vectors_are_float32_and_expected_shape(fake_sentence_transformers) -> None:
    model = EmbeddingModel("BAAI/bge-m3", expected_dim=8)
    vectors = model.encode_documents(["a", "b"])
    assert vectors.shape == (2, 8)
    assert vectors.dtype == np.float32


# --- cli.py: кэш .npy/.ids.json, идемпотентность, --force, инкремент -----------------


def _chunk_row(idx: int, text: str) -> dict:
    return {
        "id": f"fixture--embed-cli#{idx:03d}",
        "source_id": "fixture--embed-cli",
        "ordinal": idx,
        "heading_path": ["_intro"],
        "text": text,
        "char_len": len(text),
        "token_estimate": len(text) // 3,
        "atomic": False,
        "quote_anchor": text[:120],
    }


def _write_chunks_jsonl(index_dir: Path, rows: list[dict]) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    with (index_dir / "chunks.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


class _FakeSettings:
    def __init__(self, index_dir: Path, embed_model: str = "BAAI/bge-m3", embed_dim: int = 8):
        self.index_dir = index_dir
        self.embed_model = embed_model
        self.embed_dim = embed_dim
        self.database_url = "postgresql://test:test@localhost:1/test"


@pytest.fixture
def embed_env(tmp_path, monkeypatch, fake_sentence_transformers):
    index_dir = tmp_path / "index"
    settings = _FakeSettings(index_dir)
    monkeypatch.setattr("ergovault.settings.get_settings", lambda: settings)
    return index_dir, settings


def test_embed_computes_vectors_for_all_chunks_first_run(embed_env) -> None:
    index_dir, settings = embed_env
    rows = [_chunk_row(i, f"текст чанка {i}") for i in range(5)]
    _write_chunks_jsonl(index_dir, rows)

    indexing.embed(batch=2, force=False, write_db=False)

    npy_path, ids_path = indexing._cache_paths(index_dir, settings.embed_model)
    assert npy_path.exists() and ids_path.exists()
    assert json.loads(ids_path.read_text(encoding="utf-8")) == [r["id"] for r in rows]
    assert np.load(npy_path).shape == (5, settings.embed_dim)


def test_embed_repeat_run_recomputes_nothing(embed_env) -> None:
    index_dir, settings = embed_env
    rows = [_chunk_row(i, f"текст чанка {i}") for i in range(5)]
    _write_chunks_jsonl(index_dir, rows)

    indexing.embed(batch=2, force=False, write_db=False)
    npy_path, _ = indexing._cache_paths(index_dir, settings.embed_model)
    first_vectors = np.load(npy_path)
    first_mtime = npy_path.stat().st_mtime_ns
    models_after_first_run = len(_FakeSentenceTransformer.instances)

    indexing.embed(batch=2, force=False, write_db=False)

    assert np.array_equal(first_vectors, np.load(npy_path))
    assert npy_path.stat().st_mtime_ns == first_mtime  # файл не перезаписан
    assert len(_FakeSentenceTransformer.instances) == models_after_first_run  # модель не грузилась


def test_embed_force_recomputes_all(embed_env) -> None:
    index_dir, settings = embed_env
    rows = [_chunk_row(i, f"текст чанка {i}") for i in range(3)]
    _write_chunks_jsonl(index_dir, rows)
    indexing.embed(batch=2, force=False, write_db=False)
    models_after_first_run = len(_FakeSentenceTransformer.instances)

    indexing.embed(batch=2, force=True, write_db=False)

    npy_path, ids_path = indexing._cache_paths(index_dir, settings.embed_model)
    assert np.load(npy_path).shape == (3, settings.embed_dim)
    assert json.loads(ids_path.read_text(encoding="utf-8")) == [r["id"] for r in rows]
    assert len(_FakeSentenceTransformer.instances) > models_after_first_run  # модель загружена снова


def test_embed_incremental_adds_only_new_chunks(embed_env) -> None:
    index_dir, settings = embed_env
    rows = [_chunk_row(i, f"текст чанка {i}") for i in range(3)]
    _write_chunks_jsonl(index_dir, rows)
    indexing.embed(batch=2, force=False, write_db=False)
    npy_path, ids_path = indexing._cache_paths(index_dir, settings.embed_model)
    cached_before = np.load(npy_path)

    _write_chunks_jsonl(index_dir, rows + [_chunk_row(3, "новый чанк")])
    indexing.embed(batch=2, force=False, write_db=False)

    ids_after = json.loads(ids_path.read_text(encoding="utf-8"))
    cached_after = np.load(npy_path)
    assert ids_after == [r["id"] for r in rows] + ["fixture--embed-cli#003"]
    assert cached_after.shape[0] == 4
    assert np.array_equal(cached_before, cached_after[:3])  # старые векторы не пересчитаны


def test_embed_drops_stale_ids_removed_from_corpus(embed_env) -> None:
    index_dir, settings = embed_env
    rows = [_chunk_row(i, f"текст чанка {i}") for i in range(3)]
    _write_chunks_jsonl(index_dir, rows)
    indexing.embed(batch=2, force=False, write_db=False)

    _write_chunks_jsonl(index_dir, rows[:2])  # третий чанк исчез из корпуса
    indexing.embed(batch=2, force=False, write_db=False)

    _, ids_path = indexing._cache_paths(index_dir, settings.embed_model)
    assert json.loads(ids_path.read_text(encoding="utf-8")) == [rows[0]["id"], rows[1]["id"]]


def test_embed_writes_report(embed_env) -> None:
    index_dir, settings = embed_env
    _write_chunks_jsonl(index_dir, [_chunk_row(0, "чанк")])
    indexing.embed(batch=2, force=False, write_db=False)

    report = (index_dir / "embed_report.md").read_text(encoding="utf-8")
    assert settings.embed_model in report
    assert str(settings.embed_dim) in report
    # Отчёт обязан говорить, в каком состоянии осталась база: с --no-write-db это
    # осознанный офлайн-прогон, и отчёт не должен выглядеть как успешная индексация.
    assert "Postgres не обновлён" in report


def test_embed_fails_clearly_without_ingest_output(embed_env) -> None:

    index_dir, _ = embed_env
    index_dir.mkdir(parents=True, exist_ok=True)  # chunks.jsonl отсутствует
    with pytest.raises(ValueError):
        indexing.embed(batch=2, force=False, write_db=False)


# --- запись в Postgres через публичный интерфейс db ---


class _FakeConnection:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def db_spy(monkeypatch):
    """Подменяет psycopg.connect и публичные функции db: `ev embed` не должен
    знать SQL, поэтому тест проверяет вызов интерфейса, а не запрос."""
    import psycopg

    from ergovault import db as repository

    calls: dict = {"vectors": None, "dim": None, "updated": 0}

    def fake_upsert(conn, vectors, *, dim, **kwargs):
        calls["vectors"] = dict(vectors)
        calls["dim"] = dim
        return calls["updated"]

    monkeypatch.setattr(psycopg, "connect", lambda *a, **kw: _FakeConnection())
    monkeypatch.setattr(repository, "upsert_embeddings", fake_upsert)
    monkeypatch.setattr(repository, "embedding_coverage", lambda conn, **kw: (5, 5))
    return calls


def test_embed_writes_vectors_to_db_by_default(embed_env, db_spy) -> None:
    index_dir, settings = embed_env
    rows = [_chunk_row(i, f"текст чанка {i}") for i in range(3)]
    _write_chunks_jsonl(index_dir, rows)
    db_spy["updated"] = 3

    indexing.embed(batch=2, force=False)

    assert list(db_spy["vectors"]) == [row["id"] for row in rows]
    assert db_spy["dim"] == settings.embed_dim
    assert all(len(vec) == settings.embed_dim for vec in db_spy["vectors"].values())


def test_embed_writes_db_even_when_cache_is_up_to_date(embed_env, db_spy) -> None:
    # Кэш и база рассинхронизируются независимо: база могла быть пересоздана уже после
    # того, как векторы посчитаны. Повторный прогон обязан долить их, а не выйти раньше.
    index_dir, _ = embed_env
    rows = [_chunk_row(i, f"текст чанка {i}") for i in range(3)]
    _write_chunks_jsonl(index_dir, rows)
    db_spy["updated"] = 3

    indexing.embed(batch=2, force=False, write_db=False)  # только кэш
    assert db_spy["vectors"] is None

    indexing.embed(batch=2, force=False)  # пересчёта нет, запись есть
    assert list(db_spy["vectors"]) == [row["id"] for row in rows]


def test_embed_reports_vectors_that_matched_no_chunk(embed_env, db_spy) -> None:
    index_dir, _ = embed_env
    _write_chunks_jsonl(index_dir, [_chunk_row(i, f"текст {i}") for i in range(3)])
    db_spy["updated"] = 2  # одна строка в базе не нашлась

    indexing.embed(batch=2, force=False)

    report = (index_dir / "embed_report.md").read_text(encoding="utf-8")
    assert "1 векторов не легли" in report, "расхождение кэша с базой должно быть в отчёте"


def test_embed_fails_when_db_unavailable(embed_env, monkeypatch) -> None:
    import psycopg

    index_dir, settings = embed_env
    _write_chunks_jsonl(index_dir, [_chunk_row(0, "чанк")])

    def refuse(*args, **kwargs):
        raise psycopg.OperationalError("connection refused")

    monkeypatch.setattr(psycopg, "connect", refuse)

    with pytest.raises(RuntimeError, match="кэш сохранён"):
        indexing.embed(batch=2, force=False)

    npy_path, _ = indexing._cache_paths(index_dir, settings.embed_model)
    assert npy_path.exists(), "кэш сохраняется до похода в базу, пересчёт не пропадает"


# --- приёмка: осмысленность векторов на реальной BGE-m3 -------------------------------

_RELEVANT_TEXT = (
    "После закрытия выставки менеджер обязан в течение 48 часов связаться с каждым "
    "собранным лидом: отправить персональное follow-up письмо, зафиксировать контакт "
    "в CRM и передать тёплых клиентов в отдел продаж."
)
_OTHER_TEXTS = [
    "Ежегодная международная выставка проходит в выставочном центре площадью 50 000 кв. м.",
    "Стоимость аренды стенда зависит от площади и расположения в павильоне.",
    "Организаторы выставки предоставляют участникам каталог посетителей за месяц до открытия.",
    "На выставке представлены компании из более чем 40 стран мира.",
    "Регистрация посетителей открывается за три месяца до начала мероприятия.",
    "Деловая программа включает конференции, круглые столы и панельные дискуссии.",
    "Правила ввоза выставочных образцов через таможню требуют карнета АТА.",
    "Виза для участия в международной выставке оформляется по приглашению организатора.",
    "Логистическая компания отвечает за доставку экспонатов до стенда.",
    "Страховка экспонатов оформляется отдельно от страховки участника.",
    "История выставки насчитывает более тридцати лет с момента основания.",
    "Пресс-центр выставки аккредитует журналистов по заявке за две недели.",
    "Питание для участников организовано в фуд-корте на первом этаже павильона.",
    "Wi-Fi на территории выставочного центра бесплатный для всех посетителей.",
    "Парковка для легковых автомобилей платная, для автобусов — бесплатная.",
    "Электропитание стенда заказывается отдельно через технического подрядчика.",
    "Демонтаж стендов начинается через час после официального закрытия выставки.",
    "Сертификат участника выдаётся в последний день работы выставки на стойке регистрации.",
    "Языки деловой программы — русский и английский, синхронный перевод обеспечен.",
]


def test_bge_m3_ranks_relevant_chunk_above_random() -> None:
    pytest.importorskip("sentence_transformers", reason="ev embed требует sentence-transformers")
    pytest.importorskip("torch", reason="ev embed требует torch")

    texts = [_RELEVANT_TEXT, *_OTHER_TEXTS]
    assert len(texts) == 20

    try:
        model = EmbeddingModel("BAAI/bge-m3", expected_dim=1024)
        vectors = model.encode_documents(texts, batch_size=8)
        query_vec = model.encode_queries(["follow-up после выставки"])[0]
    except Exception as exc:  # noqa: BLE001 — загрузка весов падает по-разному (сеть, диск, CUDA); это skip, не assert
        pytest.skip(f"модель BGE-m3 недоступна в этом окружении: {exc}")

    cos_relevant = float(np.dot(query_vec, vectors[0]))
    cos_random = float(np.dot(query_vec, vectors[10]))  # заведомо не по теме follow-up
    assert cos_relevant > cos_random
