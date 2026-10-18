"""Контрактные тесты HTTP API (§8) на живой базе: коды ошибок, журнал, фильтр профиля.

База та же, что у остальных тестов (docker compose up -d db; ev db upgrade). Источники
фикстуры лежат под префиксом `fixture--` и своим corpus_version, поэтому не задевают
загруженный рядом корпус v1. Записи журнала, созданные тестами, удаляются по вопросу
с префиксом T22-TEST: журнал — рабочие данные, тестовый мусор в них не остаётся.

Режим поиска во всех запросах — `fts`: векторная ветвь потребовала бы весов модели,
а проверяется здесь HTTP-слой, а не качество выдачи.
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote

import psycopg
import pytest
from fastapi.testclient import TestClient
from psycopg.rows import dict_row

from ergovault import answering, api, db
from ergovault.api import conn_dep, create_app, settings_dep
from ergovault.llm import LLMError, LLMResponse
from ergovault.models import Chunk, Source
from ergovault.settings import Settings, get_settings

FIXTURES = Path(__file__).parent / "fixtures"
CORPUS_VERSION = "test-fixture-t22"
QUESTION_PREFIX = "T22-TEST"
PILOT_SOURCE = "fixture--raw-pilot-transcript"
PILOT_CHUNK = "fixture--raw-pilot-transcript#000"
PUBLIC_CHUNK = "fixture--company-alpha#000"
PILOT_QUERY = f"{QUESTION_PREFIX} черновая расшифровка внутреннего разговора на стенде"
ALPHA_QUERY = f"{QUESTION_PREFIX} чем занимается Alpha Expo Group"

ANSWER_TEXT = json.dumps(
    {"answer_markdown": "Ответ по материалам исследования [1].", "cited": [1], "refused": False},
    ensure_ascii=False,
)
REFUSAL_TEXT = json.dumps(
    {
        "answer_markdown": "В найденных фрагментах нужных сведений нет.",
        "cited": [],
        "refused": True,
    },
    ensure_ascii=False,
)


class FakeProvider:
    """Провайдер без сети: отдаёт готовый текст или падает заданной ошибкой."""

    def __init__(self, text: str = ANSWER_TEXT, error: Exception | None = None) -> None:
        self.text = text
        self.error = error
        self.calls = 0

    def complete(self, *, system, user, model, cache_system=False, max_tokens=4000):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return LLMResponse(
            text=self.text,
            input_tokens=100,
            output_tokens=20,
            latency_ms=5,
            backend="cli",
            model=model,
            raw_cost_usd=None,
            billed_cost_usd=0.0012,
        )


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
        db.upgrade(database_url)
        connection = psycopg.connect(database_url)
    except psycopg.OperationalError as exc:
        pytest.skip(f"нужен живой Postgres — docker compose up -d db && ev db upgrade: {exc}")
        return

    sources = _load_jsonl(FIXTURES / "sources.sample.jsonl", Source)
    chunks = _load_jsonl(FIXTURES / "chunks.sample.jsonl", Chunk)
    db.upsert_sources(connection, sources, corpus_version=CORPUS_VERSION)
    db.upsert_chunks(connection, chunks, corpus_version=CORPUS_VERSION)

    yield connection

    with connection.cursor() as cur:
        cur.execute("DELETE FROM query_log WHERE question LIKE %s", (f"{QUESTION_PREFIX}%",))
        cur.execute("DELETE FROM sources WHERE corpus_version = %s", (CORPUS_VERSION,))
    connection.commit()
    connection.close()


def _client(conn: psycopg.Connection, settings: Settings) -> TestClient:
    """Приложение на готовом соединении и заданных настройках.

    raise_server_exceptions=False: неожиданная ошибка должна дойти до клиента ответом
    §8, а не всплыть исключением мимо HTTP-слоя — именно это и проверяется.
    """
    app = create_app(settings)
    app.dependency_overrides[conn_dep] = lambda: conn
    app.dependency_overrides[settings_dep] = lambda: settings
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def settings() -> Settings:
    return get_settings().model_copy(update={"profile": "public", "top_k": 5})


@pytest.fixture
def client(conn, settings) -> TestClient:
    return _client(conn, settings)


@pytest.fixture
def provider(monkeypatch) -> FakeProvider:
    fake = FakeProvider()
    monkeypatch.setattr(answering, "get_provider", lambda: fake)
    return fake


def _log_row(conn: psycopg.Connection, query_id: str) -> dict:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT * FROM query_log WHERE id = %s", (query_id,))
        row = cur.fetchone()
    assert row is not None, f"записи журнала {query_id} нет"
    return row


def _chunk_url(chunk_id: str) -> str:
    """chunk_id содержит `#` — в URL он обязан быть закодирован, иначе это фрагмент,
    а не часть пути. Фронтенд кодирует id тем же способом (encodeURIComponent)."""
    return f"/api/chunks/{quote(chunk_id, safe='')}"


def _ask(client: TestClient, question: str, **extra):
    return client.post("/api/ask", json={"question": question, "retrieval_mode": "fts", **extra})


# --- ответ и журнал ---------------------------------------------------------------------


def test_ask_отдаёт_ответ_с_query_id_и_пишет_журнал(client, conn, provider):
    response = _ask(client, ALPHA_QUERY)
    assert response.status_code == 200
    payload = response.json()

    assert payload["refused"] is False
    assert payload["citations"], "ответ без цитат не должен выходить наружу"
    query_id = payload["meta"]["query_id"]

    row = _log_row(conn, query_id)
    assert row["question"] == ALPHA_QUERY
    assert row["profile"] == "public"
    assert row["retrieval_mode"] == "fts"
    assert row["error"] is None
    assert row["answer"]["answer_markdown"] == payload["answer_markdown"]
    assert row["retrieved"], "в журнале должна остаться выдача поиска"
    assert {"chunk_id", "score", "rank"} == set(row["retrieved"][0])


def test_ask_отказ_это_ответ_а_не_ошибка(client, conn, monkeypatch):
    monkeypatch.setattr(answering, "get_provider", lambda: FakeProvider(text=REFUSAL_TEXT))

    response = _ask(client, ALPHA_QUERY)
    assert response.status_code == 200
    payload = response.json()

    assert payload["refused"] is True
    assert [lim["kind"] for lim in payload["limitations"]] == ["no_data"]
    assert _log_row(conn, payload["meta"]["query_id"])["error"] is None


def test_ошибка_генерации_даёт_upstream_и_запись_в_журнал(client, conn, monkeypatch):
    monkeypatch.setattr(
        answering, "get_provider", lambda: FakeProvider(error=LLMError("claude CLI недоступен"))
    )

    response = _ask(client, f"{QUESTION_PREFIX} вопрос со сбоем генерации")
    assert response.status_code == 502
    body = response.json()
    assert body["error"]["code"] == "upstream"
    assert "Traceback" not in response.text and "claude CLI недоступен" not in response.text

    with conn.cursor() as cur:
        cur.execute(
            "SELECT answer, error, retrieved FROM query_log WHERE question LIKE %s",
            (f"{QUESTION_PREFIX} вопрос со сбоем%",),
        )
        answer, error, retrieved = cur.fetchone()
    assert answer is None
    assert "LLMError" in error
    assert retrieved is not None


def test_невалидный_ответ_модели_не_выпускается(client, conn, monkeypatch):
    # Ссылка на несуществующий фрагмент: наружу ошибка, а не «ответ» с битой ссылкой.
    broken = json.dumps({"answer_markdown": "Ответ [99].", "cited": [99], "refused": False})
    monkeypatch.setattr(answering, "get_provider", lambda: FakeProvider(text=broken))

    response = _ask(client, f"{QUESTION_PREFIX} вопрос с битой ссылкой")
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "upstream"

    with conn.cursor() as cur:
        cur.execute(
            "SELECT error FROM query_log WHERE question LIKE %s",
            (f"{QUESTION_PREFIX} вопрос с битой%",),
        )
        assert "AnswerValidationError" in cur.fetchone()[0]


@pytest.mark.parametrize("question", ["", "   ", "\n"])
def test_пустой_вопрос_это_bad_request(client, question):
    response = _ask(client, question)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "bad_request"


def test_некорректное_тело_запроса_даёт_bad_request_а_не_422(client):
    response = client.post("/api/ask", json={"question": "вопрос", "retrieval_mode": "magic"})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "bad_request"


def test_ошибки_фреймворка_тоже_в_формате_контракта(client):
    """Неизвестный адрес, чужой метод и нечитаемое тело — тоже ошибки §8, а не `detail`."""
    unknown = client.get("/api/нет-такого")
    assert unknown.status_code == 404
    assert unknown.json()["error"]["code"] == "not_found"

    wrong_method = client.get("/api/ask")
    assert wrong_method.status_code == 400
    assert wrong_method.json()["error"]["code"] == "bad_request"

    broken_body = client.post(
        "/api/ask", content="{битый json".encode(), headers={"Content-Type": "application/json"}
    )
    assert broken_body.status_code == 400
    assert set(broken_body.json()) == {"error"}


def test_сбой_базы_выходит_как_internal_без_подробностей(client, monkeypatch):
    def boom(*args, **kwargs):
        raise psycopg.OperationalError("connection to server lost: host=10.0.0.1 user=ergovault")

    monkeypatch.setattr(api, "search", boom)
    response = _ask(client, f"{QUESTION_PREFIX} вопрос при упавшей базе")

    assert response.status_code == 500
    assert response.json()["error"] == {
        "code": "internal",
        "message": "Поиск по материалам недоступен.",
    }
    assert "10.0.0.1" not in response.text


# --- профиль и закрытые источники --------------------------------------------------------


def test_публичный_инстанс_отклоняет_параметр_profile(client, provider):
    assert _ask(client, ALPHA_QUERY, profile="pilot").status_code == 400
    assert _ask(client, ALPHA_QUERY, profile="public").status_code == 400
    assert client.get("/api/sources", params={"profile": "pilot"}).status_code == 400
    assert client.get(_chunk_url(PILOT_CHUNK), params={"profile": "pilot"}).status_code == 400


def test_закрытые_источники_недоступны_публично(client, conn, provider):
    catalog = client.get("/api/sources").json()
    assert all(source["visibility"] == "public" for source in catalog)
    assert PILOT_SOURCE not in {source["id"] for source in catalog}

    assert client.get(_chunk_url(PILOT_CHUNK)).status_code == 404

    payload = _ask(client, PILOT_QUERY).json()
    cited = {citation["source_id"] for citation in payload["citations"]}
    assert PILOT_SOURCE not in cited
    retrieved = _log_row(conn, payload["meta"]["query_id"])["retrieved"]
    assert all(not item["chunk_id"].startswith(PILOT_SOURCE) for item in retrieved)


def test_закрытый_профиль_видит_свои_источники(conn, settings):
    pilot = settings.model_copy(update={"profile": "pilot"})
    client = _client(conn, pilot)

    catalog = client.get("/api/sources").json()
    assert PILOT_SOURCE in {source["id"] for source in catalog}
    assert client.get(_chunk_url(PILOT_CHUNK)).status_code == 200
    # Закрытый инстанс параметр принимает: у него есть оба профиля.
    assert client.get("/api/sources", params={"profile": "public"}).status_code == 200


# --- каталог, фрагмент, отзыв ------------------------------------------------------------


def test_каталог_отдаёт_метаданные_source(client):
    catalog = client.get("/api/sources").json()
    alpha = next(source for source in catalog if source["id"] == "fixture--company-alpha")

    assert alpha["type"] == "company"
    assert alpha["language"] == "ru"
    assert alpha["url"] == "https://alpha-expo.example"
    assert "published" in alpha and "speaker_org" in alpha and "reliability_note" in alpha


def test_фрагмент_отдаётся_вместе_с_источником(client):
    response = client.get(_chunk_url(PUBLIC_CHUNK))
    assert response.status_code == 200
    body = response.json()

    assert body["chunk"]["id"] == PUBLIC_CHUNK
    assert body["source"]["id"] == "fixture--company-alpha"
    assert body["source"]["url"] == "https://alpha-expo.example"
    assert client.get(_chunk_url("нет-такого#001")).status_code == 404


def test_отзыв_сохраняется_по_query_id(client, conn, provider):
    query_id = _ask(client, ALPHA_QUERY).json()["meta"]["query_id"]

    response = client.post(
        "/api/feedback",
        json={"query_id": query_id, "verdict": "wrong", "comment": "Цитата не о том"},
    )
    assert response.status_code == 204

    row = _log_row(conn, query_id)
    assert row["feedback"] == "wrong"
    assert row["feedback_comment"] == "Цитата не о том"


def test_отзыв_без_комментария_и_на_неизвестный_запрос(client, conn, provider):
    query_id = _ask(client, ALPHA_QUERY).json()["meta"]["query_id"]
    assert client.post("/api/feedback", json={"query_id": query_id, "verdict": "useful"}).status_code == 204
    assert _log_row(conn, query_id)["feedback_comment"] is None

    unknown = client.post(
        "/api/feedback",
        json={"query_id": "11111111-1111-4111-8111-111111111111", "verdict": "useful"},
    )
    assert unknown.status_code == 404
    assert unknown.json()["error"]["code"] == "not_found"

    assert client.post("/api/feedback", json={"query_id": "не-uuid", "verdict": "useful"}).status_code == 400
    assert client.post("/api/feedback", json={"query_id": "11111111-1111-4111-8111-111111111111", "verdict": "ok"}).status_code == 400


# --- качество и состояние ----------------------------------------------------------------


def test_summary_без_прогонов_говорит_об_этом_явно(client, monkeypatch, tmp_path):
    monkeypatch.setattr(api, "EVAL_SUMMARY_PATH", tmp_path / "нет-файла.json")
    body = client.get("/api/eval/summary").json()

    assert body["available"] is False
    assert body["runs"] == []
    assert "не было" in body["message"]
    # Отсутствие измерений не превращается в нули, а оговорка о выборке есть всегда.
    assert "n≈35" in body["sample_size_caveat"]


def test_summary_с_прогонами(client, monkeypatch, tmp_path):
    path = tmp_path / "summary.json"
    path.write_text(
        json.dumps(
            {
                "message": "Прогон 12.09.2026",
                "generated_at": "2026-09-12T10:00:00Z",
                "corpus_version": "v1",
                "prompt_version": "p1",
                "runs": [
                    {
                        "label": "hybrid",
                        "mode": "hybrid",
                        "k": 5,
                        "n_questions": 33,
                        "recall_at_k": 0.82,
                        "mrr": 0.7,
                        "refusal_precision": None,
                        "refusal_recall": None,
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(api, "EVAL_SUMMARY_PATH", path)

    body = client.get("/api/eval/summary").json()
    assert body["available"] is True
    assert body["runs"][0]["recall_at_k"] == 0.82
    # Неизмеренное остаётся неизмеренным, а не нулём.
    assert body["runs"][0]["refusal_precision"] is None
    # Полей T-30 в старом файле нет — они приходят пустыми, а не выдуманными.
    assert body["judge"] is None
    assert body["limitations"] == []
    assert body["runs"][0]["grounded_rate"] is None


def test_summary_отдаёт_сравнение_вариантов_и_ограничения(client, monkeypatch, tmp_path):
    """Файл T-30: варианты вместо режимов, состояние судьи и оговорки рядом с числами."""
    path = tmp_path / "summary.json"
    path.write_text(
        json.dumps(
            {
                "message": "Сравнение четырёх подходов на 33 вопросах.",
                "generated_at": "20260912T100000Z",
                "corpus_version": "v1",
                "prompt_version": "p1",
                "runs": [
                    {
                        "label": "full_corpus",
                        "title": "весь корпус в контексте",
                        "mode": "full_corpus",
                        "k": 8,
                        "n_questions": 33,
                        "recall_at_k": None,
                        "mrr": None,
                        "recall_note": "неприменимо: в контекст передан весь корпус",
                        "must_contain_rate": 0.61,
                        "grounded_rate": None,
                        "median_latency_ms": 21000.0,
                        "cost_usd_per_question": 1.52,
                    }
                ],
                "judge": {
                    "available": False,
                    "message": "Ручной разметки 0 пар из требуемых 40.",
                    "n_labeled": 0,
                    "agreement": None,
                    "cohen_kappa": None,
                    "self_consistency": 0.9,
                },
                "limitations": ["Разброс между прогонами не измерен."],
                "report_path": "eval/reports/final-comparison.md",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(api, "EVAL_SUMMARY_PATH", path)

    body = client.get("/api/eval/summary").json()
    run = body["runs"][0]
    # Неприменимая метрика приходит None и объясняется словами, а не нулём.
    assert run["recall_at_k"] is None
    assert "неприменимо" in run["recall_note"]
    # Подтверждённость не публикуется, пока судья не сверен с человеком.
    assert body["judge"]["available"] is False
    assert run["grounded_rate"] is None
    assert body["limitations"]
    assert body["report_path"].endswith("final-comparison.md")


def test_healthz_считает_без_тяжёлой_обработки(client, monkeypatch):
    from ergovault import embeddings

    def fail(*args, **kwargs):
        raise AssertionError("healthz не должен грузить модель эмбеддингов")

    monkeypatch.setattr(embeddings, "get_model", fail)

    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["corpus_version"] == get_settings().corpus_version
    assert body["sources"] > 0 and body["chunks"] > 0
