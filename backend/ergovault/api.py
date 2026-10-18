"""HTTP-слой (CONTRACTS.md §8): разбор запроса, вызов поиска и ответа, журнал, ошибки.

Прикладной логики здесь нет. Поиск — `search.py`, ответ — `answering.py`, SQL и журнал —
`db.py`. Профиль решается на входе: публичный инстанс не принимает параметр `profile`
вообще, а видимость источников фильтруется в SQL обеих ветвей поиска, каталога и прямой
ссылки на чанк. Наружу выходят только коды ошибок §8 — без traceback и текста исключения.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Annotated, Any, Literal

import psycopg
from fastapi import APIRouter, Depends, FastAPI, Path, Query, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from ergovault import answering, db
from ergovault.answering import AnswerValidationError
from ergovault.eval.metrics import SAMPLE_SIZE_CAVEAT
from ergovault.llm import LLMError
from ergovault.models import AnswerPayload, Chunk, Profile, Source
from ergovault.search import RetrievalMode, search
from ergovault.settings import REPO_ROOT, Settings, get_settings

log = logging.getLogger("ergovault.api")

MAX_QUESTION_CHARS = 1000
MAX_COMMENT_CHARS = 2000
EVAL_SUMMARY_PATH = REPO_ROOT / "eval" / "reports" / "summary.json"
NO_RUNS_MESSAGE = "Прогонов проверки качества ещё не было."
HAS_RUNS_MESSAGE = "Результаты последнего прогона проверочного набора."

ErrorCode = Literal["bad_request", "not_found", "upstream", "internal"]
_STATUS_BY_CODE: dict[ErrorCode, int] = {
    "bad_request": 400,
    "not_found": 404,
    "upstream": 502,
    "internal": 500,
}


_HTTP_ERRORS: dict[int, tuple[ErrorCode, str]] = {
    400: ("bad_request", "Некорректный запрос."),
    404: ("not_found", "Такого адреса нет."),
    405: ("bad_request", "Метод не поддерживается этим адресом."),
    422: ("bad_request", "Некорректный запрос."),
}


class ApiError(Exception):
    """Ошибка, которую можно показать пользователю: код §8 и понятный текст.

    Всё остальное (исключения БД, неожиданные сбои) наружу уходит как `internal` с общим
    сообщением: подробности пишутся в лог сервера, а не в тело ответа.
    """

    def __init__(self, code: ErrorCode, message: str) -> None:
        super().__init__(message)
        self.code: ErrorCode = code
        self.message = message
        self.status = _STATUS_BY_CODE[code]


def _error_response(code: ErrorCode, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=_STATUS_BY_CODE[code], content={"error": {"code": code, "message": message}}
    )


# --- схемы запросов и ответов (зеркало frontend/src/api/types.ts) ----------------------


class AskRequest(BaseModel):
    question: str
    profile: Profile | None = None
    retrieval_mode: RetrievalMode | None = None


class ChunkView(BaseModel):
    chunk: Chunk
    source: Source


class FeedbackRequest(BaseModel):
    query_id: str
    verdict: Literal["useful", "wrong"]
    comment: str | None = None


class Health(BaseModel):
    status: str
    corpus_version: str
    chunks: int
    sources: int


class EvalRunSummary(BaseModel):
    """Один вариант сравнения. Метрики — доли 0..1; None означает «не измерено»
    (например, recall неприменим там, где поиска не было), и это не ноль."""

    label: str
    mode: str
    k: int
    n_questions: int
    title: str = ""  # человеческое название варианта
    recall_at_k: float | None = None
    mrr: float | None = None
    recall_note: str = ""  # почему метрики нет, если её нет
    refusal_precision: float | None = None
    refusal_recall: float | None = None
    must_contain_rate: float | None = None
    grounded_rate: float | None = None  # приходит только вместе с калибровкой судьи
    median_latency_ms: float | None = None
    cost_usd_per_question: float | None = None


class EvalJudgeSummary(BaseModel):
    """Проверка судьи. `available=false` — ручной разметки не хватает, и подтверждённость
    в прогонах приходит как None: без согласия с человеком её не публикуют."""

    available: bool
    message: str
    n_labeled: int = 0
    agreement: float | None = None
    cohen_kappa: float | None = None
    self_consistency: float | None = None


class EvalSummary(BaseModel):
    """Состояние экрана «Проверка качества»: либо прогонов нет, либо они есть.

    Форму задаёт T-22, наполняет T-30 (файл `eval/reports/summary.json`). Оговорка о размере
    выборки приходит с данными всегда: без неё разница между режимами читается как результат.
    Ограничения прогона приходят тем же ответом — экран показывает их рядом с числами,
    а не прячет в отчёт.
    """

    available: bool
    message: str
    generated_at: str | None = None
    corpus_version: str | None = None
    prompt_version: str | None = None
    sample_size_caveat: str = SAMPLE_SIZE_CAVEAT
    runs: list[EvalRunSummary] = Field(default_factory=list)
    judge: EvalJudgeSummary | None = None
    limitations: list[str] = Field(default_factory=list)
    report_path: str | None = None


# --- зависимости ----------------------------------------------------------------------


def settings_dep() -> Settings:
    return get_settings()


@contextmanager
def _connect(settings: Settings) -> Iterator[psycopg.Connection]:
    try:
        conn = psycopg.connect(settings.database_url)
    except psycopg.OperationalError as exc:
        log.error("нет соединения с базой: %s", exc)
        raise ApiError("internal", "База данных недоступна.") from exc
    try:
        yield conn
    finally:
        conn.close()


def conn_dep(settings: Annotated[Settings, Depends(settings_dep)]) -> Iterator[psycopg.Connection]:
    """Соединение на запрос. Пул не заводится: запрос к модели длится секунды, а
    подключение — миллисекунды, и лишняя инфраструктура здесь ничего не ускорит."""
    with _connect(settings) as conn:
        yield conn


def resolve_profile(requested: Profile | None, settings: Settings) -> Profile:
    """Профиль запроса. Публичный инстанс параметр `profile` не принимает вообще:
    иначе закрытый профиль оказывался бы в одном запросе от публичного адреса."""
    if requested is None:
        return settings.profile
    if settings.profile == "public":
        raise ApiError(
            "bad_request", "Этот инстанс публичный: параметр profile не принимается."
        )
    return requested


# --- эндпоинты ------------------------------------------------------------------------

router = APIRouter()

SettingsDep = Annotated[Settings, Depends(settings_dep)]
ConnDep = Annotated[psycopg.Connection, Depends(conn_dep)]


@router.post("/api/ask", response_model=AnswerPayload)
def ask(request: AskRequest, conn: ConnDep, settings: SettingsDep) -> AnswerPayload:
    """Вопрос → поиск → ответ с проверенными цитатами. Запись журнала — в любом исходе.

    Успех и сбой генерации пишутся в `query_log` одинаково подробно: без записи о сбое
    журнал показывал бы качество выше фактического. ID записи возвращается как
    `meta.query_id` — по нему сохраняется отзыв.
    """
    question = request.question.strip()
    if not question:
        raise ApiError("bad_request", "Вопрос не может быть пустым.")
    if len(question) > MAX_QUESTION_CHARS:
        raise ApiError("bad_request", f"Вопрос длиннее {MAX_QUESTION_CHARS} символов.")

    profile = resolve_profile(request.profile, settings)
    mode = request.retrieval_mode or settings.retrieval_mode
    started = time.monotonic()

    try:
        hits = search(conn, question, settings.top_k, profile=profile, mode=mode, settings=settings)
        sources = {source.id: source for source in db.list_sources(conn, profile)}
    except psycopg.Error as exc:
        log.exception("поиск не выполнен")
        raise ApiError("internal", "Поиск по материалам недоступен.") from exc

    retrieved = [
        {"chunk_id": hit.chunk.id, "score": hit.score, "rank": hit.rank} for hit in hits
    ]
    log_common = {
        "question": question,
        "profile": profile,
        "retrieval_mode": mode,
        "corpus_version": settings.corpus_version,
        "prompt_version": settings.prompt_version,
        "retrieved": retrieved,
    }

    try:
        payload = answering.answer_hits(
            question, hits, sources=sources, retrieval_mode=mode, settings=settings
        )
    except (LLMError, AnswerValidationError) as exc:
        meta = getattr(exc, "meta", {}) or {}
        db.insert_query_log(
            conn,
            **log_common,
            model=meta.get("model") or settings.gen_model,
            latency_ms=int((time.monotonic() - started) * 1000),
            cost_usd=meta.get("cost_usd"),
            error=f"{type(exc).__name__}: {exc}",
        )
        log.warning("генерация не удалась: %s", exc)
        raise ApiError(
            "upstream",
            "Сервис генерации ответа временно недоступен. Поиск по материалам при этом "
            "работает — повторите вопрос через минуту.",
        ) from exc

    query_id = db.insert_query_log(
        conn,
        **log_common,
        model=payload.meta.get("model") or settings.gen_model,
        answer=payload.model_dump(mode="json"),
        latency_ms=payload.meta.get("latency_ms"),
        cost_usd=payload.meta.get("cost_usd"),
    )
    payload.meta["query_id"] = query_id
    return payload


@router.get("/api/sources", response_model=list[Source])
def get_sources(
    conn: ConnDep,
    settings: SettingsDep,
    profile: Annotated[Profile | None, Query()] = None,
) -> list[Source]:
    """Каталог материалов профиля: язык, дата, тип и прочие метаданные `Source`.

    Библиотека и вопросы о свойствах корпуса отвечаются по этим метаданным, а не по
    тексту чанков: тип и дата в чанке не хранятся.
    """
    return db.list_sources(conn, resolve_profile(profile, settings))


@router.get("/api/chunks/{chunk_id}", response_model=ChunkView)
def get_chunk(
    conn: ConnDep,
    settings: SettingsDep,
    chunk_id: Annotated[str, Path()],
    profile: Annotated[Profile | None, Query()] = None,
) -> ChunkView:
    """Фрагмент с его источником. Закрытый источник по прямой ссылке не открывается:
    фильтр видимости стоит в SQL, а не в проверке после выборки."""
    found = db.get_chunk_view(conn, chunk_id, resolve_profile(profile, settings))
    if found is None:
        raise ApiError("not_found", f"Фрагмент {chunk_id} не найден.")
    chunk, source = found
    return ChunkView(chunk=chunk, source=source)


@router.post("/api/feedback", status_code=204)
def post_feedback(request: FeedbackRequest, conn: ConnDep) -> Response:
    """Отзыв на конкретный ответ. `query_id` — id записи журнала из `meta.query_id`."""
    try:
        uuid.UUID(request.query_id)
    except ValueError as exc:
        raise ApiError("bad_request", "query_id должен быть UUID записи журнала.") from exc

    comment = (request.comment or "").strip() or None
    if comment and len(comment) > MAX_COMMENT_CHARS:
        raise ApiError("bad_request", f"Комментарий длиннее {MAX_COMMENT_CHARS} символов.")

    if not db.set_feedback(conn, request.query_id, request.verdict, comment):
        raise ApiError("not_found", "Запрос с таким query_id не найден.")
    return Response(status_code=204)


@router.get("/api/eval/summary", response_model=EvalSummary)
def eval_summary() -> EvalSummary:
    """Результаты последнего прогона проверочного набора.

    Файла нет — `available=false` и явное «прогонов не было»: отсутствие измерений
    не показывается как нулевая ошибка или пустая таблица без объяснения.
    """
    if not EVAL_SUMMARY_PATH.is_file():
        return EvalSummary(available=False, message=NO_RUNS_MESSAGE)
    try:
        raw: Any = json.loads(EVAL_SUMMARY_PATH.read_text(encoding="utf-8"))
        summary = EvalSummary.model_validate({**raw, "available": True})
    except (OSError, ValueError) as exc:
        log.exception("не прочитан %s", EVAL_SUMMARY_PATH)
        raise ApiError("internal", "Результаты проверки качества не читаются.") from exc
    if not summary.message:
        summary.message = HAS_RUNS_MESSAGE
    return summary


@router.get("/healthz", response_model=Health)
def healthz(conn: ConnDep, settings: SettingsDep) -> Health:
    """Состояние и счётчики. Только два COUNT: ни модели, ни поиска здесь не запускается."""
    sources, chunks = db.corpus_counts(conn, settings.profile)
    return Health(
        status="ok" if sources and chunks else "empty",
        corpus_version=settings.corpus_version,
        chunks=chunks,
        sources=sources,
    )



# --- приложение -----------------------------------------------------------------------


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(
        title="ErgoVault RAG",
        version="0.1.0",
        summary="Ответы по корпусу исследования международных выставок",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
    )

    @app.exception_handler(ApiError)
    async def _api_error(_request, exc: ApiError) -> JSONResponse:
        return _error_response(exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0]
        where = ".".join(str(part) for part in first["loc"][1:]) or "запрос"
        return _error_response("bad_request", f"Некорректный запрос ({where}: {first['msg']}).")

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_request, exc: StarletteHTTPException) -> JSONResponse:
        # Ответы самого фреймворка (неизвестный путь, метод, нечитаемое тело) тоже обязаны
        # быть в формате §8. Английский `detail` наружу не идёт: сообщение своё.
        code, message = _HTTP_ERRORS.get(
            exc.status_code, ("internal", "Внутренняя ошибка сервиса.")
        )
        return _error_response(code, message)

    @app.exception_handler(Exception)
    async def _unexpected(_request, exc: Exception) -> JSONResponse:
        # Traceback остаётся в логе сервера: наружу уходит код §8 и общее сообщение.
        log.exception("необработанная ошибка")
        return _error_response("internal", "Внутренняя ошибка сервиса.")

    app.include_router(router)
    return app


app = create_app()
