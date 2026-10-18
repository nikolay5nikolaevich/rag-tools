"""Ответ на вопрос по найденным фрагментам (CONTRACTS.md §7).

Модель пишет только текст ответа и номера фрагментов. Цитаты, URL и ограничения
собирает код из переданных чанков и метаданных источников: модель не источник фактов
о корпусе. Ответ, не прошедший проверку, не выпускается — это ошибка, а не отказ.
"""

from __future__ import annotations

import hashlib
import re
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from string import Template

import psycopg
from pydantic import BaseModel, ConfigDict, ValidationError

from ergovault import db
from ergovault.llm import LLMProvider, get_provider
from ergovault.models import (
    MAX_QUOTE_CHARS,
    AnswerPayload,
    Citation,
    Hit,
    Limitation,
    Profile,
    Source,
)
from ergovault.search import RetrievalMode, search
from ergovault.settings import Settings, get_settings

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
MAX_OUTPUT_TOKENS = 4000
PROVIDER_STANCE = "организатор-провайдер"
NOT_SPECIFIED = "не указана"
# Язык, дата и длительность — метаданные источника, а не атрибуция: на вопросы о составе
# корпуса (категория F проверочного набора) отвечают именно они, а не текст заметок.
LANGUAGE_NAMES = {
    "ru": "русский",
    "en": "английский",
    "fr": "французский",
    "mixed": "смешанный",
    "unknown": "не определён",
}

# Тексты ограничений фиксированы по kind. Подставляются только организация, название
# источника, извлечённая из корпуса оговорка или описание недостающих данных.
LIMITATION_TEXTS = {
    "no_data": "Данных для ответа в корпусе не нашлось: {missing}.",
    "single_org": (
        "Все процитированные материалы принадлежат одной организации — {org}. "
        "Это одно подтверждение, а не несколько независимых."
    ),
    "provider_stance": (
        "Среди процитированных — материалы организатора участия в выставках ({orgs}). "
        "Это заинтересованная сторона: её оценки и рекомендации не нейтральны."
    ),
    "derived_only": ("Все процитированные материалы — пересказы видео, а не сами первоисточники."),
    "reliability_note": "Оговорка к источнику «{title}»: {note}",
}
MISSING_IN_FRAGMENTS = "в найденных фрагментах нет запрошенных сведений"
MISSING_HITS = "поиск не вернул ни одного фрагмента"
EMPTY_SEARCH_ANSWER = (
    "По этому вопросу поиск не нашёл в корпусе ни одного фрагмента, поэтому ответить "
    "по материалам исследования нельзя."
)


class AnswerValidationError(RuntimeError):
    """Ответ модели не прошёл проверку и не выпускается. Техническая ошибка, не отказ.

    `raw_text` и `meta` заполняются для журнала: неудачный вызов тоже стоит денег и времени.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(f"ответ модели отклонён проверкой: {reason}")
        self.reason = reason
        self.raw_text: str | None = None
        self.meta: dict = {}


# --- промпт -----------------------------------------------------------------------------


def load_prompt(version: str) -> tuple[str, Template, str]:
    """(system, шаблон user, короткий хэш обоих файлов). Хэш ловит правку промпта без
    смены версии: такой прогон нельзя сравнивать с прежними (CONTRACTS.md §10)."""
    paths = [PROMPTS_DIR / f"{version}.system.md", PROMPTS_DIR / f"{version}.user.md"]
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"нет файлов промпта версии {version!r}: {', '.join(missing)}")
    system, user = (path.read_text(encoding="utf-8") for path in paths)
    digest = hashlib.sha256(f"{system}\0{user}".encode()).hexdigest()[:12]
    return system, Template(user), digest


def render_fragments(hits: Sequence[Hit]) -> str:
    """Нумерованные фрагменты: источник, тип, язык, дата, длительность, заголовки и текст.

    Язык, дата и длительность нужны для вопросов о самих материалах: они лежат в полях
    `Source`, а не в тексте чанка, и без них ответ про язык или дату пришлось бы угадывать
    по содержимому. Пустое значение так и передаётся — «не указана»: заглушка корпуса
    (контракт §2) не заменяется правдоподобным числом.

    URL, speaker_org, stance, reliability_note и derived сюда по-прежнему не попадают:
    это данные для кода, а не материал и не инструкции для модели.
    """
    blocks = []
    for n, hit in enumerate(hits, start=1):
        source = hit.source
        headings = " › ".join("вступление" if h == "_intro" else h for h in hit.chunk.heading_path)
        duration = f"{source.duration_minutes} мин" if source.duration_minutes else NOT_SPECIFIED
        blocks.append(
            f'<fragment n="{n}">\n'
            f"Источник: {source.title}\n"
            f"Тип: {source.type}\n"
            f"Язык: {LANGUAGE_NAMES.get(source.language, source.language)}\n"
            f"Дата публикации: {source.published or NOT_SPECIFIED}\n"
            f"Длительность: {duration}\n"
            f"Раздел: {headings}\n"
            f"Текст:\n{hit.chunk.text}\n"
            "</fragment>"
        )
    return "\n\n".join(blocks)


# --- проверка ответа модели ---------------------------------------------------------------


class _ModelAnswer(BaseModel):
    model_config = ConfigDict(strict=True)

    answer_markdown: str
    cited: list[int]
    refused: bool


_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
# Ссылка [n] и границы предложений — общее у ответа, вариантов сравнения и судьи
# (eval/variants.py, eval/judge.py), поэтому имена публичные.
REF = re.compile(r"\[(\d+)\]")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+|\n")
_REF_GROUP = re.compile(r"\[(\d+(?:\s*,\s*\d+)+)\]")
_URL = re.compile(r"https?://|www\.", re.IGNORECASE)
# Цитата в кавычках, за которой сразу идут ссылки: «…» [2] или «…».[1][3]
_QUOTED = re.compile(r"(«[^«»]+»|“[^“”\n]+”|\"[^\"\n]+\")[.,;:]?\s*((?:\[\d+\]\s*)+)")
_ELLIPSIS = re.compile(r"…|\.\.\.")


def strip_code_fence(text: str) -> str:
    """Тело JSON без обёртки ```json … ```: модель просят её не ставить, но она бывает."""
    body = text.strip()
    fenced = _FENCE.fullmatch(body)
    return fenced.group(1) if fenced else body


def parse_model_output(text: str) -> _ModelAnswer:
    body = strip_code_fence(text)
    try:
        return _ModelAnswer.model_validate_json(body)
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(part) for part in first["loc"]) or "ответ"
        raise AnswerValidationError(f"не JSON по схеме p1 ({where}: {first['msg']})") from exc


def _split_ref_groups(text: str) -> str:
    """[1, 3] → [1][3]: фронтенд и проверка работают с одиночными [n]."""
    return _REF_GROUP.sub(lambda m: "".join(f"[{n.strip()}]" for n in m.group(1).split(",")), text)


def normalize_for_match(text: str) -> str:
    """Для сравнения цитат не важны регистр, ё/е, markdown-выделение, вид тире и пробелы."""
    text = text.casefold().replace("ё", "е")
    text = re.sub(r"[*_`]", "", text)
    text = re.sub(r"[‐‑‒–—−]", "-", text)
    return " ".join(text.split())


def _check_quotes(text: str, hits: Sequence[Hit]) -> None:
    """Цитата со ссылкой обязана дословно найтись в тех фрагментах, на которые ссылается.
    Многоточие внутри цитаты допускает пропуск: каждая часть проверяется отдельно."""
    for match in _QUOTED.finditer(text):
        quote = match.group(1)[1:-1]
        numbers = [int(n) for n in REF.findall(match.group(2))]
        places = [
            normalize_for_match(place)
            for n in numbers
            for place in (
                hits[n - 1].chunk.text,
                hits[n - 1].source.title,
                *hits[n - 1].chunk.heading_path,
            )
        ]
        for piece in _ELLIPSIS.split(quote):
            piece = normalize_for_match(piece).strip(" .,;:!?")
            if piece and not any(piece in place for place in places):
                raise AnswerValidationError(
                    f"цитата «{quote[:80]}» не найдена во фрагментах {numbers}"
                )


# --- цитаты ---------------------------------------------------------------------------------

# Закрывающие ** после точки («…делам.** Дальше») принадлежат предыдущему предложению:
# иначе цитата начиналась бы с «** ».
_SENTENCE = re.compile(r"[^\n.!?…]+[.!?…]*\**")
_BREAK = re.compile(r"[.!?…](?=\s)|\n")


def _stems(text: str) -> set[str]:
    """Грубые основы слов (первые 5 букв): хватает, чтобы найти предложение, о котором
    говорит ответ, без морфологического словаря."""
    return {w[:5] for w in re.findall(r"\w+", normalize_for_match(text)) if len(w) >= 3 or w.isdigit()}


def _context(text: str, n: int) -> str:
    """Предложения ответа со ссылкой [n] — по ним выбирается цитата из фрагмента n."""
    marker = f"[{n}]"
    parts = SENTENCE_SPLIT.split(text)
    return REF.sub(" ", " ".join(part for part in parts if marker in part))


def select_quote(chunk_text: str, context: str) -> str:
    """Дословная подстрока чанка не длиннее MAX_QUOTE_CHARS. Начинается с предложения,
    сильнее всего пересекающегося с текстом ответа у ссылки; без пересечений — с начала."""
    wanted = _stems(context)
    start, best = 0, 0
    for match in _SENTENCE.finditer(chunk_text):
        score = len(wanted & _stems(match.group()))
        if score > best:
            start, best = match.start(), score
    tail = chunk_text[start:].lstrip()
    if len(tail) <= MAX_QUOTE_CHARS:
        return tail.rstrip()
    window = tail[:MAX_QUOTE_CHARS]
    min_len = MAX_QUOTE_CHARS // 4
    breaks = [m.end() for m in _BREAK.finditer(window) if m.end() >= min_len]
    space = window.rfind(" ")
    cut = breaks[-1] if breaks else (space if space >= min_len else MAX_QUOTE_CHARS)
    return window[:cut].rstrip()


# --- ограничения ------------------------------------------------------------------------------


def _origin(source: Source, sources: Mapping[str, Source]) -> Source:
    """Чья атрибуция действует: у пересказа — оригинала из derived_of, иначе своя."""
    if source.derived and source.derived_of:
        return sources.get(source.derived_of, source)
    return source


def _org(source: Source, sources: Mapping[str, Source]) -> str | None:
    return _origin(source, sources).speaker_org or source.speaker_org


def compute_limitations(
    cited: Sequence[Source],
    *,
    refused: bool,
    sources: Mapping[str, Source],
    missing: str = MISSING_IN_FRAGMENTS,
) -> list[Limitation]:
    """Ограничения из метаданных процитированных источников (CONTRACTS.md §7).

    Пересказ получает организацию, позицию и оговорку оригинала: оригинал видео и его
    пересказ — одно подтверждение, а не два. Неизвестная организация не считается общей:
    single_org — только если организация известна у всех цитат и она одна.
    """
    ids = [source.id for source in cited]
    limitations: list[Limitation] = []
    if refused:
        text = LIMITATION_TEXTS["no_data"].format(missing=missing)
        limitations.append(Limitation(kind="no_data", text=text, source_ids=[]))

    orgs = {_org(source, sources) for source in cited}
    if cited and None not in orgs and len(orgs) == 1:
        text = LIMITATION_TEXTS["single_org"].format(org=orgs.pop())
        limitations.append(Limitation(kind="single_org", text=text, source_ids=ids))

    providers = [s for s in cited if PROVIDER_STANCE in (s.stance, _origin(s, sources).stance)]
    if providers:
        names = dict.fromkeys(_org(s, sources) or s.title for s in providers)
        text = LIMITATION_TEXTS["provider_stance"].format(orgs=", ".join(names))
        limitations.append(
            Limitation(kind="provider_stance", text=text, source_ids=[s.id for s in providers])
        )

    if cited and all(source.derived for source in cited):
        text = LIMITATION_TEXTS["derived_only"]
        limitations.append(Limitation(kind="derived_only", text=text, source_ids=ids))

    # Оригинал и пересказ часто несут одну и ту же оговорку: один абзац, id всех источников.
    by_note: dict[str, list[Source]] = {}
    for source in cited:
        note = source.reliability_note or _origin(source, sources).reliability_note
        if note:
            by_note.setdefault(note.strip(), []).append(source)
    for note, group in by_note.items():
        text = LIMITATION_TEXTS["reliability_note"].format(title=group[0].title, note=note)
        limitations.append(
            Limitation(kind="reliability_note", text=text, source_ids=[s.id for s in group])
        )
    return limitations


# --- сборка ответа --------------------------------------------------------------------------


def build_payload(
    model_text: str, hits: Sequence[Hit], *, sources: Mapping[str, Source]
) -> AnswerPayload:
    """Проверить ответ модели по переданным фрагментам и собрать AnswerPayload без meta."""
    parsed = parse_model_output(model_text)
    text = _split_ref_groups(parsed.answer_markdown).strip()
    if not text:
        raise AnswerValidationError("пустой answer_markdown: отказ тоже объясняет, чего нет")
    if _URL.search(text):
        raise AnswerValidationError("модель написала URL; ссылки на источники берутся из БД")

    refs = list(dict.fromkeys(int(n) for n in REF.findall(text)))
    unknown = sorted({n for n in [*refs, *parsed.cited] if not 1 <= n <= len(hits)})
    if unknown:
        raise AnswerValidationError(
            f"ссылки на несуществующие фрагменты {unknown}; передано фрагментов: {len(hits)}"
        )
    if missing := sorted(set(refs) - set(parsed.cited)):
        raise AnswerValidationError(f"номера {missing} есть в тексте, но не указаны в cited")
    if extra := sorted(set(parsed.cited) - set(refs)):
        raise AnswerValidationError(f"номера {extra} указаны в cited, но в тексте ссылок нет")
    if not refs and not parsed.refused:
        raise AnswerValidationError("ответ без единой ссылки на фрагменты")
    _check_quotes(text, hits)

    citations = []
    for n in sorted(refs):
        hit = hits[n - 1]
        citations.append(
            Citation(
                n=n,
                chunk_id=hit.chunk.id,
                source_id=hit.source.id,
                source_title=hit.source.title,
                source_type=hit.source.type,
                heading_path=list(hit.chunk.heading_path),
                quote=select_quote(hit.chunk.text, _context(text, n)),
                url=hit.source.url,
            )
        )
    cited = list({hits[c.n - 1].source.id: hits[c.n - 1].source for c in citations}.values())
    return AnswerPayload(
        answer_markdown=text,
        citations=citations,
        limitations=compute_limitations(cited, refused=parsed.refused, sources=sources),
        refused=parsed.refused,
    )


def _elapsed_ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


def answer_hits(
    question: str,
    hits: Sequence[Hit],
    *,
    sources: Mapping[str, Source],
    retrieval_mode: str,
    settings: Settings | None = None,
    provider: LLMProvider | None = None,
) -> AnswerPayload:
    """Ответ по уже найденным фрагментам — общий вход приложения и оценки.

    `sources` — каталог источников профиля (`db.list_sources`): по нему пересказ получает
    атрибуцию оригинала, даже если оригинал не попал в выдачу. Отказ из-за нехватки
    данных — AnswerPayload(refused=True); сбой модели (LLMError) и ответ, не прошедший
    проверку (AnswerValidationError), — исключения.
    """
    settings = settings or get_settings()
    started = time.monotonic()
    system, user_template, prompt_hash = load_prompt(settings.prompt_version)
    meta: dict = {
        "corpus_version": settings.corpus_version,
        "retrieval_mode": retrieval_mode,
        "prompt_version": settings.prompt_version,
        "prompt_hash": prompt_hash,
        "model": None,
    }
    if not hits:
        # Модель не вызывается: ей не на что опереться, а честный ответ известен заранее.
        meta.update(cost_usd=0.0, latency_ms=_elapsed_ms(started))
        limitations = compute_limitations([], refused=True, sources=sources, missing=MISSING_HITS)
        return AnswerPayload(
            answer_markdown=EMPTY_SEARCH_ANSWER, limitations=limitations, refused=True, meta=meta
        )

    user = user_template.substitute(
        question=question.strip(), count=len(hits), fragments=render_fragments(hits)
    )
    response = (provider or get_provider()).complete(
        system=system,
        user=user,
        model=settings.gen_model,
        cache_system=True,
        max_tokens=MAX_OUTPUT_TOKENS,
    )
    meta.update(
        # Размер нашего промпта в символах считается кодом и проверяем: usage cli-бэкенда
        # включает его собственную служебную часть и на неё же реагирует (CONTRACTS.md §9).
        prompt_chars=len(system) + len(user),
        model=response.model,
        llm_backend=response.backend,
        llm_latency_ms=response.latency_ms,
        cost_usd=response.billed_cost_usd,
        raw_cost_usd=response.raw_cost_usd,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        cache_read_tokens=response.cache_read_tokens,
        cache_write_tokens=response.cache_write_tokens,
    )
    try:
        payload = build_payload(response.text, hits, sources=sources)
    except AnswerValidationError as exc:
        exc.raw_text = response.text
        exc.meta = {**meta, "latency_ms": _elapsed_ms(started)}
        raise
    payload.meta = {**meta, "latency_ms": _elapsed_ms(started)}
    return payload


def answer(
    conn: psycopg.Connection,
    question: str,
    *,
    profile: Profile,
    mode: RetrievalMode | None = None,
    k: int | None = None,
    settings: Settings | None = None,
    provider: LLMProvider | None = None,
    encode_query: Callable[[str], Sequence[float]] | None = None,
) -> AnswerPayload:
    """Вопрос → поиск → ответ модели → проверка. Исключения — как у answer_hits."""
    if not question.strip():
        raise ValueError("пустой вопрос")
    settings = settings or get_settings()
    mode = mode or settings.retrieval_mode
    started = time.monotonic()
    hits = search(
        conn,
        question,
        k or settings.top_k,
        profile=profile,
        mode=mode,
        settings=settings,
        encode_query=encode_query,
    )
    extra = {"profile": profile, "search_latency_ms": _elapsed_ms(started)}
    sources = {source.id: source for source in db.list_sources(conn, profile)}
    try:
        payload = answer_hits(
            question,
            hits,
            sources=sources,
            retrieval_mode=mode,
            settings=settings,
            provider=provider,
        )
    except AnswerValidationError as exc:
        exc.meta.update(extra, latency_ms=_elapsed_ms(started))
        raise
    payload.meta.update(extra, latency_ms=_elapsed_ms(started))
    return payload
