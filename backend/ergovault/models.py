"""Модели данных ErgoVault RAG.

Нормативный источник — docs/CONTRACTS.md §1. Менять эти модели можно только вместе
с контрактом: на них завязаны схема БД, формат eval-разметки и типы фронтенда.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Literal

from pydantic import BaseModel, Field, field_validator

SourceType = Literal[
    "video-source",
    "video-theses",
    "topic-hub",
    "article",
    "course-review",
    "event",
    "company",
    "synthesis",
    "primary",
]
Stance = Literal["нейтральный", "организатор-провайдер", "подрядчик"]
Profile = Literal["public", "pilot"]
Language = Literal["ru", "en", "fr", "mixed", "unknown"]
Visibility = Literal["public", "pilot"]

MAX_CHUNK_CHARS = 4000
MIN_CHUNK_CHARS = 80
MIN_ATOMIC_CHARS = 40
QUOTE_ANCHOR_CHARS = 120
MAX_QUOTE_CHARS = 300


class Source(BaseModel):
    id: str
    path: str
    type: SourceType
    title: str
    url: str | None = None
    language: Language = "unknown"
    published: str | None = None
    duration_minutes: int | None = None
    speaker_org: str | None = None
    stance: Stance = "нейтральный"
    derived: bool = False
    derived_of: str | None = None
    reliability_note: str | None = None
    visibility: Visibility = "public"
    content_hash: str

    @field_validator("derived_of")
    @classmethod
    def _derived_of_not_self(cls, v: str | None, info) -> str | None:
        if v is not None and v == info.data.get("id"):
            raise ValueError("derived_of не может ссылаться на сам источник")
        return v


class Chunk(BaseModel):
    id: str
    source_id: str
    ordinal: int = Field(ge=0)
    heading_path: list[str] = Field(min_length=1)
    text: str
    char_len: int
    token_estimate: int
    atomic: bool = False
    quote_anchor: str

    @field_validator("char_len")
    @classmethod
    def _len_matches(cls, v: int, info) -> int:
        text = info.data.get("text")
        if text is not None and v != len(text):
            raise ValueError("char_len не совпадает с длиной text")
        return v

    @field_validator("text")
    @classmethod
    def _within_bounds(cls, v: str) -> str:
        if len(v) > MAX_CHUNK_CHARS:
            raise ValueError(f"чанк длиннее {MAX_CHUNK_CHARS} символов")
        return v


class Citation(BaseModel):
    n: int = Field(ge=1)
    chunk_id: str
    source_id: str
    source_title: str
    source_type: SourceType
    heading_path: list[str]
    quote: str = Field(max_length=MAX_QUOTE_CHARS)
    url: str | None = None


LimitationKind = Literal[
    "single_org", "provider_stance", "reliability_note", "no_data", "derived_only"
]


class Limitation(BaseModel):
    kind: LimitationKind
    text: str
    source_ids: list[str] = Field(default_factory=list)


class AnswerPayload(BaseModel):
    answer_markdown: str
    citations: list[Citation] = Field(default_factory=list)
    limitations: list[Limitation] = Field(default_factory=list)
    refused: bool = False
    meta: dict = Field(default_factory=dict)


class Hit(BaseModel):
    chunk: Chunk
    source: Source
    score: float
    rank: int = Field(ge=1)
    components: dict[str, float] = Field(default_factory=dict)


_TRANSLIT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e", "ж": "zh",
    "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o",
    "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f", "х": "h", "ц": "ts",
    "ч": "ch", "ш": "sh", "щ": "sch", "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu",
    "я": "ya",
}

SOURCE_ID_MAX = 60
SOURCE_ID_DIR_MAX = 24


def _slug(text: str, limit: int) -> str:
    text = unicodedata.normalize("NFC", text).lower()
    text = "".join(_TRANSLIT.get(ch, ch) for ch in text)
    return re.sub(r"[^a-z0-9]+", "-", text).strip("-")[:limit].strip("-")


def make_source_id(rel_path: str) -> str:
    """Стабильный slug источника из пути внутри корпуса. Контракт §2 — менять нельзя.

    Идентификатор строится из родительской папки и имени файла: в корпусе есть одноимённые
    файлы в «Источники видео/» и «Тезисы видео/», и slug только по имени давал бы коллизию,
    разрешение которой зависело бы от порядка обхода — то есть id перестал бы быть стабильным.
    Он попадает в БД, eval-разметку и ссылки UI: его изменение обесценивает всю разметку.
    """
    parts = rel_path.replace("\\", "/").split("/")
    stem = parts[-1].rsplit(".", 1)[0]
    parent = parts[-2] if len(parts) > 1 else "root"
    head = _slug(parent, SOURCE_ID_DIR_MAX) or "root"
    tail = _slug(stem, SOURCE_ID_MAX) or "source"
    return f"{head}--{tail}"


def make_chunk_id(source_id: str, ordinal: int) -> str:
    return f"{source_id}#{ordinal:03d}"


def content_hash(text: str) -> str:
    """Хэш нормализованного текста: переводы строк и хвостовые пробелы не считаются
    изменением содержимого, иначе снапшот перестаёт воспроизводиться между ОС."""
    normalized = "\n".join(line.rstrip() for line in text.replace("\r\n", "\n").split("\n"))
    return hashlib.sha256(normalized.strip().encode("utf-8")).hexdigest()


def estimate_tokens(text: str) -> int:
    return len(text) // 3


def quote_anchor(text: str) -> str:
    return text[:QUOTE_ANCHOR_CHARS]
