"""Репозиторий БД: upsert источников/чанков, точечные выборки, полнотекстовый поиск.

Нормативный источник — docs/CONTRACTS.md §5 (схема) и §6 (Hit, фильтр профиля в SQL).
ORM не используется: psycopg3 и явный SQL, схема маленькая.
"""

from __future__ import annotations

from collections.abc import Sequence

import psycopg
from psycopg import sql
from psycopg.rows import dict_row

from ergovault.models import Chunk, Hit, Profile, Source

# corpus_version не входит в pydantic-модели (models.py, T-00) — это метаданные слоя
# хранения, а не домена. Репозиторий принимает его отдельным параметром.
_SOURCE_COLUMNS = [
    "id", "path", "type", "title", "url", "language", "published",
    "duration_minutes", "speaker_org", "stance", "derived",
    "reliability_note", "visibility", "content_hash", "corpus_version",
]
_CHUNK_COLUMNS = [
    "id", "source_id", "ordinal", "heading_path", "text", "char_len",
    "token_estimate", "atomic", "quote_anchor", "corpus_version",
]


def visible_values(profile: Profile) -> tuple[str, ...]:
    """Профиль -> допустимые значения visibility. См. settings.Settings.visibilities."""
    return ("public",) if profile == "public" else ("public", "pilot")


def _upsert(
    conn: psycopg.Connection, table: str, columns: list[str], rows: list[tuple]
) -> None:
    if not rows:
        return
    update_cols = [c for c in columns if c != "id"]
    query = sql.SQL(
        "INSERT INTO {table} ({cols}) VALUES ({placeholders}) "
        "ON CONFLICT (id) DO UPDATE SET {updates}"
    ).format(
        table=sql.Identifier(table),
        cols=sql.SQL(", ").join(sql.Identifier(c) for c in columns),
        placeholders=sql.SQL(", ").join(sql.Placeholder() for _ in columns),
        updates=sql.SQL(", ").join(
            sql.SQL("{c} = EXCLUDED.{c}").format(c=sql.Identifier(c)) for c in update_cols
        ),
    )
    with conn.cursor() as cur:
        cur.executemany(query, rows)


def upsert_sources(
    conn: psycopg.Connection, sources: Sequence[Source], *, corpus_version: str
) -> None:
    """Батч upsert источников, идемпотентно по id.

    `derived_of` вставляется отдельным проходом после того, как все строки батча уже
    существуют: FK `derived_of -> sources.id` иначе может сработать раньше, чем в базе
    появится сам первоисточник из того же батча (порядок обхода файлов это не гарантирует).
    """
    rows = [
        tuple(getattr(s, c) if c != "corpus_version" else corpus_version for c in _SOURCE_COLUMNS)
        for s in sources
    ]
    _upsert(conn, "sources", _SOURCE_COLUMNS, rows)
    if sources:
        with conn.cursor() as cur:
            cur.executemany(
                "UPDATE sources SET derived_of = %s WHERE id = %s",
                [(s.derived_of, s.id) for s in sources],
            )
    conn.commit()


def upsert_chunks(
    conn: psycopg.Connection, chunks: Sequence[Chunk], *, corpus_version: str
) -> None:
    """Батч upsert чанков, идемпотентно по id. tsv_ru/tsv_simple — generated-колонки,
    их не трогаем; embedding не входит в список колонок — его заполняет T-12 и upsert
    не должен затирать уже посчитанные векторы."""
    rows = [
        tuple(getattr(c, col) if col != "corpus_version" else corpus_version for col in _CHUNK_COLUMNS)
        for c in chunks
    ]
    _upsert(conn, "chunks", _CHUNK_COLUMNS, rows)
    conn.commit()


def get_chunk(conn: psycopg.Connection, chunk_id: str) -> Chunk | None:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT id, source_id, ordinal, heading_path, text, char_len, "
            "token_estimate, atomic, quote_anchor FROM chunks WHERE id = %s",
            (chunk_id,),
        )
        row = cur.fetchone()
    return Chunk(**row) if row else None


def list_sources(conn: psycopg.Connection, profile: Profile) -> list[Source]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT id, path, type, title, url, language, published, duration_minutes, "
            "speaker_org, stance, derived, derived_of, reliability_note, visibility, "
            "content_hash FROM sources WHERE visibility = ANY(%s) ORDER BY id",
            (list(visible_values(profile)),),
        )
        rows = cur.fetchall()
    return [Source(**row) for row in rows]


_SEARCH_SQL = """
SELECT
    c.id AS c_id, c.source_id AS c_source_id, c.ordinal, c.heading_path, c.text,
    c.char_len, c.token_estimate, c.atomic, c.quote_anchor,
    s.id AS s_id, s.path, s.type, s.title, s.url, s.language, s.published,
    s.duration_minutes, s.speaker_org, s.stance, s.derived, s.derived_of,
    s.reliability_note, s.visibility, s.content_hash,
    ts_rank_cd(c.tsv_ru, plainto_tsquery('russian', %(q)s)) AS rank_ru,
    ts_rank_cd(c.tsv_simple, plainto_tsquery('simple', %(q)s)) AS rank_simple
FROM chunks c
JOIN sources s ON s.id = c.source_id
WHERE s.visibility = ANY(%(visibilities)s)
  AND (
    c.tsv_ru @@ plainto_tsquery('russian', %(q)s)
    OR c.tsv_simple @@ plainto_tsquery('simple', %(q)s)
  )
ORDER BY (%(w_ru)s * ts_rank_cd(c.tsv_ru, plainto_tsquery('russian', %(q)s))
          + %(w_simple)s * ts_rank_cd(c.tsv_simple, plainto_tsquery('simple', %(q)s))) DESC
LIMIT %(k)s
"""


def search_fts(
    conn: psycopg.Connection,
    query: str,
    k: int = 10,
    *,
    profile: Profile,
    weight_ru: float = 1.0,
    weight_simple: float = 1.0,
) -> list[Hit]:
    """Полнотекстовый поиск, ранг = ts_rank_cd(russian) + ts_rank_cd(simple), веса — параметры
    (в проде приходят из settings.weight_ru/weight_simple, docs/CONTRACTS.md §9). Фильтр
    профиля — в SQL (`WHERE s.visibility = ANY(...)`), не постфильтром в Python."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            _SEARCH_SQL,
            {
                "q": query,
                "visibilities": list(visible_values(profile)),
                "w_ru": weight_ru,
                "w_simple": weight_simple,
                "k": k,
            },
        )
        rows = cur.fetchall()

    hits: list[Hit] = []
    for rank, row in enumerate(rows, start=1):
        chunk = Chunk(
            id=row["c_id"],
            source_id=row["c_source_id"],
            ordinal=row["ordinal"],
            heading_path=row["heading_path"],
            text=row["text"],
            char_len=row["char_len"],
            token_estimate=row["token_estimate"],
            atomic=row["atomic"],
            quote_anchor=row["quote_anchor"],
        )
        source = Source(
            id=row["s_id"],
            path=row["path"],
            type=row["type"],
            title=row["title"],
            url=row["url"],
            language=row["language"],
            published=row["published"],
            duration_minutes=row["duration_minutes"],
            speaker_org=row["speaker_org"],
            stance=row["stance"],
            derived=row["derived"],
            derived_of=row["derived_of"],
            reliability_note=row["reliability_note"],
            visibility=row["visibility"],
            content_hash=row["content_hash"],
        )
        rank_ru = float(row["rank_ru"])
        rank_simple = float(row["rank_simple"])
        hits.append(
            Hit(
                chunk=chunk,
                source=source,
                score=weight_ru * rank_ru + weight_simple * rank_simple,
                rank=rank,
                components={"fts_ru": rank_ru, "fts_simple": rank_simple},
            )
        )
    return hits
