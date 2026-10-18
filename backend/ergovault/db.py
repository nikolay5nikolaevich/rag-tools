"""PostgreSQL: SQL, миграции, загрузка и поиск. Фильтр видимости применяется в SQL."""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import psycopg
from psycopg import sql
from psycopg.rows import dict_row
from psycopg.types.json import Json

from ergovault.models import Chunk, Hit, Profile, Source

# corpus_version не входит в pydantic-модели (models.py) — это метаданные слоя
# хранения, а не домена. Репозиторий принимает его отдельным параметром.
_SOURCE_COLUMNS = [
    "id",
    "path",
    "type",
    "title",
    "url",
    "language",
    "published",
    "duration_minutes",
    "speaker_org",
    "stance",
    "derived",
    "reliability_note",
    "visibility",
    "content_hash",
    "corpus_version",
]
_CHUNK_COLUMNS = [
    "id",
    "source_id",
    "ordinal",
    "heading_path",
    "text",
    "char_len",
    "token_estimate",
    "atomic",
    "quote_anchor",
    "corpus_version",
]


def visible_values(profile: Profile) -> tuple[str, ...]:
    """Профиль -> допустимые значения visibility. См. settings.Settings.visibilities."""
    return ("public",) if profile == "public" else ("public", "pilot")


def _upsert(conn: psycopg.Connection, table: str, columns: list[str], rows: list[tuple]) -> None:
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
    их не трогаем; embedding не входит в список колонок — его заполняет indexing через
    upsert_embeddings(), и общий upsert не должен затирать уже посчитанные векторы."""
    rows = [
        tuple(
            getattr(c, col) if col != "corpus_version" else corpus_version for col in _CHUNK_COLUMNS
        )
        for c in chunks
    ]
    _upsert(conn, "chunks", _CHUNK_COLUMNS, rows)
    conn.commit()


def _vector_literal(vector: Sequence[float]) -> str:
    """Вектор -> текстовый литерал pgvector. 9 значащих цифр — round-trip для float32,
    а больше в колонке `vector` всё равно не хранится (pgvector кладёт float4)."""
    return "[" + ",".join(f"{float(v):.9g}" for v in vector) + "]"


def upsert_embeddings(
    conn: psycopg.Connection,
    vectors: Mapping[str, Sequence[float]],
    *,
    dim: int,
    batch_size: int = 200,
) -> int:
    """Записывает `chunks.embedding` для указанных chunk_id. Возвращает число обновлённых
    строк: оно меньше len(vectors), если часть id в базе отсутствует. Расхождение разбирает
    вызывающий — репозиторий сам чанков не создаёт, чанк без строки в БД это рассинхрон
    кэша эмбеддингов с корпусом, а не повод создавать неполный чанк.

    Размерность проверяется до записи: вектор не той длины — ошибка данных (не та модель,
    битый кэш), а не повод молча обрезать или дополнить нулями.
    """
    wrong = {cid: len(vec) for cid, vec in vectors.items() if len(vec) != dim}
    if wrong:
        sample = ", ".join(f"{cid}: {size}" for cid, size in list(wrong.items())[:3])
        raise ValueError(
            f"ожидалась размерность {dim}, получено другое у {len(wrong)} векторов ({sample})"
        )
    if not vectors:
        return 0

    items = [(_vector_literal(vec), cid) for cid, vec in vectors.items()]
    updated = 0
    with conn.cursor() as cur:
        for start in range(0, len(items), batch_size):
            cur.executemany(
                "UPDATE chunks SET embedding = %s::vector WHERE id = %s",
                items[start : start + batch_size],
            )
            # psycopg3 суммирует rowcount по всем строкам executemany; -1 означает
            # «драйвер не смог посчитать» — тогда честнее не прибавлять ничего, чем
            # выдумать число обновлённых строк.
            updated += cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
    conn.commit()
    return updated


def embedding_coverage(
    conn: psycopg.Connection, *, corpus_version: str | None = None
) -> tuple[int, int]:
    """(чанков всего, из них с непустым embedding). Нужна CLI, eval и отчётам, чтобы SQL
    не расползался за пределы этого модуля (CONTRACTS.md §0)."""
    where, params = "", ()
    if corpus_version is not None:
        where, params = " WHERE corpus_version = %s", (corpus_version,)
    with conn.cursor() as cur:
        cur.execute(f"SELECT count(*), count(embedding) FROM chunks{where}", params)
        total, with_embedding = cur.fetchone()
    return int(total), int(with_embedding)


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


# Общий для fts и vector набор колонок: Hit несёт целиком и Chunk, и Source (§6),
# поэтому оба запроса читают одно и то же и различаются только ранжированием.
_HIT_COLUMNS = """
    c.id AS c_id, c.source_id AS c_source_id, c.ordinal, c.heading_path, c.text,
    c.char_len, c.token_estimate, c.atomic, c.quote_anchor,
    s.id AS s_id, s.path, s.type, s.title, s.url, s.language, s.published,
    s.duration_minutes, s.speaker_org, s.stance, s.derived, s.derived_of,
    s.reliability_note, s.visibility, s.content_hash
"""


def _row_to_chunk(row: dict) -> Chunk:
    return Chunk(
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


def _row_to_source(row: dict) -> Source:
    return Source(
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


def list_chunk_views(conn: psycopg.Connection, profile: Profile) -> list[tuple[Chunk, Source]]:
    """Все видимые чанки с их источниками, в порядке источник → ordinal.

    Нужен варианту сравнения «весь корпус в контексте» (T-30): он передаёт модели весь
    корпус без поиска. Фильтр видимости — тот же SQL-предикат, что в поиске и каталоге:
    закрытый источник не попадает в контекст ни здесь, ни где-либо ещё.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            f"SELECT {_HIT_COLUMNS} FROM chunks c JOIN sources s ON s.id = c.source_id "
            "WHERE s.visibility = ANY(%s) ORDER BY c.source_id, c.ordinal",
            (list(visible_values(profile)),),
        )
        rows = cur.fetchall()
    return [(_row_to_chunk(row), _row_to_source(row)) for row in rows]


def get_chunk_view(
    conn: psycopg.Connection, chunk_id: str, profile: Profile
) -> tuple[Chunk, Source] | None:
    """Чанк вместе с его источником, если он виден в профиле (§8, `GET /api/chunks/{id}`).

    Видимость проверяется тем же SQL-фильтром, что и в поиске: закрытый источник не
    открывается по прямой ссылке на чанк, даже если идентификатор откуда-то известен.
    Нет строки и нет доступа неразличимы снаружи — оба случая дают None.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            f"SELECT {_HIT_COLUMNS} FROM chunks c JOIN sources s ON s.id = c.source_id "
            "WHERE c.id = %s AND s.visibility = ANY(%s)",
            (chunk_id, list(visible_values(profile))),
        )
        row = cur.fetchone()
    return (_row_to_chunk(row), _row_to_source(row)) if row else None


def corpus_counts(conn: psycopg.Connection, profile: Profile) -> tuple[int, int]:
    """(источников, чанков), видимых в профиле — для `/healthz`. Только два COUNT,
    без чтения текстов и векторов: проверка живости не должна стоить как запрос."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM sources WHERE visibility = ANY(%s)",
            (list(visible_values(profile)),),
        )
        sources = int(cur.fetchone()[0])
        cur.execute(
            "SELECT count(*) FROM chunks c JOIN sources s ON s.id = c.source_id "
            "WHERE s.visibility = ANY(%s)",
            (list(visible_values(profile)),),
        )
        chunks = int(cur.fetchone()[0])
    return sources, chunks


# --- журнал запросов (§5, §8) ---------------------------------------------------------

_QUERY_LOG_INSERT = """
INSERT INTO query_log (
  id, question, profile, retrieval_mode, corpus_version, prompt_version, model,
  retrieved, answer, latency_ms, cost_usd, error
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""


def insert_query_log(
    conn: psycopg.Connection,
    *,
    question: str,
    profile: Profile,
    retrieval_mode: str,
    corpus_version: str,
    prompt_version: str,
    model: str,
    retrieved: Sequence[Mapping[str, Any]],
    answer: Mapping[str, Any] | None = None,
    latency_ms: int | None = None,
    cost_usd: float | None = None,
    error: str | None = None,
) -> str:
    """Записать запрос и вернуть его id — он же `meta.query_id` в ответе и ключ отзыва.

    Один и тот же вызов пишет и удачный ответ (`answer`), и сбой генерации (`error`):
    неудачный вызов модели тоже стоит денег и времени, и без него журнал показывал бы
    завышенное качество. `retrieved` пишется всегда — по нему видно, что именно было
    найдено, даже когда ответа нет.
    """
    query_id = str(uuid.uuid4())
    with conn.cursor() as cur:
        cur.execute(
            _QUERY_LOG_INSERT,
            (
                query_id,
                question,
                profile,
                retrieval_mode,
                corpus_version,
                prompt_version,
                model,
                Json(list(retrieved)),
                Json(dict(answer)) if answer is not None else None,
                latency_ms,
                cost_usd,
                error,
            ),
        )
    conn.commit()
    return query_id


def set_feedback(
    conn: psycopg.Connection, query_id: str, verdict: str, comment: str | None = None
) -> bool:
    """Проставить отзыв существующей записи журнала. False — записи с таким id нет.

    Отзыв ставится на запрос, а не на отдельную таблицу: вердикт разбирается вместе с
    вопросом, выдачей и ответом, иначе он ничего не объясняет.
    """
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE query_log SET feedback = %s, feedback_comment = %s WHERE id = %s",
            (verdict, comment, query_id),
        )
        updated = cur.rowcount
    conn.commit()
    return bool(updated and updated > 0)


# Нормировка ts_rank_cd: 4 — делить на среднюю гармоническую дистанцию между покрытиями.
# Нормировки по длине (1, 2) на проверочном наборе опускали тематические хабы — самые
# длинные чанки корпуса — и снижали recall гибрида (eval/reports/retrieval-comparison.md).
FTS_RANK_NORMALIZATION = 4

# Вопрос целиком почти никогда не совпадает с чанком по всем словам, а plainto_tsquery
# связывает их через AND: на полных вопросах ветка была пустой. Поэтому чанк находится
# по любому значимому слову (any_*), а чанки со всеми словами (all_*) идут первыми.
# plainto_tsquery сама разбирает пользовательский текст, в any_* меняются только
# её операторы — синтаксис tsquery из вопроса не исполняется. Для simple стоп-слова
# отбрасываются по словарю russian: иначе OR находил бы каждый чанк с «и» или «в».
_SEARCH_SQL = f"""
WITH q AS (
  SELECT
    plainto_tsquery('russian', %(q)s) AS all_ru,
    plainto_tsquery('simple', %(q)s) AS all_simple,
    replace(plainto_tsquery('russian', %(q)s)::text, '&', '|')::tsquery AS any_ru,
    replace(plainto_tsquery('simple', (
      SELECT coalesce(string_agg(token, ' '), '')
      FROM ts_debug('russian', %(q)s)
      WHERE lexemes <> '{{}}'
    ))::text, '&', '|')::tsquery AS any_simple
)
SELECT
{_HIT_COLUMNS},
    ts_rank_cd(c.tsv_ru, q.any_ru, %(norm)s) AS rank_ru,
    ts_rank_cd(c.tsv_simple, q.any_simple, %(norm)s) AS rank_simple,
    (c.tsv_ru @@ q.all_ru OR c.tsv_simple @@ q.all_simple) AS all_terms
FROM chunks c
JOIN sources s ON s.id = c.source_id
CROSS JOIN q
WHERE s.visibility = ANY(%(visibilities)s)
  AND (c.tsv_ru @@ q.any_ru OR c.tsv_simple @@ q.any_simple)
ORDER BY all_terms DESC,
         (%(w_ru)s * ts_rank_cd(c.tsv_ru, q.any_ru, %(norm)s)
          + %(w_simple)s * ts_rank_cd(c.tsv_simple, q.any_simple, %(norm)s)) DESC,
         c.id
LIMIT %(k)s
"""


_VECTOR_SEARCH_SQL = f"""
SELECT
{_HIT_COLUMNS},
    c.embedding <=> %(vec)s::vector AS distance
FROM chunks c
JOIN sources s ON s.id = c.source_id
WHERE s.visibility = ANY(%(visibilities)s)
  AND c.embedding IS NOT NULL
ORDER BY c.embedding <=> %(vec)s::vector
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
    """Полнотекстовый поиск: чанк находится по любому значимому слову вопроса, чанки со
    всеми словами идут первыми, дальше ранг = ts_rank_cd(russian) + ts_rank_cd(simple),
    веса — параметры (в проде приходят из settings.weight_ru/weight_simple,
    docs/CONTRACTS.md §9). Фильтр профиля — в SQL (`WHERE s.visibility = ANY(...)`),
    не постфильтром в Python."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            _SEARCH_SQL,
            {
                "q": query,
                "visibilities": list(visible_values(profile)),
                "w_ru": weight_ru,
                "w_simple": weight_simple,
                "norm": FTS_RANK_NORMALIZATION,
                "k": k,
            },
        )
        rows = cur.fetchall()

    hits: list[Hit] = []
    for rank, row in enumerate(rows, start=1):
        rank_ru = float(row["rank_ru"])
        rank_simple = float(row["rank_simple"])
        hits.append(
            Hit(
                chunk=_row_to_chunk(row),
                source=_row_to_source(row),
                score=weight_ru * rank_ru + weight_simple * rank_simple,
                rank=rank,
                # fts_all_terms=1: в чанке есть все слова вопроса. Такие чанки идут
                # первыми, поэтому score внутри выдачи может быть немонотонным.
                components={
                    "fts_ru": rank_ru,
                    "fts_simple": rank_simple,
                    "fts_all_terms": 1.0 if row["all_terms"] else 0.0,
                },
            )
        )
    return hits


def search_vector(
    conn: psycopg.Connection,
    embedding: Sequence[float],
    k: int = 10,
    *,
    profile: Profile,
) -> list[Hit]:
    """kNN по косинусной близости (`<=>`, hnsw-индекс из §5). Вектор запроса считает
    вызывающий (`ergovault.embeddings`) — репозиторий модель не грузит и текст не векторизует.

    `score` — косинусная близость (1 - distance), чтобы «больше = лучше» совпадало с
    fts-ветвью и RRF сравнивал ранги, а не разнонаправленные величины; сама
    дистанция остаётся в `components` для отладки. Чанки без посчитанного embedding
    отсекаются в SQL, а не уползают в конец выдачи как «бесконечно далёкие». Фильтр
    профиля — в SQL, как и в search_fts.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            _VECTOR_SEARCH_SQL,
            {
                "vec": _vector_literal(embedding),
                "visibilities": list(visible_values(profile)),
                "k": k,
            },
        )
        rows = cur.fetchall()

    hits: list[Hit] = []
    for rank, row in enumerate(rows, start=1):
        distance = float(row["distance"])
        similarity = 1.0 - distance
        hits.append(
            Hit(
                chunk=_row_to_chunk(row),
                source=_row_to_source(row),
                score=similarity,
                rank=rank,
                components={"vector": similarity, "cosine_distance": distance},
            )
        )
    return hits


MIGRATIONS_DIR = Path(__file__).resolve().parents[1] / "migrations"

_ENSURE_TABLE = """
CREATE TABLE IF NOT EXISTS schema_migrations (
  filename text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now()
);
"""


def _migration_files(migrations_dir: Path) -> list[Path]:
    return sorted(migrations_dir.glob("*.sql"), key=lambda p: p.name)


def upgrade(database_url: str, migrations_dir: Path = MIGRATIONS_DIR) -> list[str]:
    """Применяет неприменённые миграции. Возвращает имена файлов, которые применил сейчас."""
    applied_now: list[str] = []
    with psycopg.connect(database_url, autocommit=False) as conn:
        with conn.cursor() as cur:
            cur.execute(_ENSURE_TABLE)
        conn.commit()

        for path in _migration_files(migrations_dir):
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM schema_migrations WHERE filename = %s", (path.name,))
                if cur.fetchone() is not None:
                    continue
                cur.execute(path.read_text(encoding="utf-8"))
                cur.execute("INSERT INTO schema_migrations (filename) VALUES (%s)", (path.name,))
            conn.commit()
            applied_now.append(path.name)
    return applied_now


def load_index(index_dir: Path, *, settings=None) -> tuple[int, int]:
    """Загрузить JSONL существующими upsert-функциями; вернуть число источников и чанков."""

    from ergovault.settings import get_settings

    settings = settings or get_settings()
    sources = [
        Source.model_validate_json(line)
        for line in (index_dir / "sources.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    chunks = [
        Chunk.model_validate_json(line)
        for line in (index_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    with psycopg.connect(settings.database_url) as conn:
        upsert_sources(conn, sources, corpus_version=settings.corpus_version)
        upsert_chunks(conn, chunks, corpus_version=settings.corpus_version)
    return len(sources), len(chunks)
