# T-11 — База: Docker Compose, миграции, репозитории, полнотекстовый поиск

**Пакет:** `backend/ergovault/index/`, `ops/`, `docker-compose.yml`, `backend/migrations/`.
**Читать перед стартом:** `docs/CONTRACTS.md` §5, §6, §9.
**Зависит от:** T-00.

## Задача

Поднять PostgreSQL 16 + pgvector + pg_trgm, накатить схему §5, дать слой репозиториев и рабочий
полнотекстовый поиск с двумя tsvector.

## Что сделать

1. `docker-compose.yml`: сервис `db` на образе `pgvector/pgvector:pg16`, том, healthcheck,
   порт из `.env`. Сервис `api` — заглушка-плейсхолдер, его наполнит T-22.
2. Миграции — простые пронумерованные SQL-файлы (`001_init.sql`, …) и раннер `ev db upgrade`.
   Alembic не нужен: одна схема, одна база, миграции читает человек. Не тащить лишнее.
3. `ergovault/index/repository.py`:
   - `upsert_sources(list[Source])`, `upsert_chunks(list[Chunk])` — батчами, идемпотентно по PK;
   - `get_chunk(id)`, `list_sources(profile)`;
   - `search_fts(query, k, profile) -> list[Hit]` (без embedding).
4. Полнотекстовый поиск — нормативно:
   ранг = `ts_rank_cd(tsv_ru, plainto_tsquery('russian', q))` +
   `ts_rank_cd(tsv_simple, plainto_tsquery('simple', q))`, веса в конфиге.
   `components` в `Hit` заполняется обоими слагаемыми раздельно.
   Фильтр профиля — в SQL (`WHERE s.visibility = ANY(...)`), не в Python.
5. `ev db load` — загрузка `corpus/v1/index/*.jsonl` (выход T-10) в базу.
   До готовности T-10 работать на фикстуре `backend/tests/fixtures/chunks.sample.jsonl`
   (сделать её самому, 20 строк, формат по §1) — это снимает зависимость от T-10.
6. Тесты на живой базе (testcontainers или compose-профиль `test`):
   - «GITEX» и «Baselworld» находятся (проверка ветки `simple` — на `russian` они разваливаются);
   - «обработка контактов после выставки» находится с морфологией (проверка ветки `russian`);
   - `profile="public"` никогда не возвращает строки с `visibility='pilot'`.

## Границы

- Не писать ретриверы верхнего уровня и RRF — это T-20. Здесь только `search_fts`.
- Не считать эмбеддинги — колонка есть, заполняет её T-12.
- ORM не вводить: psycopg3 + явный SQL. Схема маленькая, слой абстракции над ней — лишний вес.

## Готово, когда

`docker compose up -d db && ev db upgrade && ev db load` проходит с нуля на чистой машине,
тесты двух веток FTS и теста изоляции профиля зелёные.
