# T-11 — База: Docker Compose, миграции, репозитории, полнотекстовый поиск

**Пакет:** `backend/ergovault/index/`, `ops/`, `docker-compose.yml`, `backend/migrations/`.
**Читать перед стартом:** `docs/CONTRACTS.md` §5, §6, §9.
**Зависит от:** T-00.

---

## Результат: что должно лежать на диске, когда тикет закрыт

| Артефакт | Что в нём |
|---|---|
| `docker-compose.yml` | сервис `db` (pgvector/pgvector:pg16) с healthcheck и томом, плюс плейсхолдер `api` |
| `backend/migrations/001_init.sql` | схема §5 целиком: расширения, две таблицы, четыре индекса, query_log |
| `backend/ergovault/index/repository.py` | upsert_sources, upsert_chunks, get_chunk, list_sources, search_fts |
| `backend/ergovault/index/migrate.py` | раннер `ev db upgrade`, применяет SQL-файлы по номеру |
| `backend/tests/fixtures/chunks.sample.jsonl` | 20 строк по схеме §1 — чтобы не ждать T-10 |
| `backend/tests/test_repository.py` | тесты из раздела «Приёмка», на живой базе |

---

## Приёмка: чем проверяется, что готово

| Проверка | Ожидание |
|---|---|
| `docker compose up -d db` на чистой машине | контейнер поднимается, healthcheck зелёный |
| `ev db upgrade` | схема накатывается с нуля; повторный запуск идемпотентен |
| `ev db load corpus/v1/index/` | строки в `sources` и `chunks`, счётчики совпадают с JSONL |
| Поиск «GITEX» | находит. На ветке `russian` в одиночку не нашёлся бы — это проверка того, что `simple` реально работает |
| Поиск «Baselworld», «HubSpot» | находит: имена собственные не теряются |
| Поиск «обработка контактов после выставки» | находит по морфологии — проверка ветки `russian` |
| `Hit.components` | заполнен обоими слагаемыми ранга раздельно, не одним итоговым числом |
| `search_fts(profile='public')` | ноль строк с `visibility='pilot'`; фильтр в SQL, не в Python |

Строки таблицы — условия закрытия тикета. Разошлось с реальностью — эскалация
тех-лиду с конкретными файлами, а не правка ожидаемого под факт.

---

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

---

## Границы

- Не писать ретриверы верхнего уровня и RRF — это T-20. Здесь только `search_fts`.
- Не считать эмбеддинги: колонка есть, заполняет её T-12.
- ORM не вводить: psycopg3 и явный SQL. Схема маленькая, абстракция над ней — лишний вес.
- Alembic не тащить: миграции нумерованными SQL-файлами, их читает человек.

---
