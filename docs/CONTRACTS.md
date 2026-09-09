# ErgoVault RAG — контракты

> Единственный источник правды для всех исполнителей. Ни один тикет не меняет этот файл
> самостоятельно: расхождение с контрактом = баг тикета. Изменение контракта — через тех-лида.
> Версия контракта: `v1`.

## 0. Границы ответственности

| Слой | Пакет | Тикет-владелец |
|---|---|---|
| Парсинг vault → Source/Chunk | `ergovault.ingest` | T-10 |
| Хранение, миграции, поиск по БД | `ergovault.index` | T-11 |
| Эмбеддинги | `ergovault.embed` | T-12 |
| Стратегии поиска | `ergovault.retrieval` | T-20 |
| Промпт, генерация, валидация ответа | `ergovault.generation` | T-21 |
| HTTP API, журнал | `ergovault.api` | T-22 |
| Метрики, судья, эксперимент | `ergovault.eval` | T-13 / T-23 / T-30 |
| UI | `frontend/` | T-14 |

Пересекать границы запрещено. Нужен чужой код — импортируй его публичный интерфейс,
описанный ниже; если интерфейса нет — это блокер, эскалируй, а не дописывай чужой пакет.

## 1. Модель данных (pydantic v2, `ergovault/models.py`, владелец T-00)

```python
SourceType = Literal["video-source", "video-theses", "topic-hub", "article",
                     "course-review", "event", "company", "synthesis", "primary"]
Stance     = Literal["нейтральный", "организатор-провайдер", "подрядчик"]
Profile    = Literal["public", "pilot"]

class Source(BaseModel):
    id: str                      # стабильный slug, см. §2
    path: str                    # путь внутри corpus/v1, POSIX
    type: SourceType
    title: str
    url: str | None              # публичный URL первоисточника; None если не указан
    language: Literal["ru", "en", "fr", "mixed", "unknown"]
    published: str | None        # как в frontmatter, нормализация не требуется
    duration_minutes: int | None
    speaker_org: str | None      # организация/спикер — ключ схлопывания подтверждений
    stance: Stance
    derived: bool                # True для «Тезисы видео» и прочих пересказов своих же заметок
    derived_of: str | None       # id первичной заметки, если derived
    reliability_note: str | None # извлечённый текст, НЕ сгенерированный
    visibility: Literal["public", "pilot"]
    content_hash: str            # sha256 нормализованного текста файла

class Chunk(BaseModel):
    id: str                      # f"{source_id}#{ordinal:03d}"
    source_id: str
    ordinal: int                 # порядок внутри источника, с 0
    heading_path: list[str]      # ["Основные тезисы", "Подготовка"] — от H1 (не включая) вниз
    text: str                    # markdown без frontmatter, Obsidian-ссылки развёрнуты
    char_len: int
    token_estimate: int          # len(text)//3, для бюджета контекста; точность не важна
    atomic: bool                 # True для отдельного нумерованного тезиса
    quote_anchor: str            # первые ≤120 символов text, для подсветки во фронте
```

Инварианты (покрываются тестами T-10):
- `chunk.id` уникален по корпусу; `source_id` всегда существует в `sources`.
- `heading_path` непустой; для преамбулы — `["_intro"]`.
- Чанк не длиннее 4000 символов и не короче 80 (для `atomic` минимум 40).
- `visibility == "pilot"` ⇒ источник из `raw/`, тип `primary`.

## 2. Идентификаторы

`source.id` = `slug(родительская папка, ≤24)` + `--` + `slug(имя файла без расширения, ≤60)`,
где slug = транслитерация → lowercase → замена `[^a-z0-9]+` на `-`. Реализация —
`ergovault.models.make_source_id(rel_path)`, она нормативна.

Родительская папка входит в id не для красоты: в корпусе есть одноимённые файлы в
`Источники видео/` и `Тезисы видео/`, и slug только по имени файла давал бы коллизию,
разрешение которой зависело бы от порядка обхода — то есть id перестал бы быть стабильным.
На снапшоте `v1` все 43 идентификатора уникальны, коллизий нет и суффиксы не нужны.

Идентификатор стабилен между переиндексациями; он попадает в БД, eval-разметку и ссылки UI.
Менять алгоритм нельзя — это обесценивает всю накопленную ручную разметку.

### Заглушки вместо отсутствующих значений

В корпусе `url` часто заполнен строкой-заглушкой: «не указана (Telegram-стрим)»,
«не предоставлен», «не указана пользователем», «не указана в транскрипте». Такие значения
приводятся к `None`, а не попадают в поле как текст — иначе UI покажет ссылку-обманку.
Правило: значение, начинающееся с «не указан», «не предоставл», «нет ссылки», «unknown»,
или не начинающееся с `http` — это `None`. То же для `duration`, `published`, `channel`.

## 3. Чанкинг (нормативно, T-10)

1. Отрезать frontmatter, распарсить в метаданные.
2. Резать по `##` и `###`. Секция = чанк, `heading_path` — цепочка заголовков.
3. В секциях `Основные тезисы` (и любых, где ≥3 нумерованных пункта верхнего уровня)
   каждый пункт — отдельный чанк с `atomic=True`, `heading_path` наследуется.
4. `Полная расшифровка` в публичном профиле содержит только embed `![[raw/...]]`; если после
   удаления embed остаётся <80 символов — чанк не создаётся.
5. Строка `> Примечание по достоверности: …` — текст после двоеточия уходит в
   `source.reliability_note` и **исключается** из тела чанка.
6. Obsidian-ссылки: `[[X|Y]]` → `Y`, `[[X]]` → `X`, `![[X]]` → удалить.
7. Секция длиннее 4000 символов режется по абзацам, перекрытие 0.

## 4. Атрибуция источников (T-10, детерминированные правила)

`speaker_org` берётся из frontmatter `channel` и нормализуется по словарю
`corpus/v1/speaker_org.yml` — он **уже заполнен тех-лидом** по снапшоту v1, вместе с
`corpus/v1/derived_map.yml`. Матчинг: точное вхождение любого alias в поле `channel`,
регистронезависимо. Читать эти файлы, а не изобретать свои правила.

Два факта из словаря, которые влияют на качество ответов:
- материалов с участием RusGlobalExport в корпусе **пять**, а не четыре: четыре собственных
  ролика плюс совместный подкаст со Stan&Tan Talks (плюс страница `Компании/RusGlobalExport`).
  Все они схлопываются в одно подтверждение;
- два источника намеренно оставлены без `speaker_org` (`channel: не указан`). Это не пробел
  в работе, а отказ от догадки: неверная атрибуция ломает схлопывание сильнее, чем её отсутствие.

`stance` по умолчанию `нейтральный`. Канала нет в словаре → `speaker_org=None` и строка в отчёте
`ev ingest --report`. Пустое значение всегда лучше выдуманного.

`derived=True` для всех `type: video-theses`; `derived_of` берётся из `derived_map.yml`
(13 связей, выверены вручную). Поле `source_video` во frontmatter заполнено лишь у одной заметки
из тринадцати, поэтому полагаться на него нельзя. Fuzzy-матчинг по названиям запрещён: ошибка
здесь означает, что пересказ и оригинал будут засчитаны как два независимых подтверждения.

## 5. Схема БД (PostgreSQL 16 + pgvector, владелец T-11)

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE sources (
  id text PRIMARY KEY, path text NOT NULL, type text NOT NULL, title text NOT NULL,
  url text, language text NOT NULL, published text, duration_minutes int,
  speaker_org text, stance text NOT NULL, derived bool NOT NULL DEFAULT false,
  derived_of text REFERENCES sources(id), reliability_note text,
  visibility text NOT NULL DEFAULT 'public', content_hash text NOT NULL,
  corpus_version text NOT NULL
);

CREATE TABLE chunks (
  id text PRIMARY KEY, source_id text NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
  ordinal int NOT NULL, heading_path text[] NOT NULL, text text NOT NULL,
  char_len int NOT NULL, token_estimate int NOT NULL, atomic bool NOT NULL,
  quote_anchor text NOT NULL, corpus_version text NOT NULL,
  tsv_ru tsvector GENERATED ALWAYS AS (to_tsvector('russian', text)) STORED,
  tsv_simple tsvector GENERATED ALWAYS AS (to_tsvector('simple', text)) STORED,
  embedding vector(1024)
);
CREATE INDEX chunks_tsv_ru_idx     ON chunks USING gin (tsv_ru);
CREATE INDEX chunks_tsv_simple_idx ON chunks USING gin (tsv_simple);
CREATE INDEX chunks_trgm_idx       ON chunks USING gin (text gin_trgm_ops);
CREATE INDEX chunks_embedding_idx  ON chunks USING hnsw (embedding vector_cosine_ops);
```

Два tsvector обязательны: `russian` даёт морфологию, `simple` спасает GITEX / HubSpot /
Baselworld / Organic & Natural. Размерность 1024 = BGE-m3; смена модели эмбеддингов — миграция,
а не правка на месте.

Журнал:

```sql
CREATE TABLE query_log (
  id uuid PRIMARY KEY, created_at timestamptz NOT NULL DEFAULT now(),
  question text NOT NULL, profile text NOT NULL, retrieval_mode text NOT NULL,
  corpus_version text NOT NULL, prompt_version text NOT NULL, model text NOT NULL,
  retrieved jsonb NOT NULL,     -- [{chunk_id, score, rank}]
  answer jsonb,                 -- AnswerPayload целиком
  latency_ms int, cost_usd numeric(10,6), error text,
  feedback text                 -- null | 'useful' | 'wrong'
);
```

## 6. Интерфейс поиска (T-20)

```python
class Hit(BaseModel):
    chunk: Chunk; source: Source; score: float; rank: int; components: dict[str, float]

class Retriever(Protocol):
    name: str   # "fts" | "vector" | "hybrid"
    def search(self, query: str, k: int = 10, *, profile: Profile) -> list[Hit]: ...
```

Гибрид — RRF (`score = Σ 1/(60 + rank_i)`), веса в конфиге. `components` обязателен: без него
не отладить, почему имя собственное не нашлось. Профиль фильтрует по `visibility` **в SQL**, не
в Python: утечка pilot-контента в public — дефект уровня «стоп-релиз».

## 7. Контракт ответа (T-21, он же контракт фронта T-14)

```python
class Citation(BaseModel):
    n: int                  # номер [1], [2] в тексте
    chunk_id: str; source_id: str; source_title: str; source_type: SourceType
    heading_path: list[str]
    quote: str                               # ДОСЛОВНАЯ подстрока chunk.text, ≤300 символов
    url: str | None                          # ТОЛЬКО из source.url

class Limitation(BaseModel):
    kind: Literal["single_org", "provider_stance", "reliability_note", "no_data", "derived_only"]
    text: str; source_ids: list[str]

class AnswerPayload(BaseModel):
    answer_markdown: str      # ссылки [n] строго на существующие Citation.n
    citations: list[Citation]
    limitations: list[Limitation]
    refused: bool             # True = корректный отказ; answer_markdown объясняет, чего не хватает
    meta: dict                # corpus_version, retrieval_mode, prompt_version, model, latency_ms, cost_usd
```

Программные (не модельные) гарантии, покрытые тестами T-21:

- каждый `[n]` в `answer_markdown` имеет Citation; лишних Citation нет;
- `Citation.quote` — точная подстрока переданного чанка (нормализация пробелов допустима);
- `Citation.url` берётся из БД; модель URL не пишет вообще — доля выдуманных URL нулевая
  по построению, а не по измерению;
- `limitations` **вычисляются кодом**, не моделью:
  - все цитаты с одним `speaker_org` → `single_org`;
  - среди цитат есть `stance="организатор-провайдер"` → `provider_stance`;
  - у процитированного источника есть `reliability_note` → своя `reliability_note` на источник;
  - все цитаты `derived=True` → `derived_only`;
  - `refused=True` → `no_data`.
- Схлопывание подтверждений — по `speaker_org`, для `derived` — по `derived_of`. Файл ≠ источник:
  оригинал видео и его пересказ никогда не считаются двумя подтверждениями.

## 8. HTTP API (T-22)

```
POST /api/ask          {question, profile?, retrieval_mode?} -> AnswerPayload
GET  /api/sources      -> [Source]   (public-профиль по умолчанию)
GET  /api/chunks/{id}  -> {chunk, source}
POST /api/feedback     {query_id, verdict: "useful"|"wrong", comment?} -> 204
GET  /api/eval/summary -> результаты последнего прогона (экран «Проверка качества»)
GET  /healthz          -> {status, corpus_version, chunks, sources}
```

Ошибки: `{"error": {"code": str, "message": str}}`, коды `bad_request|not_found|upstream|internal`.

## 9. Конфигурация

`.env` / `settings.py` (pydantic-settings, префикс `EV_`):
`EV_PROFILE=public|pilot`, `EV_DATABASE_URL`, `EV_EMBED_MODEL=BAAI/bge-m3`,
`EV_GEN_MODEL=claude-opus-5`, `EV_JUDGE_MODEL=claude-sonnet-5`, `EV_RETRIEVAL_MODE=hybrid`,
`EV_TOP_K=8`, `EV_CORPUS_VERSION=v1`, `EV_PROMPT_VERSION=p1`,
`EV_LLM_BACKEND=cli|api`, `EV_ANTHROPIC_API_KEY` (опционален).

### Доступ к модели: подписка, а не ключ

Основной режим — `EV_LLM_BACKEND=cli`: генерация идёт через Claude Code CLI в headless-режиме
по подписке, платный API-ключ не нужен. Весь код ходит в модель **только** через
`ergovault.llm.LLMProvider` (T-15); прямые вызовы SDK из других пакетов запрещены.

Два следствия, которые обязаны быть отражены в отчёте:

- headless-режим добавляет ~17 000 токенов служебного промпта Claude Code к каждому вызову
  (замеры и флаги минимизации — в T-15). Поэтому `total_cost_usd`, сообщаемый CLI, в отчёт
  не идёт: стоимость считается по прайсу от наших собственных токенов;
- в headless-режиме **нельзя задать temperature**. Для судьи (T-23) это означает, что
  детерминизм не гарантирован конструкцией и должен быть измерен: повторные прогоны и
  зафиксированный разброс вместо утверждения «temperature=0».

Индексация в обоих режимах работает без единого внешнего ключа — эмбеддинги локальные.
Это требование раздела 9 спеки, и оно не зависит от способа доступа к генератору.

## 10. Версионирование

`corpus_version`, `prompt_version`, `retrieval_mode`, `model` пишутся в каждую строку
`query_log` и в каждый прогон eval. Результат эксперимента, не привязанный к этим четырём,
считается недействительным и в отчёт не идёт.
