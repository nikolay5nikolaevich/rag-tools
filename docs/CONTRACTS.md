# ErgoVault RAG — форматы данных и гарантии

> Версия форматов: `v1`. Рефакторинг 10.09.2026 меняет расположение кода, но не данные.
> Карта модулей, запуск и очередь — [ARCHITECTURE.md](ARCHITECTURE.md).
> Действующие задания находятся в `docs/tickets/`.

## 0. Где находится код

- `ingest.py` и `chunking.py`: документы → Source/Chunk → JSONL.
- `indexing.py`: подготовка базы, кэш и запуск векторизации.
- `embeddings.py`: модель и кодирование; веса загружаются только при необходимости.
- `db.py`: SQL, миграции, загрузка, фильтр видимости.
- `search.py`: общий поиск для приложения и оценки качества.
- `llm/`: два способа обращения к Claude.
- `eval/`: вопросы, прогоны и метрики.
- `cli.py`: только команды; прикладной код не зависит от Typer.
- `models.py`, `settings.py`: данные и конфигурация.

- `answering.py`, `prompts/`: ответ по найденным фрагментам, проверка, ограничения (§7).
- `api.py`: HTTP-слой (§8) — разбор запроса, вызов поиска и ответа, журнал, коды ошибок.

Прикладной логики в `api.py` нет: поиск остаётся в `search.py`, ответ — в `answering.py`,
SQL и журнал — в `db.py`.
Команды объявляются явно в одном `cli.py`; основной запуск подготовки — `ev prepare`.

## 1. Модель данных (pydantic v2, `ergovault/models.py`)

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

Инварианты покрываются тестами:
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

## 3. Чанкинг (нормативно)

1. Отрезать frontmatter, распарсить в метаданные.
2. Резать по `##` и `###`. Секция = чанк, `heading_path` — цепочка заголовков.
3. **Заголовки нормализуются перед сравнением**: ведущая нумерация снимается
   (`## 4. Основные тезисы` → `Основные тезисы`). В корпусе два семейства заметок с разной
   разметкой — «Источники видео» без номеров, «Тезисы видео» с номерами 1–8. Сравнение
   по сырой строке промахнётся мимо всех тезисных заметок. В `heading_path` пишется
   нормализованный заголовок.
4. В секции `Основные тезисы` (и любой другой, где ≥3 нумерованных пункта верхнего уровня)
   каждый пункт — отдельный чанк с `atomic=True`, `heading_path` наследуется.
5. `Полная расшифровка` в публичном профиле содержит только embed `![[raw/...]]`; если после
   удаления embed остаётся <80 символов — чанк не создаётся.
6. Obsidian-ссылки: `[[X|Y]]` → `Y`, `[[X]]` → `X`, `![[X]]` → удалить.
7. Секция длиннее 4000 символов режется по абзацам, перекрытие 0.

### 3.1. Извлечение `reliability_note` — два источника, не один

Оговорки о достоверности лежат в корпусе в двух разных формах, и обе обязательны:

| Где | В каких заметках | Формат |
|---|---|---|
| Раздел `Ограничения источника` (обычно `## 7. …`) | «Тезисы видео», ~12 заметок | маркированный список, несколько пунктов |
| Блок `> Примечание по достоверности: …` | «Источники видео», 2 заметки | одна цитата-абзац |

Обе формы складываются в `source.reliability_note` (несколько пунктов — через перевод строки)
и **исключаются из тела чанков**: это метаданные источника, а не материал для ответа.

**Перенос на оригинал.** Раздел «Ограничения источника» написан в производной заметке
(`video-theses`), а цитировать система будет в основном оригинал (`video-source`). Поэтому
`reliability_note`, извлечённый из производной заметки, **копируется на источник, указанный
в её `derived_of`**, если у того ещё нет собственного. Без этого переноса цитата из оригинала
придёт в ответ без оговорки — а именно эти оговорки и есть содержимое блока «Ограничения»
(спека 5.3: они извлекаются, а не генерируются).

## 4. Атрибуция источников (детерминированные правила)

### Маппинг `type` корпуса в `SourceType` контракта

Во frontmatter корпуса свои имена типов. Таблица нормативна, гадать не нужно:

| `type` в корпусе | `SourceType` | Штук |
|---|---|---|
| `video-source` | `video-source` | 13 |
| `video-theses` | `video-theses` | 13 |
| `topic` | `topic-hub` | 5 |
| `articles-report` | `article` | 1 |
| `courses-report`, файл «Курсы … анализ.md» (без `type`) | `course-review` | 2 |
| `company` | `company` | 1 |
| `event` | `event` | 1 |
| `final-report`, `wiki-overview`, `wiki-index`, `sources-index`, `theses-index`, `companies-index`, `events-index` | `synthesis` | 7 |

Всего 43 источника. Единственный файл без `type` — `wiki/Курсы по международным выставкам —
анализ.md`; он определяется по имени, а не эвристикой по содержимому.


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

## 5. Схема БД (PostgreSQL 16 + pgvector)

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
  feedback text,                -- null | 'useful' | 'wrong'
  feedback_comment text         -- необязательный текст отзыва (миграция 002)
);
```

`feedback_comment` добавлен отдельной миграцией под T-22: вердикт без пояснения часто
не даёт понять, что именно не так с ответом. Поле nullable — комментарий необязателен,
и пустое поле хранится как NULL, а не как пустая строка.

### Публичный интерфейс модуля `db`

Никто, кроме `ergovault.db`, не пишет SQL — остальные слои ходят через эти функции:

```python
upsert_sources(conn, sources, *, corpus_version) -> None
upsert_chunks(conn, chunks, *, corpus_version) -> None        # embedding не трогает
upsert_embeddings(conn, {chunk_id: vector}, *, dim) -> int    # -> число обновлённых строк
embedding_coverage(conn, *, corpus_version=None) -> tuple[int, int]   # (всего, с вектором)
get_chunk(conn, chunk_id) -> Chunk | None
list_sources(conn, profile) -> list[Source]
search_fts(conn, query, k, *, profile, weight_ru, weight_simple) -> list[Hit]
search_vector(conn, embedding, k, *, profile) -> list[Hit]
get_chunk_view(conn, chunk_id, profile) -> tuple[Chunk, Source] | None
corpus_counts(conn, profile) -> tuple[int, int]              # (источников, чанков)
insert_query_log(conn, *, question, profile, retrieval_mode, corpus_version,
                 prompt_version, model, retrieved, answer=None, latency_ms=None,
                 cost_usd=None, error=None) -> str           # -> id записи журнала
set_feedback(conn, query_id, verdict, comment=None) -> bool  # False — записи нет
```

`upsert_embeddings` проверяет размерность до записи и никогда не создаёт чанки: вектор без
строки в `chunks` — рассинхрон кэша с базой, о нём сообщается числом обновлённых строк, а не
молчаливой вставкой. `search_vector` принимает уже посчитанный вектор (модель грузит
`ergovault.embeddings`), отсекает `embedding IS NULL` в SQL и возвращает `score` как косинусную
близость `1 - distance`, чтобы «больше = лучше» совпадало с fts-ветвью и RRF работал
с однонаправленными величинами.

## 6. Интерфейс поиска

```python
class Hit(BaseModel):
    chunk: Chunk; source: Source; score: float; rank: int; components: dict[str, float]

def search(conn, query: str, k: int = 10, *, profile: Profile,
           mode: Literal["fts", "vector", "hybrid"] | None = None,
           settings: Settings | None = None, encode_query=None) -> list[Hit]: ...
```

Гибрид — RRF (`score = Σ 1/(60 + rank_i)`), веса в конфиге. `components` обязателен: без него
не отладить, почему имя собственное не нашлось. Профиль фильтрует по `visibility` **в SQL**, не
в Python: утечка pilot-контента в public — дефект уровня «стоп-релиз».

Полнотекстовая ветвь находит чанк по любому значимому слову вопроса; чанки со всеми словами
(`components["fts_all_terms"] = 1`) идут первыми. Для `simple` стоп-слова отбрасываются по
словарю `russian`. Во всех режимах в top-k попадает не больше одного чанка источника
(`search.MAX_CHUNKS_PER_SOURCE`). Оригинал и пересказ — разные источники и не схлопываются.
Обоснование и замеры — `eval/reports/retrieval-comparison.md`.

## 7. Контракт ответа

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

Программные (не модельные) гарантии, покрытые тестами:

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

### 7.1. Реализация (`answering.py`)

- `answer(conn, question, *, profile, mode=None, k=None)` ищет фрагменты и вызывает
  `answer_hits(question, hits, *, sources, retrieval_mode)`; вторая функция — вход для оценки.
  `sources` — каталог `db.list_sources(conn, profile)`: пересказ получает `speaker_org`,
  `stance` и `reliability_note` оригинала, даже если оригинал не попал в выдачу.
  Одинаковая оговорка у нескольких процитированных источников (обычно оригинал и
  пересказ) выводится одним `reliability_note` со всеми их `source_ids`.
- Промпт — `prompts/{prompt_version}.system.md` и `.user.md`. Модель получает вопрос и
  нумерованные фрагменты (название, тип, язык, дата публикации, длительность, путь
  заголовков, текст) и возвращает `{"answer_markdown", "cited", "refused"}`. URL,
  `speaker_org`, `stance`, `reliability_note` и `derived` модели не передаются.
  Язык, дата и длительность нужны для вопросов о самих материалах (категория F
  проверочного набора): эти сведения лежат в полях `Source`, а не в тексте чанка.
  Отсутствующее значение передаётся как «не указана» — заглушка корпуса (§2) не
  заменяется правдоподобным числом.
- `Citation.n` — номер фрагмента в промпте. Номера в тексте и `cited` совпадают и лежат
  в диапазоне переданных фрагментов; ответ без ссылок допустим только при отказе.
  Цитата «…» со ссылкой сразу после неё обязана найтись в тексте, названии или заголовках
  именно этих фрагментов (без учёта регистра, ё/е, пробелов, markdown и вида тире;
  многоточие — пропуск). `Citation.quote` выбирает код: предложение фрагмента, ближайшее
  к тексту ответа у ссылки, обрезанное до 300 символов.
- Нехватка данных — `AnswerPayload(refused=True)` с объяснением и `no_data`; при пустой
  выдаче поиска модель не вызывается. Сбой модели (`LLMError`) и ответ, не прошедший
  проверку (`AnswerValidationError` с `raw_text` и `meta` для журнала), — исключения.
- `meta`: `corpus_version`, `retrieval_mode`, `prompt_version`, `prompt_hash`, `model`,
  `llm_backend`, `latency_ms` (вся функция), `llm_latency_ms`, `cost_usd` (расчёт по токенам),
  `raw_cost_usd` (значение бэкенда), токены; у `answer` ещё `profile` и `search_latency_ms`.

## 8. HTTP API

```
POST /api/ask          {question, profile?, retrieval_mode?} -> AnswerPayload
GET  /api/sources      -> [Source]   (public-профиль по умолчанию)
GET  /api/chunks/{id}  -> {chunk, source}
POST /api/feedback     {query_id, verdict: "useful"|"wrong", comment?} -> 204
GET  /api/eval/summary -> результаты последнего прогона (фронтенд не использует)
GET  /healthz          -> {status, corpus_version, chunks, sources}
```

Ошибки: `{"error": {"code": str, "message": str}}`, коды `bad_request|not_found|upstream|internal`.

Формат ошибки един для всех ответов, включая ответы самого фреймворка (неизвестный адрес,
метод, нечитаемое тело): наружу выходит код §8 и сообщение на русском, traceback и текст
исключения остаются в логе сервера.

`/api/ask` возвращает `meta.query_id` — id записи `query_log`, по нему сохраняется отзыв.
В журнал пишется и удачный ответ, и сбой генерации (`error`): без второго журнал показывал
бы качество выше фактического. `POST /api/feedback` принимает необязательный `comment`
(поле `feedback_comment`, миграция 002).

Параметр `profile` (тело `/api/ask`, query-параметр `/api/sources` и `/api/chunks/{id}`)
публичный инстанс (`EV_PROFILE=public`) **не принимает вообще** — запрос с ним отклоняется
как `bad_request`. Видимость фильтруется в SQL: закрытый источник недоступен ни через поиск,
ни через каталог, ни по прямой ссылке на чанк. `chunk_id` содержит `#` и в URL кодируется.

`/api/eval/summary` — одна схема на два состояния:

```python
class EvalRunSummary(BaseModel):    # метрика None = не измерена, это не ноль
    label: str; mode: str; k: int; n_questions: int; title: str
    recall_at_k: float | None; mrr: float | None
    recall_note: str                # почему метрики нет: «неприменимо: поиска нет»
    refusal_precision: float | None; refusal_recall: float | None
    must_contain_rate: float | None; grounded_rate: float | None
    median_latency_ms: float | None; cost_usd_per_question: float | None

class EvalJudgeSummary(BaseModel):  # проверка судьи; available=False — разметки не хватает
    available: bool; message: str; n_labeled: int
    agreement: float | None; cohen_kappa: float | None; self_consistency: float | None

class EvalSummary(BaseModel):
    available: bool                 # False — прогонов ещё не было, runs пуст
    message: str                    # готовая формулировка для экрана
    generated_at: str | None; corpus_version: str | None; prompt_version: str | None
    sample_size_caveat: str         # оговорка о размере выборки, приходит всегда
    runs: list[EvalRunSummary]
    judge: EvalJudgeSummary | None  # None — судья не проверялся
    limitations: list[str]          # оговорки прогона, показываются рядом с числами
    report_path: str | None         # путь к полному отчёту в репозитории
```

Источник данных — `eval/reports/summary.json`; его пишет T-30. Файла нет — `available=false`
и явное «прогонов не было»: отсутствие измерений не показывается нулями.

`label` — вариант сравнения (`no_search`, `fts`, `rag`, `full_corpus`), а не режим поиска:
`recall_at_k` и `mrr` приходят только у вариантов с поиском, у остальных они `None`
с объяснением в `recall_note`. `grounded_rate` приходит `None`, пока `judge.available`
ложно: подтверждённость не публикуется без согласия судьи с ручной разметкой.

## 9. Конфигурация

`.env` / `settings.py` (pydantic-settings, префикс `EV_`):
`EV_PROFILE=public|pilot`, `EV_DATABASE_URL`, `EV_EMBED_MODEL=BAAI/bge-m3`,
`EV_GEN_MODEL=claude-opus-5`, `EV_JUDGE_MODEL=claude-sonnet-5`, `EV_RETRIEVAL_MODE=hybrid`,
`EV_TOP_K=8`, `EV_CORPUS_VERSION=v1`, `EV_PROMPT_VERSION=p1`,
`EV_LLM_BACKEND=cli|api`, `EV_ANTHROPIC_API_KEY` (опционален),
`EV_CORS_ORIGINS` — адреса фронтенда через запятую (по умолчанию `http://localhost:5173`
и `http://127.0.0.1:5173`).

### Доступ к модели: подписка, а не ключ

Основной режим — `EV_LLM_BACKEND=cli`: генерация идёт через Claude Code CLI в headless-режиме
по подписке, платный API-ключ не нужен. Весь код ходит в модель **только** через
`ergovault.llm.LLMProvider`; прямые вызовы SDK из других модулей запрещены.

Для отчёта T-30:

- `billed_cost_usd` рассчитывается по полям usage. Перед экспериментом проверить состав
  токенов и тарифы; расчётную стоимость, значение CLI и фактические списания обозначать отдельно.
- Текущий CLI-бэкенд не задаёт temperature и игнорирует `cache_system`.
  Измерять разброс и фактическое использование кэша; не заявлять детерминизм,
  исключённый оверхед или одинаковую стоимость служебной части без проверки.
- CLI-бэкенд передаёт сообщение через stdin (командная строка Windows ограничена 32 767
  символами) и отключает все встроенные инструменты (`--tools ""`). Со списком запретов
  модель получала новые инструменты CLI, делала несколько ходов за вызов, и `cache_read`
  доходил до ~150 тыс. токенов; прогоны до 11.09.2026 со старым флагом несопоставимы по стоимости.

Индексация в обоих режимах работает без единого внешнего ключа — эмбеддинги локальные.
Это требование раздела 9 спеки, и оно не зависит от способа доступа к генератору.

## 10. Версионирование

`corpus_version`, `prompt_version`, `retrieval_mode`, `model` пишутся в каждую строку
`query_log` и в каждый прогон eval. Результат эксперимента, не привязанный к этим четырём,
считается недействительным и в отчёт не идёт.
