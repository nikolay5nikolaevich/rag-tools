# T-10 — Ingest: парсинг корпуса, чанкинг, метаданные

**Пакет:** `backend/ergovault/ingest/` — только он. Чужие пакеты не трогать.
**Читать перед стартом:** `docs/CONTRACTS.md` §1–4 (нормативно), `docs/PLAN.md`.
**Зависит от:** T-00 (модели уже в `ergovault/models.py`), T-02 (корпус в `corpus/v1/`).

## Задача

Превратить снапшот `corpus/v1/` в `Source[]` и `Chunk[]` по правилам контракта §3–4 и выгрузить
в `corpus/v1/index/sources.jsonl` и `chunks.jsonl`.

## Что сделать

1. `ergovault/ingest/frontmatter.py` — разбор YAML-frontmatter. Ключи в корпусе неоднородны:
   встречаются `duration` и `duration_minutes`, `source` и `source_video`, `raw`, `channel`,
   `published`, `video_number`, `status`, `source_note`, `playlist_url`, `industry`, `country`.
   Маппинг в `Source` — явной таблицей, а не «что нашлось». Неизвестный ключ — в лог, не в модель.
2. `ergovault/ingest/markdown.py` — чанкинг по §3. Отдельно и аккуратно:
   - нумерованные пункты в «Основные тезисы» → `atomic=True` (это ядро качества поиска:
     каждый пункт корпуса — законченное утверждение);
   - `> Примечание по достоверности: …` → `source.reliability_note`, из тела чанка удаляется;
   - Obsidian-ссылки разворачиваются, `![[...]]` вырезаются.
3. `ergovault/ingest/attribution.py` — `speaker_org`/`stance` по `corpus/v1/speaker_org.yml`,
   `derived`/`derived_of` по типу и `corpus/v1/derived_map.yml`. Никаких fuzzy-совпадений.
4. `ergovault/ingest/cli.py` — `ev ingest [--profile public|pilot] [--report]`.
   `--report` печатает: число источников и чанков по типам, источники без `speaker_org`,
   источники без `url`, распределение длин чанков, найденные `reliability_note`.
5. Тесты `backend/tests/test_ingest.py`:
   - все инварианты из §1 контракта на реальном корпусе (не на выдуманных фикстурах);
   - 13 `video-source` и 13 `video-theses`, у каждого `video-theses` проставлен `derived_of`;
   - ни один чанк публичного профиля не содержит текста из `raw/`;
   - идемпотентность: два прогона дают побайтово одинаковый JSONL.

## Границы

- Не подключаться к БД, не считать эмбеддинги, не звать сеть. Ingest — чистая функция от файлов.
- Не «улучшать» текст корпуса: не переписывать, не исправлять опечатки, не дополнять.
- Не заполнять отсутствующие метаданные догадками — `None` и строка в отчёте.

## Готово, когда

`ev ingest --report` отрабатывает на `corpus/v1`, тесты зелёные, в отчёте явно перечислено, какие
источники остались без `speaker_org` и без `url` (это вход для тех-лида, а не ошибка).
