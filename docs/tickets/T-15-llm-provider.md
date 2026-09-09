# T-15 — Абстракция LLM-провайдера: подписка вместо API-ключа

**Пакет:** `backend/ergovault/llm/`. **Зависит от:** T-00. Идёт в волне 1, параллельно всем.
**Блокирует:** T-21, T-23, T-30 — поэтому приоритет высокий.

## Зачем

Проект должен работать **без платного API-ключа**: генерация идёт через подписку Claude
(Claude Code CLI в headless-режиме). Ключ остаётся опциональным путём — для чистых замеров
стоимости в T-30 и для запуска рецензентом, у которого нет подписки.

Индексация от этого не зависит: эмбеддинги локальные, ключ ей не нужен ни в одном варианте.

## Интерфейс

```python
class LLMResponse(BaseModel):
    text: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    latency_ms: int
    backend: Literal["cli", "api"]
    model: str
    raw_cost_usd: float | None   # то, что сообщил бэкенд; для cli включает служебный оверхед
    billed_cost_usd: float       # расчёт по прайсу от НАШИХ токенов, см. §«Стоимость»

class LLMProvider(Protocol):
    def complete(self, *, system: str, user: str, model: str,
                 cache_system: bool = False, max_tokens: int = 4000) -> LLMResponse: ...
```

Два бэкенда, выбор через `EV_LLM_BACKEND=cli|api`, по умолчанию `cli`.

## Бэкенд `cli` (подписка, основной)

Вызов подпроцессом, проверенная рабочая команда:

```
claude -p <user> --output-format json --model <model>
  --system-prompt <system>
  --exclude-dynamic-system-prompt-sections
  --disable-slash-commands --strict-mcp-config --setting-sources ""
  --disallowedTools Bash Read Write Edit Glob Grep WebSearch WebFetch Task NotebookEdit Artifact Skill ToolSearch
```

Ответ — JSON-массив; нужный элемент `type == "result"`, поля `result`, `usage`,
`total_cost_usd`, `duration_api_ms`. Парсить именно этот элемент, не последний по порядку.

**Замерено на этой машине (Haiku 4.5, тривиальный запрос):**

| Конфигурация | Служебный оверхед |
|---|---|
| `claude -p` без флагов | 34 206 токенов |
| + `--system-prompt --exclude-dynamic-system-prompt-sections --setting-sources ""` | 26 577 |
| + `--disable-slash-commands --disallowedTools <всё>` | **16 819** |

Ниже ~17k не опускается: это базовый системный промпт Claude Code и определения оставшихся
инструментов. Флаг `--bare` оверхед снимает, но вместе с ним отваливается авторизация
(`Not logged in`) — не использовать.

Требования к реализации:
- таймаут и ретрай с backoff; `claude` иногда стартует секунды;
- ошибку `Not logged in` распознавать отдельно и объяснять пользователю, что делать;
- на Windows вызывать без `shell=True`, аргументы списком (в путях есть кириллица и пробелы);
- **temperature задать нельзя** — в headless-режиме её нет. Последствия для судьи — ниже.

## Бэкенд `api` (ключ, опциональный)

Прямой Anthropic SDK, prompt caching через `cache_control` на системном промпте и на корпусе.
Включается только при заданном `EV_ANTHROPIC_API_KEY`. Отсутствие ключа — не ошибка,
а переключение на `cli`.

## Стоимость: как считать честно

`total_cost_usd` из CLI **в отчёт не идёт**: в него входят те самые ~17k служебных токенов,
которых в чистом API-вызове не было бы. В `billed_cost_usd` пишется расчёт по публичному
прайсу от токенов нашего промпта и ответа (`input_tokens`, `output_tokens`, кэш — по своим
тарифам). Прайс — в `backend/ergovault/llm/pricing.py`, одной таблицей, с датой актуальности.

В отчёте T-30 указывается прямо: стоимость расчётная по прайсу, а не по счёту; служебный
оверхед CLI вычтен; он одинаков во всех сравниваемых вариантах, поэтому сравнение вариантов
между собой корректно, а абсолютные цифры — нижняя оценка.

## Тесты

- парсинг `result`-элемента на зафиксированном JSON-ответе (фикстура, без вызова сети);
- `billed_cost_usd` считается от наших токенов и не равен `raw_cost_usd`;
- отсутствие ключа → бэкенд `cli`, а не падение;
- один живой smoke-вызов, помеченный `@pytest.mark.live`, не в основном прогоне.

## Границы

Не встраивать вызовы CLI напрямую в T-21/T-23/T-30 — они ходят только через этот интерфейс.
Не логировать содержимое системного промпта в journal целиком: там версия, а не текст.
