"""Бэкенд `cli`: генерация через Claude Code CLI по подписке, без платного ключа.

Команда и флаги минимизации служебного оверхеда — docs/tickets/T-15-llm-provider.md;
замеры на этой машине — docs/tickets/T-15-overhead.md. Аргументы передаются списком,
без `shell=True`: в путях корпуса есть кириллица и пробелы (требование T-15 к Windows).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time

from ergovault.llm.base import LLMResponse
from ergovault.llm.pricing import billed_cost_usd

# Список должен буквально совпадать с флагом --disallowedTools из тикета T-15:
# cli-провайдер отвечает только текстом, инструменты Claude Code ему не нужны.
DISALLOWED_TOOLS = [
    "Bash", "Read", "Write", "Edit", "Glob", "Grep", "WebSearch", "WebFetch",
    "Task", "NotebookEdit", "Artifact", "Skill", "ToolSearch",
]

DEFAULT_TIMEOUT_S = 120.0
MAX_RETRIES = 3
BACKOFF_BASE_S = 2.0

_NOT_LOGGED_IN_MARKER = "Not logged in"


class LLMError(RuntimeError):
    """Общая ошибка вызова модели через cli-бэкенд."""


class NotLoggedInError(LLMError):
    """`claude` CLI не авторизован в этой системе."""

    def __init__(self) -> None:
        super().__init__(
            "claude CLI не авторизован (Not logged in). "
            "Выполни `claude login` в терминале и повтори запуск."
        )


def _build_command(*, user: str, system: str, model: str) -> list[str]:
    claude_path = shutil.which("claude") or "claude"
    return [
        claude_path,
        "-p", user,
        "--output-format", "json",
        "--model", model,
        "--system-prompt", system,
        "--exclude-dynamic-system-prompt-sections",
        "--disable-slash-commands",
        "--strict-mcp-config",
        "--setting-sources", "",
        "--disallowedTools", *DISALLOWED_TOOLS,
    ]


def parse_result_item(stdout: str) -> dict:
    """Разбирает JSON-массив ответа cli, возвращает элемент с type == 'result'.

    Элемент с type == 'result' не обязан быть последним в массиве (T-15, «Приёмка») —
    поэтому ищем по полю, а не берём items[-1].
    """
    try:
        items = json.loads(stdout)
    except json.JSONDecodeError as e:
        raise LLMError(f"claude CLI вернул не-JSON вывод: {stdout[:500]!r}") from e
    for item in items:
        if isinstance(item, dict) and item.get("type") == "result":
            return item
    raise LLMError("в ответе claude CLI нет элемента с type == 'result'")


class CliLLMProvider:
    """Провайдер через Claude Code CLI (headless), см. docs/CONTRACTS.md §9."""

    def __init__(
        self, *, timeout_s: float = DEFAULT_TIMEOUT_S, max_retries: int = MAX_RETRIES
    ) -> None:
        self._timeout_s = timeout_s
        self._max_retries = max_retries

    def complete(
        self,
        *,
        system: str,
        user: str,
        model: str,
        cache_system: bool = False,
        max_tokens: int = 4000,
    ) -> LLMResponse:
        # cache_system и max_tokens не используются: у headless-режима нет флагов
        # управления кэшем и лимитом вывода — параметры приняты ради единого
        # интерфейса с api-бэкендом (T-15).
        command = _build_command(user=user, system=system, model=model)

        last_error: Exception | None = None
        for attempt in range(self._max_retries):
            start = time.monotonic()
            try:
                proc = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    timeout=self._timeout_s,
                    shell=False,
                    check=False,
                )
            except subprocess.TimeoutExpired as e:
                last_error = e
                time.sleep(BACKOFF_BASE_S * (2**attempt))
                continue
            latency_ms = int((time.monotonic() - start) * 1000)

            combined_output = (proc.stdout or "") + (proc.stderr or "")
            if _NOT_LOGGED_IN_MARKER in combined_output:
                raise NotLoggedInError()

            if proc.returncode != 0:
                last_error = LLMError(
                    f"claude CLI завершился с кодом {proc.returncode}: "
                    f"{(proc.stderr or '')[:1000]}"
                )
                time.sleep(BACKOFF_BASE_S * (2**attempt))
                continue

            result = parse_result_item(proc.stdout)
            usage = result.get("usage", {})
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            cache_read = usage.get("cache_read_input_tokens", 0)
            cache_write = usage.get("cache_creation_input_tokens", 0)

            return LLMResponse(
                text=result.get("result", ""),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_tokens=cache_read,
                cache_write_tokens=cache_write,
                latency_ms=latency_ms,
                backend="cli",
                model=model,
                raw_cost_usd=result.get("total_cost_usd"),
                billed_cost_usd=billed_cost_usd(
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cache_read_tokens=cache_read,
                    cache_write_tokens=cache_write,
                ),
            )

        raise LLMError(
            f"claude CLI не ответил после {self._max_retries} попыток"
        ) from last_error
