from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from ergovault.llm import get_provider
from ergovault.llm.cli_backend import (
    CliLLMProvider,
    LLMError,
    NotLoggedInError,
    parse_result_item,
)
from ergovault.llm.pricing import billed_cost_usd
from ergovault.settings import get_settings

FIXTURE = Path(__file__).parent / "fixtures" / "claude_cli_result.json"
REPO_ROOT = Path(__file__).parents[2]


@pytest.fixture(autouse=True)
def _clear_settings_cache(monkeypatch: pytest.MonkeyPatch):
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


class _FakeCompletedProcess:
    def __init__(self, stdout: str = "", stderr: str = "", returncode: int = 0) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


# --- get_provider() ---------------------------------------------------------


def test_get_provider_without_key_returns_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EV_ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("EV_LLM_BACKEND", raising=False)
    provider = get_provider()
    assert isinstance(provider, CliLLMProvider)


def test_get_provider_api_backend_without_key_falls_back_to_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EV_LLM_BACKEND", "api")
    monkeypatch.delenv("EV_ANTHROPIC_API_KEY", raising=False)
    provider = get_provider()
    assert isinstance(provider, CliLLMProvider)


def test_get_provider_api_backend_with_key_returns_api(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EV_LLM_BACKEND", "api")
    monkeypatch.setenv("EV_ANTHROPIC_API_KEY", "sk-test-not-real")
    provider = get_provider()
    from ergovault.llm.api_backend import ApiLLMProvider

    assert isinstance(provider, ApiLLMProvider)


# --- разбор ответа cli -------------------------------------------------------


def test_parse_result_item_picks_result_not_last() -> None:
    stdout = FIXTURE.read_text(encoding="utf-8")
    items = json.loads(stdout)
    assert items[-1]["type"] != "result", "фикстура должна содержать элемент после result"

    result = parse_result_item(stdout)
    assert result["type"] == "result"
    assert result["result"] == "да"


def test_parse_result_item_raises_without_result_element() -> None:
    with pytest.raises(LLMError):
        parse_result_item(json.dumps([{"type": "system", "subtype": "init"}]))


# --- billed_cost_usd vs raw_cost_usd ----------------------------------------


def test_billed_cost_usd_not_equal_to_raw_cost_usd() -> None:
    fixture_item = json.loads(FIXTURE.read_text(encoding="utf-8"))[2]
    usage = fixture_item["usage"]

    billed = billed_cost_usd(
        model="claude-haiku-4-5-20251001",
        input_tokens=usage["input_tokens"],
        output_tokens=usage["output_tokens"],
        cache_read_tokens=usage["cache_read_input_tokens"],
        cache_write_tokens=usage["cache_creation_input_tokens"],
    )

    assert billed != fixture_item["total_cost_usd"]
    assert billed > 0


# --- ошибка "Not logged in" ---------------------------------------------------


def test_not_logged_in_detected_distinctly(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*args, **kwargs):
        return _FakeCompletedProcess(
            stdout="", stderr="Error: Not logged in. Run `claude login`.", returncode=1
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    provider = CliLLMProvider(max_retries=1)

    with pytest.raises(NotLoggedInError) as exc_info:
        provider.complete(system="сист", user="привет", model="claude-haiku-4-5-20251001")
    assert "claude login" in str(exc_info.value)


# --- вызов подпроцессом: список аргументов, без shell=True --------------------


def test_cli_backend_calls_subprocess_without_shell(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return _FakeCompletedProcess(stdout=FIXTURE.read_text(encoding="utf-8"), returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    provider = CliLLMProvider()
    provider.complete(system="сист ёж", user="вопрос с пробелами", model="claude-haiku-4-5-20251001")

    assert isinstance(captured["command"], list)
    assert all(isinstance(part, str) for part in captured["command"])
    assert captured["kwargs"].get("shell", False) is False


def test_cli_backend_parses_fixture_into_llmresponse(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(command, **kwargs):
        return _FakeCompletedProcess(stdout=FIXTURE.read_text(encoding="utf-8"), returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    provider = CliLLMProvider()
    response = provider.complete(
        system="сист", user="вопрос", model="claude-haiku-4-5-20251001"
    )

    assert response.text == "да"
    assert response.input_tokens == 10
    assert response.output_tokens == 79
    assert response.cache_read_tokens == 16119
    assert response.cache_write_tokens == 695
    assert response.backend == "cli"
    assert response.raw_cost_usd == 0.0043889
    assert response.billed_cost_usd != response.raw_cost_usd


# --- изоляция пакета: прямых обращений к SDK вне ergovault.llm нет ------------


def test_no_direct_anthropic_sdk_calls_outside_llm_package() -> None:
    ergovault_dir = REPO_ROOT / "backend" / "ergovault"
    offenders = []
    for path in ergovault_dir.rglob("*.py"):
        rel = path.relative_to(ergovault_dir)
        if rel.parts[0] == "llm":
            continue
        text = path.read_text(encoding="utf-8")
        if "import anthropic" in text:
            offenders.append(str(rel))
    assert offenders == [], f"прямой импорт anthropic вне ergovault.llm: {offenders}"


# --- живой вызов через подписку ----------------------------------------------


@pytest.mark.live
def test_live_cli_call_returns_text_tokens_latency() -> None:
    provider = CliLLMProvider()
    response = provider.complete(
        system="Ты — тестовый ассистент.",
        user="Ответь одним словом: да",
        model="claude-haiku-4-5-20251001",
    )
    assert response.text.strip() != ""
    assert response.input_tokens > 0
    assert response.output_tokens > 0
    assert response.latency_ms > 0
    assert response.backend == "cli"
