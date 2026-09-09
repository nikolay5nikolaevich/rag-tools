"""Корневая точка входа `ev`. Владелец — T-00 (тех-лид), исполнители тикетов её не меняют.

Подкоманды подключаются лениво: каждый пакет объявляет собственное Typer-приложение
в своём `cli.py`, а корень лишь пытается его импортировать. Пакет, который ещё не написан,
просто отсутствует в выводе `ev --help` и не ломает остальные команды.

Так пять параллельных исполнителей не редактируют один общий файл: чтобы добавить свою
команду, тикету достаточно создать `ergovault/<пакет>/cli.py` с объектом `app`.

    ergovault/ingest/cli.py  -> ev ingest ...   (T-10)
    ergovault/index/cli.py   -> ev db ...       (T-11)
    ergovault/embed/cli.py   -> ev embed ...    (T-12)
    ergovault/eval/cli.py    -> ev eval ...     (T-13)
    ergovault/api/cli.py     -> ev serve ...    (T-22)

Требования к модулю подкоманды:
  * объект называется `app` и является `typer.Typer()`;
  * импорт модуля не должен ничего делать, кроме объявления команд — ни подключения
    к базе, ни загрузки модели: `ev --help` обязан работать без Postgres и без весов.
"""

from __future__ import annotations

import importlib
import typer

app = typer.Typer(
    name="ev",
    help="ErgoVault RAG — обработка корпуса, индексация, поиск, оценка.",
    no_args_is_help=True,
    add_completion=False,
)

# (модуль, имя команды, краткое описание) — порядок задаёт вывод `ev --help`.
_SUBCOMMANDS: list[tuple[str, str, str]] = [
    ("ergovault.ingest.cli", "ingest", "Разбор корпуса в источники и фрагменты"),
    ("ergovault.index.cli", "db", "Миграции и загрузка в базу"),
    ("ergovault.embed.cli", "embed", "Векторизация фрагментов"),
    ("ergovault.eval.cli", "eval", "Прогон проверочного набора и метрики"),
    ("ergovault.api.cli", "serve", "Запуск HTTP API"),
]


def _register() -> list[str]:
    """Подключает те подкоманды, чьи модули уже существуют. Возвращает пропущенные."""
    missing: list[str] = []
    for module_path, name, help_text in _SUBCOMMANDS:
        try:
            module = importlib.import_module(module_path)
        except ModuleNotFoundError:
            missing.append(name)
            continue
        sub = getattr(module, "app", None)
        if sub is None:
            missing.append(name)
            continue
        app.add_typer(sub, name=name, help=help_text)
    return missing


MISSING = _register()


@app.command()
def version() -> None:
    """Версии корпуса, промпта и профиля — то, к чему привязан любой результат."""
    from ergovault.settings import get_settings

    s = get_settings()
    typer.echo(
        f"corpus={s.corpus_version} prompt={s.prompt_version} "
        f"profile={s.profile} retrieval={s.retrieval_mode} llm={s.llm_backend}"
    )
    if MISSING:
        typer.echo(f"ещё не реализованы: {', '.join(MISSING)}")


if __name__ == "__main__":
    app()
