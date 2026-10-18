"""Команды терминала: параметры, вызовы прикладных функций, сообщения об ошибках."""

from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(
    name="ev",
    no_args_is_help=True,
    add_completion=False,
    help="ErgoVault: подготовка поисковой базы и оценка качества.",
)
db_app = typer.Typer(help="Миграции и загрузка JSONL")
eval_app = typer.Typer(help="Оценка качества")
app.add_typer(db_app, name="db")
app.add_typer(eval_app, name="eval")


def _run(action, *args, **kwargs):
    """Ошибка операции даёт ненулевой код завершения и понятное сообщение."""
    import psycopg

    try:
        return action(*args, **kwargs)
    except (ValueError, OSError, RuntimeError, psycopg.Error) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc


@app.command()
def prepare(
    batch: Annotated[int, typer.Option(min=1, help="Размер батча векторизации")] = 32,
    force: Annotated[bool, typer.Option(help="Пересчитать все векторы")] = False,
):
    """Подготовить базу целиком: миграции, корпус, загрузка, эмбеддинги."""
    from ergovault.indexing import prepare as prepare_index

    _run(prepare_index, batch=batch, force=force)


@app.command()
def ingest(
    profile: Annotated[str, typer.Option(help="public или pilot")] = "public",
    report: Annotated[bool, typer.Option(help="Напечатать отчёт")] = False,
    out: Annotated[Path | None, typer.Option(help="Каталог результата")] = None,
):
    """Прочитать корпус и сохранить sources.jsonl и chunks.jsonl."""
    from ergovault.ingest import ingest_corpus
    from ergovault.settings import get_settings

    if profile not in ("public", "pilot"):
        raise typer.BadParameter("ожидалось public или pilot", param_hint="--profile")
    settings = get_settings()
    rendered = _run(ingest_corpus, settings.corpus_dir, out or settings.index_dir, profile)
    if report:
        typer.echo(rendered, nl=False)


@app.command()
def embed(
    batch: Annotated[int, typer.Option(min=1, help="Размер батча векторизации")] = 32,
    force: Annotated[bool, typer.Option(help="Пересчитать все векторы")] = False,
    write_db: Annotated[bool, typer.Option("--write-db/--no-write-db")] = True,
):
    """Векторизовать чанки, сохранить кэш и записать векторы в БД."""
    from ergovault.indexing import embed as embed_index

    _run(embed_index, batch=batch, force=force, write_db=write_db)


@db_app.command("upgrade")
def db_upgrade():
    """Применить SQL-миграции."""
    from ergovault.db import upgrade
    from ergovault.settings import get_settings

    applied = _run(upgrade, get_settings().database_url)
    typer.echo("применены: " + ", ".join(applied) if applied else "схема уже актуальна")


@db_app.command("load")
def db_load(index_dir: Annotated[Path | None, typer.Argument()] = None):
    """Загрузить JSONL; по умолчанию — индекс текущего корпуса."""
    from ergovault.db import load_index
    from ergovault.settings import get_settings

    sources, chunks = _run(load_index, index_dir or get_settings().index_dir)
    typer.echo(f"sources: {sources}, chunks: {chunks}")


@eval_app.command("retrieval")
def evaluate_retrieval(
    mode: Annotated[str, typer.Option(help="fts | vector | hybrid")] = "hybrid",
    k: Annotated[int, typer.Option(min=1, help="Число фрагментов")] = 5,
    include_holdout: Annotated[bool, typer.Option(help="Открыть отложенный набор")] = False,
    profile: Annotated[str, typer.Option(help="public | pilot")] = "public",
    label: Annotated[str | None, typer.Option(help="Метка отчёта")] = None,
):
    """Прогнать проверочные вопросы через поиск и записать метрики."""
    from ergovault.eval.evaluate import evaluate

    _run(evaluate, mode=mode, k=k, include_holdout=include_holdout, profile=profile, label=label)


def _split_list(value: str | None) -> list[str]:
    return [item.strip() for item in (value or "").split(",") if item.strip()]


@eval_app.command("compare")
def evaluate_compare(
    variants: Annotated[str, typer.Option(help="Через запятую: no_search,fts,rag,full_corpus")] = (
        "no_search,fts,rag,full_corpus"
    ),
    k: Annotated[int | None, typer.Option(min=1, help="Фрагментов вариантам с поиском")] = None,
    repeats: Annotated[int, typer.Option(min=1, help="Повторов каждого вопроса")] = 1,
    include_holdout: Annotated[bool, typer.Option(help="Открыть отложенный набор")] = False,
    only: Annotated[str | None, typer.Option(help="Только эти id вопросов, через запятую")] = None,
    limit: Annotated[int | None, typer.Option(min=1, help="Взять первые N вопросов")] = None,
    label: Annotated[str, typer.Option(help="Метка файла прогона")] = "compare",
    probe: Annotated[bool, typer.Option(help="Замерить холодный и тёплый вызов")] = True,
    judge: Annotated[bool, typer.Option(help="Считать подтверждённость судьёй")] = False,
    retry: Annotated[
        Path | None, typer.Option(help="Доделать прогон: повторить только сорвавшиеся вызовы")
    ] = None,
):
    """Сравнить четыре подхода и записать отчёт, summary.json и сырые строки."""
    from ergovault.eval.experiment import run_experiment

    _run(
        run_experiment,
        variants=_split_list(variants),
        k=k,
        repeats=repeats,
        include_holdout=include_holdout,
        only=_split_list(only),
        limit=limit,
        label=label,
        with_probe=probe,
        with_judge=judge,
        retry=retry,
        echo=typer.echo,
    )


@eval_app.command("report")
def evaluate_report(
    run: Annotated[Path, typer.Argument(help="Файл прогона eval/runs/*.jsonl")],
    include_holdout: Annotated[bool, typer.Option(help="В прогоне есть holdout")] = False,
):
    """Пересобрать отчёт и summary.json из готового прогона, не вызывая модель."""
    from ergovault.eval.experiment import FINAL_REPORT_PATH, rebuild, render_stdout

    comparison = _run(rebuild, run, include_holdout=include_holdout)
    typer.echo(render_stdout(comparison))
    typer.echo(f"\nотчёт: {FINAL_REPORT_PATH}")


@eval_app.command("judge-pairs")
def evaluate_judge_pairs(
    run: Annotated[
        Path, typer.Argument(help="Файл прогона, из ответов которого брать утверждения")
    ],
    target: Annotated[int, typer.Option(min=10, help="Сколько пар собрать")] = 48,
):
    """Собрать пары «утверждение — фрагмент» для ручной разметки судьи."""
    from ergovault.eval.experiment import load_rows, refresh_judge_pairs
    from ergovault.eval.judge import MIN_CALIBRATION_PAIRS, PAIRS_PATH

    rows = _run(load_rows, run)
    total, labeled = _run(refresh_judge_pairs, rows, target=target)
    typer.echo(f"пар: {total}, из них размечено человеком: {labeled}")
    typer.echo(f"файл: {PAIRS_PATH}")
    if labeled < MIN_CALIBRATION_PAIRS:
        typer.echo(
            f"проставьте human_verdict хотя бы у {MIN_CALIBRATION_PAIRS} пар: "
            "supported | partially | not_supported. Разметка — работа человека."
        )


@eval_app.command("calibrate")
def evaluate_calibrate(
    repeats: Annotated[int, typer.Option(min=1, help="Повторов каждой пары")] = 3,
):
    """Проверить судью на парах: устойчивость, согласие с человеком и между конфигурациями."""
    from ergovault.eval.judge import run_calibration

    _run(run_calibration, repeats=repeats, echo=typer.echo)


@eval_app.command("probe")
def evaluate_probe(
    repeats: Annotated[int, typer.Option(min=2, help="Сколько одинаковых вызовов подряд")] = 3,
    question: Annotated[str, typer.Option(help="Вопрос для замера")] = (
        "Как готовиться к участию в международной выставке?"
    ),
):
    """Замерить холодный и тёплый вызов и проверить, работает ли кэш промпта."""
    import psycopg

    from ergovault.eval.experiment import probe_cache
    from ergovault.settings import get_settings

    settings = get_settings()

    def measure():
        with psycopg.connect(settings.database_url) as conn:
            return probe_cache(
                conn,
                question,
                model=settings.gen_model,
                repeats=repeats,
                profile=settings.profile,
                k=settings.top_k,
                settings=settings,
            )

    probe = _run(measure)
    names = {0: "почти пустой", 1: "холодный"}
    for call in [*probe.calls, *([probe.minimal_call] if probe.minimal_call else [])]:
        name = names.get(call.n, f"тёплый #{call.n - 1}" if call.n > 1 else f"размер {call.n}")
        typer.echo(
            f"{name:<14} {call.latency_ms / 1000:>6.1f} с  символов {call.prompt_chars:>7} "
            f"всего токенов {call.total_prompt_tokens:>7} (cache_read {call.cache_read_tokens:>7}) "
            f"расчёт ${call.cost_usd:.4f}"
        )
    if probe.tokens_per_char is not None:
        typer.echo(
            f"по точкам разного размера: {probe.tokens_per_char:.2f} токена на символ нашего "
            f"промпта, помимо него в запрос входит ещё ~{probe.service_tokens:,.0f} токенов"
        )
    typer.echo(probe.note)


@app.command()
def version():
    """Показать версии и выбранные режимы."""
    from ergovault.settings import get_settings

    s = get_settings()
    typer.echo(
        f"corpus={s.corpus_version} prompt={s.prompt_version} "
        f"profile={s.profile} retrieval={s.retrieval_mode} llm={s.llm_backend}"
    )


if __name__ == "__main__":
    app()
