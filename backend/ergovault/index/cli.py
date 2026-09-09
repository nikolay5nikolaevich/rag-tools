"""`ev db …` — миграции и загрузка индекса в базу. Контракт §0: импорт модуля не должен
подключаться к БД, только объявлять команды."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import psycopg
import typer

app = typer.Typer(add_completion=False, help="Миграции и загрузка в базу")


@app.command("upgrade")
def upgrade_cmd() -> None:
    """Накатывает пронумерованные SQL-миграции из backend/migrations/."""
    from ergovault.index.migrate import upgrade
    from ergovault.settings import get_settings

    applied = upgrade(get_settings().database_url)
    if applied:
        typer.echo("применены: " + ", ".join(applied))
    else:
        typer.echo("схема уже актуальна, ничего не применено")


@app.command("load")
def load_cmd(
    index_dir: Annotated[
        Path, typer.Argument(help="Каталог с sources.jsonl и chunks.jsonl (выход ev ingest)")
    ],
) -> None:
    """Загружает sources.jsonl и chunks.jsonl из index_dir в базу."""
    from ergovault.index.repository import upsert_chunks, upsert_sources
    from ergovault.models import Chunk, Source
    from ergovault.settings import get_settings

    settings = get_settings()
    sources = [
        Source(**json.loads(line))
        for line in (index_dir / "sources.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    chunks = [
        Chunk(**json.loads(line))
        for line in (index_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    with psycopg.connect(settings.database_url) as conn:
        upsert_sources(conn, sources, corpus_version=settings.corpus_version)
        upsert_chunks(conn, chunks, corpus_version=settings.corpus_version)

    typer.echo(f"sources: {len(sources)}, chunks: {len(chunks)}")


if __name__ == "__main__":
    app()
