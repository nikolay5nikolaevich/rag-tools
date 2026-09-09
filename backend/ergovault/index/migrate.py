"""Раннер миграций: применяет пронумерованные SQL-файлы из backend/migrations/ по порядку.

Без Alembic — схема маленькая, миграции читает человек (docs/CONTRACTS.md, «Чего не делать»).
Идемпотентность обеспечивается таблицей `schema_migrations`: файл применяется один раз,
повторный `upgrade()` пропускает уже применённые.
"""

from __future__ import annotations

from pathlib import Path

import psycopg

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"

_ENSURE_TABLE = """
CREATE TABLE IF NOT EXISTS schema_migrations (
  filename text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now()
);
"""


def _migration_files(migrations_dir: Path) -> list[Path]:
    return sorted(migrations_dir.glob("*.sql"), key=lambda p: p.name)


def upgrade(database_url: str, migrations_dir: Path = MIGRATIONS_DIR) -> list[str]:
    """Применяет неприменённые миграции. Возвращает имена файлов, которые применил сейчас."""
    applied_now: list[str] = []
    with psycopg.connect(database_url, autocommit=False) as conn:
        with conn.cursor() as cur:
            cur.execute(_ENSURE_TABLE)
        conn.commit()

        for path in _migration_files(migrations_dir):
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM schema_migrations WHERE filename = %s", (path.name,))
                if cur.fetchone() is not None:
                    continue
                cur.execute(path.read_text(encoding="utf-8"))
                cur.execute(
                    "INSERT INTO schema_migrations (filename) VALUES (%s)", (path.name,)
                )
            conn.commit()
            applied_now.append(path.name)
    return applied_now
