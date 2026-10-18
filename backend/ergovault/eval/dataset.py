"""Загрузка проверочных вопросов; формат описан в eval/golden/README.md. Разметка не изменяется."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from ergovault.settings import REPO_ROOT

Category = Literal["A", "B", "C", "D", "E", "F"]

DEFAULT_GOLDEN_DIR = REPO_ROOT / "eval" / "golden"
HOLDOUT_FILENAME = "holdout.yml"


class GoldenQuestion(BaseModel):
    id: str
    category: Category
    question: str
    answerable: bool
    expected_source_ids: list[str] = Field(default_factory=list)
    must_contain: list[str] = Field(default_factory=list)
    must_not_contain: list[str] = Field(default_factory=list)
    expected_limitations: list[str] = Field(default_factory=list)
    notes: str = ""
    holdout: bool = False


class DuplicateGoldenIdError(ValueError):
    pass


def load_golden_set(
    golden_dir: Path = DEFAULT_GOLDEN_DIR, *, include_holdout: bool = False
) -> list[GoldenQuestion]:
    """Читает `*.yml` из `golden_dir` в список `GoldenQuestion`.

    Без `include_holdout=True` файл `holdout.yml` не открывается вовсе, а вопрос с
    `holdout: true` в любом другом файле всё равно отфильтровывается — двойная защита
    от случайной утечки holdout в рабочий прогон (спека: "Loading без явного
    --include-holdout запрещена и покрыта тестом").
    """
    questions: list[GoldenQuestion] = []
    for path in sorted(golden_dir.glob("*.yml")):
        if path.name == HOLDOUT_FILENAME and not include_holdout:
            continue
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
        for item in raw:
            q = GoldenQuestion(**item)
            if q.holdout and not include_holdout:
                continue
            questions.append(q)

    _check_no_duplicate_ids(questions)
    return questions


def _check_no_duplicate_ids(questions: list[GoldenQuestion]) -> None:
    seen: set[str] = set()
    for q in questions:
        if q.id in seen:
            raise DuplicateGoldenIdError(f"дублирующийся id в golden set: {q.id}")
        seen.add(q.id)
