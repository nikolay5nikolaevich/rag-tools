"""Проверка проверочного набора: ссылки на источники, дубли, согласованность разметки.

Запускать после любой правки eval/golden/*.yml:
    python scripts/validate_golden.py

Ловит главную ошибку разметки — ссылку на source_id, которого в корпусе нет.
Такая разметка молча занижает recall и заставляет чинить исправно работающий поиск.
"""

from __future__ import annotations

import glob
import io
import json
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))

from ergovault.models import make_source_id  # noqa: E402


def main() -> int:
    manifest = json.loads((REPO / "corpus/v1/manifest.json").read_text(encoding="utf-8"))
    valid = {make_source_id(e["path"]) for e in manifest["files"]}

    problems: list[str] = []
    seen: set[str] = set()
    by_category: dict[str, int] = {}
    total = holdout = 0

    for path in sorted(glob.glob(str(REPO / "eval/golden/*.yml"))):
        questions = yaml.safe_load(io.open(path, encoding="utf-8")) or []
        for q in questions:
            total += 1
            if q.get("holdout"):
                holdout += 1
            by_category[q["category"]] = by_category.get(q["category"], 0) + 1

            if q["id"] in seen:
                problems.append(f"дубль id: {q['id']}")
            seen.add(q["id"])

            sources = q.get("expected_source_ids") or []
            for sid in sources:
                if sid not in valid:
                    problems.append(f"{q['id']}: несуществующий source_id -> {sid}")

            if q["answerable"] and not sources:
                problems.append(f"{q['id']}: answerable, но источники не указаны")
            if not q["answerable"] and sources:
                problems.append(f"{q['id']}: отказ, но источники указаны")
            if not q["answerable"] and "no_data" not in (q.get("expected_limitations") or []):
                problems.append(f"{q['id']}: отказ без ожидаемого limitation no_data")

    print(f"вопросов: {total} | holdout: {holdout} | рабочих: {total - holdout}")
    print(f"по категориям: {dict(sorted(by_category.items()))}")
    refusals = by_category.get("E", 0)
    print(f"доля отказов от всего набора: {round(100 * refusals / total)}%")

    if problems:
        print(f"\nПРОБЛЕМЫ: {len(problems)}")
        for p in problems:
            print("  ", p)
        return 1
    print("\nразметка согласована, все ссылки на источники существуют")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
