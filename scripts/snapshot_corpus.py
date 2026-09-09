"""T-02 — снапшот корпуса ErgoVault в corpus/<version>/.

Исходный vault не изменяется никогда: скрипт только читает. Vault живой (13 источников,
цель 30-40), поэтому без заморозки цифры итогового эксперимента перестанут воспроизводиться.

Запуск:
    python scripts/snapshot_corpus.py --vault "<путь к ErgoVault>" [--version v1] [--check]

--check ничего не пишет: пересчитывает хэши и сравнивает с манифестом.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from ergovault.models import content_hash  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]

# Что берём. raw/ копируется только с --include-raw (пилотный профиль, в репозиторий не идёт).
INCLUDE_DIRS = ["wiki"]

# Что не берём никогда. build/ — 2456 файлов, включая профили Chrome и краш-дампы.
EXCLUDE_PARTS = {
    "build", ".obsidian", ".git", "system", "docs",
    "Архив — международный бизнес",
}
EXCLUDE_NAMES = {"Журнал.md"}
EXCLUDE_SUFFIXES = {".docx", ".pdf", ".png", ".jpg", ".jpeg", ".zip"}


def is_wanted(path: Path, root: Path) -> bool:
    rel = path.relative_to(root)
    if any(part in EXCLUDE_PARTS for part in rel.parts):
        return False
    if path.name in EXCLUDE_NAMES or path.name.startswith("~$"):
        return False
    if path.suffix.lower() in EXCLUDE_SUFFIXES:
        return False
    return path.suffix.lower() == ".md"


def collect(vault: Path, include_raw: bool) -> list[Path]:
    dirs = INCLUDE_DIRS + (["raw"] if include_raw else [])
    files: list[Path] = []
    for d in dirs:
        base = vault / d
        if not base.exists():
            print(f"внимание: {base} не найден, пропускаю", file=sys.stderr)
            continue
        files.extend(p for p in base.rglob("*") if p.is_file() and is_wanted(p, vault))
    return sorted(files, key=lambda p: p.relative_to(vault).as_posix())


def build_manifest(vault: Path, files: list[Path], version: str) -> dict:
    entries = []
    for p in files:
        text = p.read_text(encoding="utf-8")
        rel = p.relative_to(vault).as_posix()
        entries.append(
            {
                "path": rel,
                "sha256": content_hash(text),
                "bytes": p.stat().st_size,
                "words": len(text.split()),
                "visibility": "pilot" if rel.startswith("raw/") else "public",
            }
        )
    return {
        "corpus_version": version,
        "snapshot_date": date.today().isoformat(),
        "vault_source": str(vault),
        "file_count": len(entries),
        "total_words": sum(e["words"] for e in entries),
        "files": entries,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vault", required=True, type=Path)
    ap.add_argument("--version", default="v1")
    ap.add_argument("--include-raw", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    vault = args.vault.resolve()
    if not vault.exists():
        print(f"vault не найден: {vault}", file=sys.stderr)
        return 1

    target = REPO_ROOT / "corpus" / args.version
    manifest_path = target / "manifest.json"
    files = collect(vault, args.include_raw)
    manifest = build_manifest(vault, files, args.version)

    if args.check:
        if not manifest_path.exists():
            print("манифест отсутствует — нечего проверять", file=sys.stderr)
            return 1
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        old_map = {e["path"]: e["sha256"] for e in old["files"]}
        new_map = {e["path"]: e["sha256"] for e in manifest["files"]}
        drift = [p for p in old_map if old_map[p] != new_map.get(p)]
        added = sorted(set(new_map) - set(old_map))
        if drift or added:
            for p in drift:
                print(f"изменён: {p}")
            for p in added:
                print(f"новый в vault (не в снапшоте): {p}")
            return 2
        print(f"снапшот {args.version} совпадает с vault: {len(new_map)} файлов")
        return 0

    docs_dir = target / "docs"
    if docs_dir.exists():
        shutil.rmtree(docs_dir)
    for p in files:
        rel = p.relative_to(vault)
        dest = docs_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)

    (target / "index").mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    by_dir: dict[str, int] = {}
    for e in manifest["files"]:
        top = e["path"].split("/")[1] if e["path"].count("/") > 1 else e["path"].split("/")[0]
        by_dir[top] = by_dir.get(top, 0) + 1
    print(f"снапшот {args.version}: {manifest['file_count']} файлов, "
          f"{manifest['total_words']} слов -> {target}")
    for k in sorted(by_dir):
        print(f"  {k}: {by_dir[k]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
