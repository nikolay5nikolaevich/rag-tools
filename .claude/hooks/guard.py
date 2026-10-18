"""PreToolUse: запрещает агенту то, что CLAUDE.md запрещает словами.

Правила в тексте агент может забыть; хук срабатывает всегда. Код возврата 2
блокирует вызов, а stderr уходит агенту объяснением.
"""

from __future__ import annotations

import json
import re
import sys

# Входные данные и ручная разметка: читать можно, менять нельзя.
READ_ONLY = ("corpus/v1/docs/", "eval/golden/")
# Не читать вовсе: holdout — для финального эксперимента, .env — секреты.
NO_READ = (
    re.compile(r"(^|/)eval/golden/holdout\.yml$"),
    re.compile(r"(^|/)\.env$"),
)
BASH_NO_READ = re.compile(r"holdout\.yml|(^|[\s/\"'])\.env(\s|$|[\"'])")


def norm(path: str) -> str:
    return path.replace("\\", "/")


def block(reason: str) -> None:
    print(f"Заблокировано хуком guard.py: {reason}", file=sys.stderr)
    sys.exit(2)


def main() -> None:
    sys.stderr.reconfigure(encoding="utf-8")
    event = json.load(sys.stdin)
    tool = event.get("tool_name", "")
    args = event.get("tool_input", {})

    if tool == "Bash":
        command = args.get("command", "")
        if BASH_NO_READ.search(command) and "--include-holdout" not in command:
            block("команда трогает holdout.yml или .env (CLAUDE.md: holdout не читать без "
                  "явного --include-holdout, секреты не выводить)")
        return

    path = norm(args.get("file_path") or args.get("notebook_path") or "")
    if not path:
        return
    if any(p.search(path) for p in NO_READ):
        block(f"{path} закрыт для агента (holdout или секреты)")
    if tool in ("Edit", "Write", "NotebookEdit") and any(d in path for d in READ_ONLY):
        block(f"{path} — входные данные или ручная разметка; их не переписывают")


if __name__ == "__main__":
    main()
