"""Stop: если в рабочем дереве менялся backend/, агент не закончит ход с красными тестами.

Запускается ruff по backend и pytest без live-тестов (реальные вызовы Claude не идут).
Повторный Stop после замечаний пропускается (`stop_hook_active`), чтобы не зациклиться:
второй раз агент докладывает человеку о падении, а не чинит бесконечно.
"""

from __future__ import annotations

import json
import subprocess
import sys


def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(args), capture_output=True, text=True, encoding="utf-8", check=False)


def main() -> None:
    sys.stderr.reconfigure(encoding="utf-8")
    event = json.load(sys.stdin)
    if event.get("stop_hook_active"):
        return
    changed = run("git", "status", "--porcelain", "--", "backend").stdout.strip()
    if not changed:
        return

    ruff = run(sys.executable, "-m", "ruff", "check", "backend")
    if ruff.returncode != 0:
        print(f"ruff check backend:\n{ruff.stdout[-3000:]}", file=sys.stderr)
        sys.exit(2)

    tests = run(sys.executable, "-m", "pytest", "backend/tests", "-q", "-x", "--no-header")
    if tests.returncode != 0:
        print("pytest backend/tests упал. Если причина — недоступная база, "
              "скажи об этом человеку, а не правь тесты.\n" + tests.stdout[-3000:],
              file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
