"""PostToolUse: после правки .py в backend/ сразу прогоняет ruff по файлу.

Замечания возвращаются агенту (код 2), и он чинит их в том же ходе, а не в конце.
"""

from __future__ import annotations

import json
import subprocess
import sys


def main() -> None:
    sys.stderr.reconfigure(encoding="utf-8")
    event = json.load(sys.stdin)
    path = (event.get("tool_input", {}).get("file_path") or "").replace("\\", "/")
    if not (path.endswith(".py") and "/backend/" in f"/{path}"):
        return
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", path],
        capture_output=True, text=True, encoding="utf-8", check=False,
    )
    if result.returncode != 0:
        print(f"ruff нашёл замечания в {path}:\n{result.stdout}{result.stderr}",
              file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
