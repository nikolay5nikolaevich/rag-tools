from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_speaker_org(path: Path) -> dict[str, dict[str, Any]]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_derived_map(path: Path) -> dict[str, str]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def attribute(channel: str | None, mapping: dict[str, dict[str, Any]]) -> tuple[str | None, str]:
    if channel:
        folded = channel.casefold()
        for organization, entry in mapping.items():
            if any(str(alias).casefold() in folded for alias in entry["aliases"]):
                return organization, entry["stance"]
    return None, "нейтральный"
