from __future__ import annotations

import re
from typing import Any

import yaml

KEY_MAP: dict[str, str | None] = {
    "type": "type",
    "title": "title",
    "url": "url",
    "language": "language",
    "duration": "duration_minutes",
    "duration_minutes": "duration_minutes",
    "channel": "channel",
    "published": "published",
    "source": None,
    "source_video": None,
    "raw": None,
    "video_number": None,
    "status": None,
    "source_note": None,
    "playlist_url": None,
    "industry": None,
    "country": None,
    "tags": None,
}
_ABSENT = ("не указан", "не предоставл", "нет ссылки", "unknown", "неизвест")


def _present(value: Any, *, url: bool = False) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lstrip("([{ ").casefold().startswith(_ABSENT):
        return None
    if url and not text.casefold().startswith("http"):
        return None
    return text


def _duration(value: Any) -> int | None:
    text = _present(value)
    if text is None:
        return None
    match = re.search(r"\d+(?::\d+){1,2}", text)
    if match:
        parts = [int(part) for part in match.group().split(":")]
        return parts[0] * 60 + parts[1] if len(parts) == 3 else parts[0]
    match = re.search(r"\d+", text)
    return int(match.group()) if match else None


def _language(value: Any) -> str:
    text = str(value or "").strip().casefold()
    if text.startswith("en") or "англ" in text:
        return "en"
    if text.startswith("fr") or "фран" in text:
        return "fr"
    if text.startswith("ru") or "рус" in text:
        return "ru"
    if "mixed" in text or "смеш" in text:
        return "mixed"
    return "unknown"


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str, set[str]]:
    """Возвращает нормализованные метаданные, markdown-тело и неизвестные ключи."""
    normalized_text = text.replace("\r\n", "\n")
    if not normalized_text.startswith("---\n"):
        return {}, normalized_text, set()
    end = normalized_text.find("\n---\n", 4)
    if end < 0:
        return {}, normalized_text, set()

    raw = yaml.load(normalized_text[4:end], Loader=yaml.BaseLoader) or {}
    unknown = set(raw) - KEY_MAP.keys()
    metadata: dict[str, Any] = {}
    for key, target in KEY_MAP.items():
        if target is not None and key in raw:
            metadata[target] = raw[key]

    metadata["url"] = _present(metadata.get("url"), url=True)
    metadata["channel"] = _present(metadata.get("channel"))
    metadata["published"] = _present(metadata.get("published"))
    metadata["duration_minutes"] = _duration(metadata.get("duration_minutes"))
    metadata["language"] = _language(metadata.get("language"))
    return metadata, normalized_text[end + 5 :], unknown
