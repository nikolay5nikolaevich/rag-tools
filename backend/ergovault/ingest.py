"""Чтение корпуса: метаданные, атрибуция, чанки и воспроизводимый JSONL."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

import yaml

from ergovault.chunking import chunk_markdown, extract_reliability
from ergovault.models import (
    Chunk,
    Profile,
    Source,
    content_hash,
    estimate_tokens,
    make_chunk_id,
    make_source_id,
    quote_anchor,
)

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


TYPE_MAP = {
    "video-source": "video-source",
    "video-theses": "video-theses",
    "topic": "topic-hub",
    "articles-report": "article",
    "courses-report": "course-review",
    "company": "company",
    "event": "event",
    "final-report": "synthesis",
    "wiki-overview": "synthesis",
    "wiki-index": "synthesis",
    "sources-index": "synthesis",
    "theses-index": "synthesis",
    "companies-index": "synthesis",
    "events-index": "synthesis",
}
TYPE_ORDER = [
    "video-source",
    "video-theses",
    "topic-hub",
    "article",
    "course-review",
    "event",
    "company",
    "synthesis",
    "primary",
]


def _source_type(metadata: dict[str, Any], rel_path: str) -> str:
    raw_type = metadata.get("type")
    if raw_type is None and rel_path == "docs/wiki/Курсы по международным выставкам — анализ.md":
        return "course-review"
    if raw_type not in TYPE_MAP:
        raise ValueError(f"неизвестный type {raw_type!r}: {rel_path}")
    return TYPE_MAP[raw_type]


def _write_jsonl(path: Path, rows: list[Source] | list[Chunk]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row.model_dump(), ensure_ascii=False) + "\n")


def _report(
    sources: list[Source],
    chunks: list[Chunk],
    limit_hits: int,
    origins: dict[str, str],
    unknown: set[str],
) -> str:
    counts = Counter(source.type for source in sources)
    lengths = [chunk.char_len for chunk in chunks]

    def listing(values: list[str]) -> list[str]:
        return [f"- {value}" for value in values] or ["- нет"]

    lines = ["## Источники по SourceType"]
    lines.extend(f"- {source_type}: {counts[source_type]}" for source_type in TYPE_ORDER)
    lines.extend(
        [
            "",
            "## Чанки",
            f"- всего: {len(chunks)}",
            f"- atomic: {sum(chunk.atomic for chunk in chunks)}",
            "",
            "## Длины чанков",
            f"- min: {min(lengths)}",
            f"- медиана: {median(lengths):g}",
            f"- max: {max(lengths)}",
            f"- упёрлось в лимит 4000: {limit_hits}",
            "",
            "## Источники без speaker_org",
            *listing([source.id for source in sources if source.speaker_org is None]),
            "",
            "## Источники без url",
            *listing([source.id for source in sources if source.url is None]),
            "",
            "## Источники с reliability_note",
            *listing(
                [
                    f"{source.id} — {origins[source.id]}"
                    for source in sources
                    if source.reliability_note
                ]
            ),
            "",
            "## Непрочитанные ключи frontmatter",
            *listing(sorted(unknown)),
        ]
    )
    return "\n".join(lines) + "\n"


def ingest_corpus(corpus_root: Path, out: Path, profile: Profile = "public") -> str:
    speaker_map = load_speaker_org(corpus_root / "speaker_org.yml")
    derived_map = load_derived_map(corpus_root / "derived_map.yml")
    documents: list[tuple[str, str, dict[str, Any], str, str | None, str | None]] = []
    unknown: set[str] = set()

    # ponytail: v1 содержит только public; обработку raw/ добавлять при подключении пилота.
    for path in sorted((corpus_root / "docs").rglob("*.md"), key=lambda item: item.as_posix()):
        rel_path = path.relative_to(corpus_root).as_posix()
        raw_text = path.read_text(encoding="utf-8")
        metadata, body, unread = parse_frontmatter(raw_text)
        clean_body, reliability_note, origin = extract_reliability(body)
        documents.append((rel_path, raw_text, metadata, clean_body, reliability_note, origin))
        unknown.update(unread)

    sources: list[Source] = []
    chunks: list[Chunk] = []
    limit_hits = 0
    origins: dict[str, str] = {}
    for rel_path, raw_text, metadata, body, reliability_note, origin in documents:
        source_id = make_source_id(rel_path)
        source_type = _source_type(metadata, rel_path)
        speaker_org, stance = attribute(metadata.get("channel"), speaker_map)
        source = Source(
            id=source_id,
            path=rel_path,
            type=source_type,
            title=metadata.get("title") or Path(rel_path).stem,
            url=metadata.get("url"),
            language=metadata.get("language", "unknown"),
            published=metadata.get("published"),
            duration_minutes=metadata.get("duration_minutes"),
            speaker_org=speaker_org,
            stance=stance,
            derived=source_type == "video-theses",
            derived_of=derived_map.get(source_id),
            reliability_note=reliability_note,
            visibility="public",
            content_hash=content_hash(raw_text),
        )
        sources.append(source)
        if origin:
            origins[source_id] = origin
        document_chunks, document_limit_hits = chunk_markdown(body)
        limit_hits += document_limit_hits
        for ordinal, (heading_path, text, atomic) in enumerate(document_chunks):
            chunks.append(
                Chunk(
                    id=make_chunk_id(source_id, ordinal),
                    source_id=source_id,
                    ordinal=ordinal,
                    heading_path=heading_path,
                    text=text,
                    char_len=len(text),
                    token_estimate=estimate_tokens(text),
                    atomic=atomic,
                    quote_anchor=quote_anchor(text),
                )
            )

    by_id = {source.id: source for source in sources}
    for source in sources:
        if source.reliability_note and source.derived_of:
            original = by_id[source.derived_of]
            if original.reliability_note is None:
                original.reliability_note = source.reliability_note
                origins[original.id] = f"раздел (перенесено из {source.id})"

    out.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out / "sources.jsonl", sources)
    _write_jsonl(out / "chunks.jsonl", chunks)
    report = _report(sources, chunks, limit_hits, origins, unknown)
    with (out / "ingest_report.md").open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(report)
    return report
