from __future__ import annotations

import re

from ergovault.models import MAX_CHUNK_CHARS, MIN_ATOMIC_CHARS, MIN_CHUNK_CHARS

_HEADING = re.compile(r"^(#{1,3})\s+(.+?)\s*$")
_NUMBER = re.compile(r"(?m)^\d+[.)]\s+")
_RELIABILITY_QUOTE = re.compile(r"(?m)^>\s*Примечание по достоверности:\s*(.*(?:\n>.*)*)$")


def normalize_heading(heading: str) -> str:
    return re.sub(r"^\s*\d+(?:\.\d+)*[.)]?\s+", "", heading).strip()


def extract_reliability(body: str) -> tuple[str, str | None, str | None]:
    notes: list[str] = []
    origins: list[str] = []

    def take_quote(match: re.Match[str]) -> str:
        notes.append("\n".join(line.removeprefix("> ") for line in match.group(1).splitlines()))
        origins.append("блок-цитата")
        return ""

    body = _RELIABILITY_QUOTE.sub(take_quote, body)
    lines = body.splitlines()
    kept: list[str] = []
    index = 0
    while index < len(lines):
        match = _HEADING.match(lines[index])
        if (
            match
            and len(match.group(1)) in (2, 3)
            and normalize_heading(match.group(2)).casefold() == "ограничения источника"
        ):
            level = len(match.group(1))
            index += 1
            section: list[str] = []
            while index < len(lines):
                next_heading = _HEADING.match(lines[index])
                if next_heading and len(next_heading.group(1)) <= level:
                    break
                section.append(lines[index])
                index += 1
            note = "\n".join(line.strip() for line in section if line.strip()).strip()
            if note:
                notes.append(note)
                origins.append("раздел")
            continue
        kept.append(lines[index])
        index += 1
    return "\n".join(kept), "\n".join(notes) or None, " + ".join(origins) or None


def _links(text: str) -> str:
    text = re.sub(r"!\[\[[^\]]+\]\]", "", text)
    text = re.sub(r"\[\[([^|\]]+)\|([^\]]+)\]\]", r"\2", text)
    return re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)


def _hard_split(text: str) -> list[str]:
    pieces: list[str] = []
    while len(text) > MAX_CHUNK_CHARS:
        upper = min(MAX_CHUNK_CHARS, len(text) - MIN_CHUNK_CHARS)
        cut = text.rfind(" ", 0, upper + 1)
        if cut < 1:
            cut = upper
        pieces.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        pieces.append(text)
    return pieces


def _split_long(text: str) -> list[str]:
    paragraphs = [
        paragraph.strip() for paragraph in re.split(r"\n\s*\n", text) if paragraph.strip()
    ]
    result: list[str] = []
    current = ""
    for paragraph in paragraphs:
        for piece in _hard_split(paragraph):
            candidate = f"{current}\n\n{piece}" if current else piece
            if len(candidate) <= MAX_CHUNK_CHARS:
                current = candidate
            else:
                if current:
                    result.append(current)
                current = piece
    if current:
        result.append(current)
    if len(result) > 1 and len(result[-1]) < MIN_CHUNK_CHARS:
        result[-2:] = _hard_split(f"{result[-2]}\n\n{result[-1]}")
    return result


def _section_chunks(text: str) -> list[tuple[str, bool]]:
    matches = list(_NUMBER.finditer(text))
    if len(matches) < 3:
        return [(text, False)]
    chunks: list[tuple[str, bool]] = []
    prefix = text[: matches[0].start()].strip()
    if prefix:
        chunks.append((prefix, False))
    chunks.extend(
        (
            text[match.start() : matches[index + 1].start() if index + 1 < len(matches) else None],
            True,
        )
        for index, match in enumerate(matches)
    )
    return chunks


def chunk_markdown(body: str) -> tuple[list[tuple[list[str], str, bool]], int]:
    """Режет markdown по H2/H3 и нумерованным пунктам согласно контракту §3."""
    sections: list[tuple[list[str], str]] = []
    path = ["_intro"]
    content: list[str] = []

    def flush() -> None:
        text = _links("\n".join(content)).strip()
        if text:
            sections.append((path.copy(), text))

    for line in body.splitlines():
        match = _HEADING.match(line)
        if not match:
            content.append(line)
            continue
        level = len(match.group(1))
        if level == 1:
            continue
        flush()
        content = []
        heading = normalize_heading(match.group(2))
        path = (
            [heading] if level == 2 else ([*path[:1], heading] if path != ["_intro"] else [heading])
        )
    flush()

    result: list[tuple[list[str], str, bool]] = []
    limit_hits = 0
    for heading_path, section in sections:
        for text, atomic in _section_chunks(section):
            minimum = MIN_ATOMIC_CHARS if atomic else MIN_CHUNK_CHARS
            pieces = _split_long(text.strip())
            limit_hits += len(pieces) - 1
            for piece in pieces:
                if len(piece) >= minimum:
                    result.append((heading_path, piece, atomic))
    return result, limit_hits
