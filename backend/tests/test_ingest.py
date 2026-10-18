from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from typer.testing import CliRunner

from ergovault.chunking import chunk_markdown
from ergovault.cli import app
from ergovault.models import Chunk, Source

ROOT = Path(__file__).parents[2]
CORPUS = ROOT / "corpus" / "v1"
EXPECTED_TYPES = {
    "video-source": 13,
    "video-theses": 13,
    "topic-hub": 5,
    "article": 1,
    "course-review": 2,
    "event": 1,
    "company": 1,
    "synthesis": 7,
}


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _run_ingest(out: Path, *, report: bool = False) -> str:
    args = ["ingest", "--profile", "public", "--out", str(out)]
    if report:
        args.append("--report")
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    return result.output


def test_real_corpus_sources_and_attribution(tmp_path: Path) -> None:
    output = _run_ingest(tmp_path, report=True)
    sources = [Source.model_validate(item) for item in _read_jsonl(tmp_path / "sources.jsonl")]

    assert len(sources) == 43
    assert Counter(source.type for source in sources) == EXPECTED_TYPES

    by_id = {source.id: source for source in sources}
    theses = [source for source in sources if source.type == "video-theses"]
    assert len(theses) == 13
    assert all(source.derived and source.derived_of in by_id for source in theses)
    assert all(by_id[source.derived_of].type == "video-source" for source in theses)
    assert sum(source.reliability_note is not None for source in sources) >= 13
    assert all(
        not source.url
        or ("не указан" not in source.url.lower() and "не предоставл" not in source.url.lower())
        for source in sources
    )
    assert all(
        not source.published
        or not source.published.lstrip("([{ ")
        .casefold()
        .startswith(("не указан", "не предоставл", "нет ссылки", "unknown", "неизвест"))
        for source in sources
    )
    assert "- упёрлось в лимит 4000: 0" not in output
    assert output == (tmp_path / "ingest_report.md").read_text(encoding="utf-8")


def test_real_corpus_chunks_obey_contract(tmp_path: Path) -> None:
    _run_ingest(tmp_path)
    sources = [Source.model_validate(item) for item in _read_jsonl(tmp_path / "sources.jsonl")]
    chunks = [Chunk.model_validate(item) for item in _read_jsonl(tmp_path / "chunks.jsonl")]

    source_ids = {source.id for source in sources}
    chunk_ids = [chunk.id for chunk in chunks]
    assert 500 <= len(chunks) <= 1200
    assert len(set(chunk_ids)) == len(chunk_ids)
    assert all(chunk.source_id in source_ids for chunk in chunks)
    assert all(chunk.heading_path for chunk in chunks)
    assert all((40 if chunk.atomic else 80) <= chunk.char_len <= 4000 for chunk in chunks)
    assert sum(chunk.atomic for chunk in chunks) >= 100
    assert all(not source.path.startswith("raw/") for source in sources)
    assert all("![[" not in chunk.text for chunk in chunks)
    assert all(source.visibility == "public" for source in sources)


def test_real_corpus_ingest_is_byte_for_byte_idempotent(tmp_path: Path) -> None:
    _run_ingest(tmp_path)
    first = {name: (tmp_path / name).read_bytes() for name in ("sources.jsonl", "chunks.jsonl")}

    _run_ingest(tmp_path)

    assert all((tmp_path / name).read_bytes() == content for name, content in first.items())
    for name, model in (("sources.jsonl", Source), ("chunks.jsonl", Chunk)):
        lines = (tmp_path / name).read_text(encoding="utf-8").splitlines(keepends=True)
        assert all(line.endswith("\n") for line in lines)
        assert list(json.loads(lines[0])) == list(model.model_fields)


def test_long_section_keeps_short_tail() -> None:
    chunks, limit_hits = chunk_markdown("## Раздел\n\n" + "x" * 4050)

    assert [len(text) for _, text, _ in chunks] == [3970, 80]
    assert "".join(text for _, text, _ in chunks) == "x" * 4050
    assert limit_hits == 1
