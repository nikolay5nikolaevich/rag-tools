"""Подготовка базы доступна одной командой; ошибка не превращается в успех."""

import json

from typer.testing import CliRunner

from ergovault.cli import app


def test_prepare_exposes_batch_and_force_options():
    result = CliRunner().invoke(app, ["prepare", "--help"])
    assert result.exit_code == 0, result.output
    assert "batch" in result.output and "force" in result.output


def test_prepare_builds_corpus_before_embedding(tmp_path, monkeypatch):
    from ergovault import db, indexing
    from ergovault.settings import Settings

    corpus = tmp_path / "v1"
    docs = corpus / "docs"
    docs.mkdir(parents=True)
    (corpus / "speaker_org.yml").write_text("{}", encoding="utf-8")
    (corpus / "derived_map.yml").write_text("{}", encoding="utf-8")
    (docs / "sample.md").write_text(
        "---\ntype: articles-report\n---\n## Подготовка\n" + "Документ для проверки. " * 10,
        encoding="utf-8",
    )

    class LocalSettings(Settings):
        @property
        def corpus_dir(self):
            return corpus

    settings = LocalSettings()
    monkeypatch.setattr("ergovault.settings.get_settings", lambda: settings)
    stored = {}
    monkeypatch.setattr(db, "upgrade", lambda url: stored.update(migrated=True))

    def load_index(index_dir, *, settings):
        assert stored["migrated"]
        stored["chunks"] = [
            json.loads(line)
            for line in (index_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        return 1, len(stored["chunks"])

    def embed(*, batch, force, write_db, settings):
        assert stored["chunks"][0]["text"].startswith("Документ для проверки.")
        stored["embed"] = (batch, force, write_db)

    monkeypatch.setattr(db, "load_index", load_index)
    monkeypatch.setattr(indexing, "embed", embed)
    result = CliRunner().invoke(app, ["prepare", "--batch", "2", "--force"])
    assert result.exit_code == 0, result.output
    assert stored["embed"] == (2, True, True)

    def unavailable(url):
        raise RuntimeError("База недоступна")

    stored.pop("embed")
    monkeypatch.setattr(db, "upgrade", unavailable)
    result = CliRunner().invoke(app, ["prepare"])
    assert result.exit_code == 1
    assert "База недоступна" in result.output
    assert "embed" not in stored
