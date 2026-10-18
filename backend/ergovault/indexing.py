"""Подготовка поисковой базы: корпус, загрузка в БД, кэш и векторизация."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path


def prepare(*, batch: int = 32, force: bool = False, settings=None) -> None:
    """Полный офлайн-конвейер. Ошибка шага прерывает подготовку."""
    from ergovault import db
    from ergovault.ingest import ingest_corpus
    from ergovault.settings import get_settings

    if batch < 1:
        raise ValueError("Размер batch должен быть положительным")
    settings = settings or get_settings()
    db.upgrade(settings.database_url)
    ingest_corpus(settings.corpus_dir, settings.index_dir, settings.profile)
    sources, chunks = db.load_index(settings.index_dir, settings=settings)
    print(f"sources: {sources}, chunks: {chunks}")
    embed(batch=batch, force=force, write_db=True, settings=settings)


def _safe_model_name(model_name: str) -> str:
    return model_name.replace("/", "__")


def _cache_paths(index_dir: Path, model_name: str) -> tuple[Path, Path]:
    stem = _safe_model_name(model_name)
    return (
        index_dir / f"embeddings.{stem}.npy",
        index_dir / f"embeddings.{stem}.ids.json",
    )


def _load_chunks(index_dir: Path) -> list[dict]:
    path = index_dir / "chunks.jsonl"
    if not path.exists():
        raise ValueError(f"{path} не найден — сначала `ev ingest`")
    chunks = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not chunks:
        raise ValueError(f"{path} пуст")
    return chunks


@dataclass
class _Cache:
    ids: list[str]
    vectors: object  # np.ndarray — не аннотирован явно, чтобы не тянуть numpy при --help


def _load_cache(npy_path: Path, ids_path: Path) -> _Cache | None:
    if not (npy_path.exists() and ids_path.exists()):
        return None
    import numpy as np

    ids = json.loads(ids_path.read_text(encoding="utf-8"))
    vectors = np.load(npy_path)
    if len(ids) != vectors.shape[0]:
        # Кэш повреждён/рассинхронизирован — не пытаемся угадать, что к чему относится.
        return None
    return _Cache(ids=ids, vectors=vectors)


def _save_cache(npy_path: Path, ids_path: Path, ids: list[str], vectors) -> None:
    import numpy as np

    npy_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_npy = npy_path.with_name(npy_path.name + ".tmp")
    tmp_ids = ids_path.with_name(ids_path.name + ".tmp")
    # np.save дописывает ".npy" к имени файла, если его там нет — передаём файловый
    # объект, а не путь, иначе tmp-файл окажется не там, куда его потом переименовываем.
    with tmp_npy.open("wb") as stream:
        np.save(stream, vectors.astype(np.float32))
    tmp_ids.write_text(json.dumps(ids, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_npy.replace(npy_path)
    tmp_ids.replace(ids_path)


def embed(*, batch: int = 32, force: bool = False, write_db: bool = True, settings=None) -> None:
    """Посчитать недостающие векторы, сохранить кэш и при необходимости записать БД."""
    if batch < 1:
        raise ValueError("Размер batch должен быть положительным")
    import numpy as np

    from ergovault.embeddings import EmbeddingModel
    from ergovault.settings import get_settings

    settings = settings or get_settings()
    index_dir = settings.index_dir
    npy_path, ids_path = _cache_paths(index_dir, settings.embed_model)

    chunk_rows = _load_chunks(index_dir)
    corpus_ids = [row["id"] for row in chunk_rows]
    text_by_id = {row["id"]: row["text"] for row in chunk_rows}

    cache = None if force else _load_cache(npy_path, ids_path)

    if cache is not None:
        cached_set = set(cache.ids)
        corpus_set = set(corpus_ids)
        missing_ids = [cid for cid in corpus_ids if cid not in cached_set]
        stale_ids = cached_set - corpus_set
        if not missing_ids and not stale_ids:
            print(
                f"кэш актуален: {len(corpus_ids)} векторов, пересчёт не требуется ({npy_path.name})"
            )
            by_cached_id = dict(zip(cache.ids, cache.vectors))
            db_status = _write_to_db(
                settings,
                corpus_ids,
                [by_cached_id[cid] for cid in corpus_ids],
                enabled=write_db,
            )
            _write_report(
                settings,
                index_dir,
                npy_path,
                vector_count=len(corpus_ids),
                elapsed_seconds=0.0,
                device="—",
                recomputed=0,
                db_status=db_status,
            )
            return
    else:
        missing_ids = list(corpus_ids)
        stale_ids: set[str] = set()

    model = EmbeddingModel(settings.embed_model, settings.embed_dim)

    print(f"модель: {settings.embed_model} (размерность {settings.embed_dim})")
    if cache is not None:
        print(
            f"новых чанков: {len(missing_ids)}, устаревших в кэше: {len(stale_ids)}, "
            f"уже посчитано: {len(cache.ids) - len(stale_ids)}"
        )
    else:
        print(f"чанков к векторизации: {len(missing_ids)} (force={force})")

    start = time.monotonic()
    new_vectors = np.empty((0, settings.embed_dim), dtype=np.float32)
    if missing_ids:
        texts = [text_by_id[cid] for cid in missing_ids]
        computed_chunks: list[np.ndarray] = []
        for i in range(0, len(texts), batch):
            batch_texts = texts[i : i + batch]
            computed_chunks.append(model.encode_documents(batch_texts, batch_size=batch))
            print(f"  {min(i + batch, len(texts))}/{len(texts)}")
        new_vectors = np.concatenate(computed_chunks, axis=0)
    elapsed = time.monotonic() - start

    if cache is not None:
        kept_mask = [cid not in stale_ids for cid in cache.ids]
        kept_ids = [cid for cid, keep in zip(cache.ids, kept_mask) if keep]
        kept_vectors = (
            cache.vectors[kept_mask]
            if kept_ids
            else np.empty((0, settings.embed_dim), dtype=np.float32)
        )
        by_id = dict(zip(kept_ids, kept_vectors))
        by_id.update(dict(zip(missing_ids, new_vectors)))
    else:
        by_id = dict(zip(missing_ids, new_vectors))

    final_ids = corpus_ids
    final_vectors = np.stack([by_id[cid] for cid in final_ids]).astype(np.float32)

    _save_cache(npy_path, ids_path, final_ids, final_vectors)

    device = model.device if missing_ids else "—"
    print(
        f"готово: {len(final_ids)} векторов ({len(missing_ids)} посчитано заново), "
        f"{elapsed:.1f}с, устройство={device}, кэш={npy_path.name}"
    )
    db_status = _write_to_db(settings, final_ids, list(final_vectors), enabled=write_db)

    _write_report(
        settings,
        index_dir,
        npy_path,
        vector_count=len(final_ids),
        elapsed_seconds=elapsed,
        device=device,
        recomputed=len(missing_ids),
        db_status=db_status,
    )


def _write_to_db(settings, ids: list[str], vectors, *, enabled: bool) -> str:
    """Заливает векторы в `chunks.embedding` через db.upsert_embeddings и возвращает
    строку статуса для отчёта.

    Недоступная база при `--write-db` — это ошибка, а не предупреждение: пользователь
    просил запись, записи не произошло. Кэш .npy к этому моменту уже сохранён, повторный
    прогон ничего не пересчитывает.
    """
    if not enabled:
        message = "Postgres не обновлён (`--no-write-db`): векторы только в .npy/.ids.json"
        print(message)
        return message

    import psycopg

    from ergovault.db import embedding_coverage, upsert_embeddings

    payload = {cid: list(map(float, vec)) for cid, vec in zip(ids, vectors)}
    try:
        with psycopg.connect(settings.database_url) as conn:
            updated = upsert_embeddings(conn, payload, dim=settings.embed_dim)
            total, covered = embedding_coverage(conn)
    except psycopg.OperationalError as exc:
        print(
            f"база недоступна ({exc.__class__.__name__}): {exc}\n"
            "Кэш .npy сохранён. Поднимите базу (`docker compose up -d db`, `ev db upgrade`, "
            "`ev db load`) и повторите `ev embed`, либо запустите с --no-write-db."
        )
        raise RuntimeError("Не удалось записать векторы в Postgres; кэш сохранён") from exc

    message = f"Postgres: обновлено {updated} строк, покрытие chunks.embedding {covered}/{total}"
    print(message)
    if updated < len(payload):
        # Не создаём недостающие чанки: расхождение кэша с базой разбирает человек.
        missing = len(payload) - updated
        warning = (
            f"внимание: {missing} векторов не легли ни в одну строку chunks — "
            "кэш эмбеддингов и содержимое базы разошлись, нужен `ev db load`"
        )
        print(warning)
        message = f"{message}; {warning}"
    return message


def _write_report(
    settings,
    index_dir: Path,
    npy_path: Path,
    *,
    vector_count: int,
    elapsed_seconds: float,
    device: str,
    recomputed: int,
    db_status: str,
) -> None:
    import datetime

    from ergovault.settings import REPO_ROOT

    try:
        displayed_path = npy_path.relative_to(REPO_ROOT)
    except ValueError:
        displayed_path = npy_path

    report_path = index_dir / "embed_report.md"
    report_path.write_text(
        "# Отчёт индексации эмбеддингов\n\n"
        f"- модель: `{settings.embed_model}`\n"
        f"- размерность: {settings.embed_dim}\n"
        f"- векторов в кэше: {vector_count}\n"
        f"- пересчитано за этот прогон: {recomputed}\n"
        f"- время пересчёта: {elapsed_seconds:.1f} с\n"
        f"- устройство: {device}\n"
        f"- кэш: `{displayed_path}`\n"
        f"- HF_HOME: веса модели кэшируются HuggingFace Hub по переменной `HF_HOME` "
        "(по умолчанию `~/.cache/huggingface`); первая загрузка BGE-m3 — ~2 ГБ.\n"
        f"- {db_status}\n"
        f"- сформировано: {datetime.datetime.now(datetime.UTC).isoformat(timespec='seconds')}\n",
        encoding="utf-8",
    )
