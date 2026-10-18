"""Обёртка над sentence-transformers: локальная модель эмбеддингов, без внешних API.

Нормативный источник — docs/CONTRACTS.md §5 (размерность вектора в схеме БД) и §9
(`EV_EMBED_MODEL`). Импорт этого модуля не должен грузить веса — модель поднимается
лениво, при первом вызове encode_*(), чтобы `ev --help` и юнит-тесты других пакетов
не тянули за собой ~2 ГБ весов и torch.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np


class EmbedDimensionMismatch(RuntimeError):
    """Модель вернула вектор другой размерности, чем колонка chunks.embedding (§5).

    Падать с этой ошибкой, а не молча обрезать/дополнять вектор — размерность подобрана
    под vector(1024) в схеме, расхождение почти всегда значит «модель поменяли, а
    миграцию — нет» (§5: смена модели эмбеддингов — миграция, не правка на месте).
    """

    def __init__(self, model_name: str, expected: int, actual: int) -> None:
        super().__init__(
            f"модель {model_name!r} отдаёт векторы размерности {actual}, "
            f"а колонка chunks.embedding — vector({expected}); "
            "смена модели эмбеддингов требует миграции схемы (CONTRACTS.md §5), не тихой правки"
        )
        self.model_name = model_name
        self.expected = expected
        self.actual = actual


def _uses_query_document_prefix(model_name: str) -> bool:
    """e5-семейство (intfloat/multilingual-e5-*, e5-large и т.п.) требует префиксы
    query:/passage:, иначе теряет заметную долю качества. BGE-m3 — без префиксов.
    Ветка выбирается по имени модели, а не отдельным флагом."""
    return "e5" in model_name.lower()


class EmbeddingModel:
    """Ленивая обёртка над SentenceTransformer с проверкой размерности и префиксами."""

    def __init__(self, model_name: str, expected_dim: int) -> None:
        self.model_name = model_name
        self.expected_dim = expected_dim
        self._uses_prefix = _uses_query_document_prefix(model_name)
        self._model = None

    @property
    def device(self) -> str:
        self._ensure_loaded()
        return str(self._model.device)

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                'sentence-transformers не установлен: pip install -e "backend[embed]"'
            ) from exc

        model = SentenceTransformer(self.model_name)
        # sentence-transformers >=5 переименовал метод; старое имя пока работает, но с
        # FutureWarning — используем новое там, где оно уже есть.
        get_dim = getattr(model, "get_embedding_dimension", model.get_sentence_embedding_dimension)
        actual_dim = get_dim()
        if actual_dim != self.expected_dim:
            raise EmbedDimensionMismatch(self.model_name, self.expected_dim, actual_dim)
        self._model = model

    def encode_documents(self, texts: list[str], *, batch_size: int = 32) -> np.ndarray:
        """Эмбеддинги для чанков корпуса (сторона, которую индексируем)."""
        prefix = "passage: " if self._uses_prefix else ""
        return self._encode(texts, prefix=prefix, batch_size=batch_size)

    def encode_queries(self, texts: list[str], *, batch_size: int = 32) -> np.ndarray:
        """Эмбеддинги для пользовательских вопросов."""
        prefix = "query: " if self._uses_prefix else ""
        return self._encode(texts, prefix=prefix, batch_size=batch_size)

    def _encode(self, texts: list[str], *, prefix: str, batch_size: int) -> np.ndarray:
        self._ensure_loaded()
        inputs = [prefix + t for t in texts] if prefix else list(texts)
        vectors = self._model.encode(
            inputs,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(vectors, dtype=np.float32)


@lru_cache(maxsize=1)
def get_model(model_name: str, expected_dim: int) -> EmbeddingModel:
    """Одна модель на процесс для последовательных вопросов; веса грузятся при encode."""
    return EmbeddingModel(model_name, expected_dim)
