// Экран «Библиотека»: тема, язык, дата, тип, speaker_org, stance, пометка derived,
// reliability_note. Пустые значения показываются как «не указано» — это факт корпуса
// (контракт §2), а не пробел в интерфейсе, и подменять его нечем.
//
// Оговорки длинные (несколько пунктов по сотне символов), дата бывает свободным текстом,
// поэтому материал показан строкой: слева название и оговорки на читаемую ширину,
// справа короткие метаданные. Таблица на семь колонок сжимала оговорки в узкий столбец.

import { useEffect, useMemo, useState } from "react";
import { ApiError, getSources } from "../api/client";
import type { Language, Source, SourceType } from "../api/types";
import { usePersistentState } from "../storage";

const TYPE_LABEL: Record<SourceType, string> = {
  "video-source": "видео",
  "video-theses": "тезисы видео",
  "topic-hub": "тематический раздел",
  article: "разбор статей",
  "course-review": "разбор курсов",
  event: "мероприятие",
  company: "компания",
  synthesis: "сводная заметка",
  primary: "первичный материал",
};

const LANGUAGE_LABEL: Record<Language, string> = {
  ru: "русский",
  en: "английский",
  fr: "французский",
  mixed: "смешанный",
  unknown: "не определён",
};

/** Сколько пунктов оговорок видно до раскрытия. */
const NOTES_PREVIEW = 2;

const isString = (v: unknown): v is string => typeof v === "string";
const isTypeFilter = (v: unknown): v is SourceType | "all" =>
  v === "all" || (typeof v === "string" && v in TYPE_LABEL);

function Empty() {
  return <span className="muted">не указано</span>;
}

/** Оговорка в корпусе — либо markdown-список «- …», либо сплошной текст. */
function splitNote(note: string): string[] {
  const lines = note
    .split("\n")
    .map((l) => l.trim())
    .filter((l) => l.length > 0);
  if (lines.length > 0 && lines.every((l) => l.startsWith("- "))) {
    return lines.map((l) => l.slice(2));
  }
  return [lines.join(" ")];
}

function matches(source: Source, query: string): boolean {
  if (query === "") return true;
  const haystack = [source.title, source.speaker_org, source.reliability_note, source.path]
    .filter((v): v is string => v !== null)
    .join("\n")
    .toLowerCase();
  return haystack.includes(query);
}

function Notes({ note }: { note: string | null }) {
  const [expanded, setExpanded] = useState(false);
  if (note === null) {
    return <p className="source-notes-empty">Оговорок нет.</p>;
  }
  const items = splitNote(note);
  const hidden = items.length - NOTES_PREVIEW;
  const shown = expanded || hidden <= 0 ? items : items.slice(0, NOTES_PREVIEW);

  return (
    <div className="source-notes">
      <h3>Оговорки</h3>
      <ul>
        {shown.map((item, i) => (
          <li key={i}>{item}</li>
        ))}
      </ul>
      {hidden > 0 && (
        <button
          type="button"
          className="link-button"
          aria-expanded={expanded}
          onClick={() => setExpanded((v) => !v)}
        >
          {expanded ? "Свернуть" : `Показать все (${items.length})`}
        </button>
      )}
    </div>
  );
}

function SourceRow({ s }: { s: Source }) {
  const flagged = s.stance !== "нейтральный";
  return (
    <li className="source" data-testid={`source-${s.id}`}>
      <div className="source-main">
        <h2 className="source-title">
          {s.url !== null ? (
            <a href={s.url} target="_blank" rel="noreferrer">
              {s.title}
            </a>
          ) : (
            s.title
          )}
          {s.derived && (
            <span className="badge" title={`Пересказ материала ${s.derived_of ?? ""}`}>
              пересказ
            </span>
          )}
        </h2>
        <div className="source-path">{s.path}</div>
        <Notes note={s.reliability_note} />
      </div>

      <dl className="source-meta">
        <div>
          <dt>Тип</dt>
          <dd>{TYPE_LABEL[s.type]}</dd>
        </div>
        <div>
          <dt>Язык</dt>
          <dd>{LANGUAGE_LABEL[s.language]}</dd>
        </div>
        <div>
          <dt>Дата</dt>
          <dd>{s.published ?? <Empty />}</dd>
        </div>
        <div>
          <dt>Организация спикера</dt>
          <dd>{s.speaker_org ?? <Empty />}</dd>
        </div>
        <div>
          <dt>Позиция</dt>
          <dd className={flagged ? "stance-flagged" : undefined}>{s.stance}</dd>
        </div>
      </dl>
    </li>
  );
}

export function Library() {
  const [sources, setSources] = useState<Source[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = usePersistentState("library:query:v1", "", isString);
  const [typeFilter, setTypeFilter] = usePersistentState<SourceType | "all">(
    "library:type:v1",
    "all",
    isTypeFilter,
  );

  useEffect(() => {
    const controller = new AbortController();
    getSources(controller.signal)
      .then(setSources)
      .catch((e: unknown) => {
        if (controller.signal.aborted) return;
        setError(e instanceof ApiError ? e.message : "Список материалов не загрузился.");
      });
    return () => controller.abort();
  }, []);

  // Типы в фильтре — только те, что есть в корпусе, от самого частого к редкому.
  const typeCounts = useMemo(() => {
    const counts = new Map<SourceType, number>();
    for (const s of sources ?? []) counts.set(s.type, (counts.get(s.type) ?? 0) + 1);
    return [...counts.entries()].sort((a, b) => b[1] - a[1]);
  }, [sources]);

  const normalizedQuery = query.trim().toLowerCase();
  const visible = useMemo(
    () =>
      (sources ?? []).filter(
        (s) => (typeFilter === "all" || s.type === typeFilter) && matches(s, normalizedQuery),
      ),
    [sources, typeFilter, normalizedQuery],
  );

  if (error !== null) {
    return (
      <div className="screen">
        <section className="panel panel-error">
          <h2>Библиотека не загрузилась</h2>
          <p>{error}</p>
        </section>
      </div>
    );
  }

  if (sources === null) {
    return (
      <div className="screen screen-wide">
        <p className="muted">Загружаю список материалов…</p>
      </div>
    );
  }

  const filtered = typeFilter !== "all" || normalizedQuery !== "";

  return (
    <div className="screen screen-wide">
      <header className="library-head">
        <h1>Библиотека</h1>
        <p className="lead">
          Материалы исследования: {sources.length}. Пересказы помечены — оригинал видео и его
          пересказ не считаются двумя подтверждениями.
        </p>
      </header>

      <div className="library-controls">
        <label className="visually-hidden" htmlFor="library-search">
          Поиск по библиотеке
        </label>
        <input
          id="library-search"
          type="search"
          className="library-search"
          placeholder="Найти материал"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <div className="type-filter" role="group" aria-label="Тип материала">
          <button
            type="button"
            className="chip"
            aria-pressed={typeFilter === "all"}
            onClick={() => setTypeFilter("all")}
          >
            Все <span className="chip-count">{sources.length}</span>
          </button>
          {typeCounts.map(([type, count]) => (
            <button
              key={type}
              type="button"
              className="chip"
              aria-pressed={typeFilter === type}
              onClick={() => setTypeFilter(type)}
            >
              {TYPE_LABEL[type]} <span className="chip-count">{count}</span>
            </button>
          ))}
        </div>
      </div>

      {filtered && (
        <p className="library-status" aria-live="polite">
          Показано {visible.length} из {sources.length}
        </p>
      )}

      {visible.length === 0 ? (
        <div className="library-empty">
          <p>Под эти условия ничего не подходит.</p>
          <button
            type="button"
            onClick={() => {
              setQuery("");
              setTypeFilter("all");
            }}
          >
            Сбросить фильтры
          </button>
        </div>
      ) : (
        <ul className="sources" data-testid="library-table">
          {visible.map((s) => (
            <SourceRow key={s.id} s={s} />
          ))}
        </ul>
      )}
    </div>
  );
}
