// Блок «На чём основан ответ». Раскрытая карточка показывает конкретный фрагмент:
// раздел (heading_path) + дословную цитату + ссылку на оригинал — не название документа.
// url === null — источник показан без ссылки; заглушка не подставляется, источник не прячется.

import type { Citation, SourceType } from "../api/types";
import { sourcesPdfHref } from "../sourcesPdf";

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

interface Props {
  citations: Citation[];
  openedN: number | null;
  onToggle: (n: number) => void;
}

export function CitationList({ citations, openedN, onToggle }: Props) {
  if (citations.length === 0) return null;

  return (
    <section className="panel" data-testid="citations">
      <h2>На чём основан ответ</h2>
      <ul className="citations">
        {citations.map((c) => {
          const opened = openedN === c.n;
          const pdfHref = sourcesPdfHref(c);
          return (
            <li key={c.n} className="citation" data-testid={`citation-${c.n}`}>
              <button
                type="button"
                className="citation-head"
                aria-expanded={opened}
                onClick={() => onToggle(c.n)}
              >
                <span className="citation-n">[{c.n}]</span>
                <span className="citation-title">{c.source_title}</span>
                <span className="citation-type">{TYPE_LABEL[c.source_type]}</span>
                <span className="citation-chevron" aria-hidden="true">
                  {opened ? "−" : "+"}
                </span>
              </button>

              {opened && (
                <div className="citation-body" data-testid={`citation-body-${c.n}`}>
                  <div className="citation-heading-path" data-testid={`heading-path-${c.n}`}>
                    {c.heading_path.join(" → ")}
                  </div>
                  <blockquote data-testid={`quote-${c.n}`}>{c.quote}</blockquote>
                  {c.url !== null ? (
                    <a
                      className="citation-link"
                      href={c.url}
                      target="_blank"
                      rel="noreferrer"
                      data-testid={`citation-url-${c.n}`}
                    >
                      Открыть источник
                    </a>
                  ) : (
                    <span className="citation-nourl" data-testid={`citation-nourl-${c.n}`}>
                      Ссылка на первоисточник в материалах не указана
                    </span>
                  )}
                  {pdfHref !== null && (
                    <a
                      className="citation-link citation-pdf"
                      href={pdfHref}
                      target="_blank"
                      rel="noreferrer"
                      data-testid={`citation-pdf-${c.n}`}
                    >
                      Расшифровка в PDF
                    </a>
                  )}
                  <div className="citation-chunk-id">Фрагмент: {c.chunk_id}</div>
                </div>
              )}
            </li>
          );
        })}
      </ul>
    </section>
  );
}
