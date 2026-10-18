// Ссылка на сводный PDF с расшифровками видео (frontend/public/sources.pdf).
// Страница — начало раздела видео по оглавлению PDF; #page=N открывает PDF сразу на ней.
// Сопоставление по названию источника: заметки «Источники видео» и «Тезисы видео»
// называются так же, как разделы PDF. Нет совпадения — ссылки нет.

import type { Citation } from "./api/types";

export const SOURCES_PDF_URL = "/sources.pdf";

const VIDEO_PAGES: Array<[string, number]> = [
  ["trade show and conference lead follow up secrets", 4],
  ["strategies for growing international sales through trade shows", 23],
  ["zéro 1 hebdo", 50],
  ["ucloudlink", 89],
  ["global sources april 2024", 104],
  ["приглашение клиентов и продавливание заказов", 126],
  ["gitex 2023", 134],
  ["подготовка к выставке organic & natural", 141],
  ["международные выставки: как продавать по всему миру", 171],
  ["выставочный маркетинг", 195],
  ["международные выставки и как на них работать", 225],
  ["как работать на международных выставках", 240],
  ["бизнес на международной торговле", 259],
];

export function sourcesPdfHref(citation: Citation): string | null {
  if (citation.source_type !== "video-source" && citation.source_type !== "video-theses") {
    return null;
  }
  const title = citation.source_title.toLowerCase();
  const hit = VIDEO_PAGES.find(([key]) => title.includes(key));
  return hit ? `${SOURCES_PDF_URL}#page=${hit[1]}` : null;
}
