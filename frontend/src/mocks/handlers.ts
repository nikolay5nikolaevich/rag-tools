// msw-обработчики §8. Живут только в dev/тестах; на живой API фронт переключается
// переменной VITE_USE_MOCKS=off без правок кода экранов.

import { delay, http, HttpResponse } from "msw";
import type {
  ApiErrorBody,
  ApiErrorCode,
  AskRequest,
  FeedbackRequest,
} from "../api/types";
import { pickScenario, SCENARIOS } from "./scenarios";
import { SOURCES } from "./sources";

const BASE = import.meta.env.VITE_API_BASE_URL ?? "";

function path(p: string): string {
  return `${BASE}${p}`;
}

function apiError(code: ApiErrorCode, message: string, status: number) {
  const body: ApiErrorBody = { error: { code, message } };
  return HttpResponse.json(body, { status });
}

/** Журнал фидбека — чтобы тест мог проверить, что запрос действительно ушёл. */
export const feedbackLog: FeedbackRequest[] = [];

export const handlers = [
  http.post(path("/api/ask"), async ({ request }) => {
    const body = (await request.json()) as AskRequest;
    const question = (body?.question ?? "").trim();

    if (question.length === 0) {
      return apiError("bad_request", "Вопрос не может быть пустым.", 400);
    }

    const scenario = pickScenario(question);
    await delay(scenario.delayMs);

    if (scenario.answer === null) {
      return apiError(
        "upstream",
        "Сервис генерации ответа временно недоступен. Поиск по материалам при этом работает — повторите вопрос через минуту.",
        502,
      );
    }
    return HttpResponse.json(scenario.answer);
  }),

  http.get(path("/api/sources"), async () => {
    await delay(200);
    return HttpResponse.json(SOURCES.filter((s) => s.visibility === "public"));
  }),

  http.get(path("/api/chunks/:id"), async ({ params }) => {
    const id = String(params.id);
    const citation = SCENARIOS.flatMap((s) => s.answer?.citations ?? []).find(
      (c) => c.chunk_id === id,
    );
    const source = SOURCES.find((s) => s.id === citation?.source_id);
    if (!citation || !source) {
      return apiError("not_found", `Фрагмент ${id} не найден.`, 404);
    }
    return HttpResponse.json({
      chunk: {
        id: citation.chunk_id,
        source_id: citation.source_id,
        ordinal: Number(citation.chunk_id.split("#")[1] ?? 0),
        heading_path: citation.heading_path,
        text: citation.quote,
        char_len: citation.quote.length,
        token_estimate: Math.floor(citation.quote.length / 3),
        atomic: true,
        quote_anchor: citation.quote.slice(0, 120),
      },
      source,
    });
  }),

  http.post(path("/api/feedback"), async ({ request }) => {
    const body = (await request.json()) as FeedbackRequest;
    if (!body?.query_id || (body.verdict !== "useful" && body.verdict !== "wrong")) {
      return apiError("bad_request", "Нужны query_id и verdict.", 400);
    }
    feedbackLog.push(body);
    await delay(150);
    return new HttpResponse(null, { status: 204 });
  }),

  http.get(path("/healthz"), () =>
    HttpResponse.json({ status: "mock", corpus_version: "v1", chunks: 0, sources: SOURCES.length }),
  ),
];
