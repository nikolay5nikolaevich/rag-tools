// Fetch-слой. Базовый URL — из env (VITE_API_BASE_URL); по умолчанию запросы идут
// на текущий origin. Моки msw включаются отдельно (VITE_USE_MOCKS=on) и в обычном
// запуске выключены — фронт работает с живым API.

import type {
  AnswerPayload,
  ApiErrorBody,
  ApiErrorCode,
  AskRequest,
  ChunkView,
  FeedbackRequest,
  Health,
  Source,
} from "./types";

export const API_BASE_URL: string = import.meta.env.VITE_API_BASE_URL ?? "";

export class ApiError extends Error {
  readonly code: ApiErrorCode | "network";
  readonly status: number;

  constructor(code: ApiErrorCode | "network", message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.code = code;
    this.status = status;
  }
}

function url(path: string): string {
  return `${API_BASE_URL}${path}`;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(url(path), {
      ...init,
      headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    });
  } catch (cause) {
    throw new ApiError("network", `Сервис недоступен: ${String(cause)}`, 0);
  }

  if (!response.ok) {
    let code: ApiErrorCode = "internal";
    // Тела §8 может не быть вовсе: упавший или ещё не поднятый бэкенд отвечает через
    // прокси голым 5xx. Тогда пользователю нужен не код ответа, а что именно случилось.
    let message =
      response.status >= 500
        ? "Сервис недоступен. Проверьте, что бэкенд запущен, и повторите запрос."
        : `Запрос завершился с кодом ${response.status}`;
    try {
      const body = (await response.json()) as ApiErrorBody;
      if (body?.error) {
        code = body.error.code;
        message = body.error.message;
      }
    } catch {
      // тело не разобралось — оставляем сообщение по коду ответа
    }
    throw new ApiError(code, message, response.status);
  }

  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

/** POST /api/ask */
export function ask(payload: AskRequest, signal?: AbortSignal): Promise<AnswerPayload> {
  return request<AnswerPayload>("/api/ask", {
    method: "POST",
    body: JSON.stringify(payload),
    signal,
  });
}

/** GET /api/sources */
export function getSources(signal?: AbortSignal): Promise<Source[]> {
  return request<Source[]>("/api/sources", { signal });
}

/** GET /api/chunks/{id} */
export function getChunk(id: string, signal?: AbortSignal): Promise<ChunkView> {
  return request<ChunkView>(`/api/chunks/${encodeURIComponent(id)}`, { signal });
}

/** POST /api/feedback -> 204 */
export function sendFeedback(payload: FeedbackRequest, signal?: AbortSignal): Promise<void> {
  return request<void>("/api/feedback", {
    method: "POST",
    body: JSON.stringify(payload),
    signal,
  });
}

/** GET /healthz */
export function getHealth(signal?: AbortSignal): Promise<Health> {
  return request<Health>("/healthz", { signal });
}

/**
 * `query_id` для `POST /api/feedback` API возвращает в `meta.query_id` (§8). Если поля
 * нет (старый ответ, другой бэкенд), кнопки отзыва остаются видимыми, но неактивными:
 * отсутствие идентификатора видно, а не замаскировано.
 */
export function queryIdOf(answer: AnswerPayload): string | null {
  const value = answer.meta?.["query_id"];
  return typeof value === "string" && value.length > 0 ? value : null;
}
