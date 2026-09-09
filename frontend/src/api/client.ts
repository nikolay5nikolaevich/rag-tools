// Fetch-слой. Базовый URL — из env (VITE_API_BASE_URL). Пока API нет (T-22),
// запросы перехватывает msw; переключение на живой бэкенд — смена одной переменной.

import type {
  AnswerPayload,
  ApiErrorBody,
  ApiErrorCode,
  AskRequest,
  ChunkView,
  EvalSummary,
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
    let message = `Запрос завершился с кодом ${response.status}`;
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

/** GET /api/eval/summary */
export function getEvalSummary(signal?: AbortSignal): Promise<EvalSummary> {
  return request<EvalSummary>("/api/eval/summary", { signal });
}

/** GET /healthz */
export function getHealth(signal?: AbortSignal): Promise<Health> {
  return request<Health>("/healthz", { signal });
}

/**
 * `POST /api/feedback` требует `query_id` (§8), но контракт не говорит, откуда фронт его
 * берёт: в `AnswerPayload` отдельного поля нет, `meta` — свободный словарь (§7).
 * До ответа тех-лида читаем `meta.query_id`; если его нет — кнопки фидбека остаются
 * видимыми, но неактивными, и это видно, а не замаскировано.
 */
export function queryIdOf(answer: AnswerPayload): string | null {
  const value = answer.meta?.["query_id"];
  return typeof value === "string" && value.length > 0 ? value : null;
}
