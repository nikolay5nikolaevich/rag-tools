// TypeScript-зеркало docs/CONTRACTS.md §1, §7, §8.
// Файл написан руками по контракту. Расхождение с контрактом — баг этого файла,
// а не устаревший контракт: править здесь, эскалировать там.

// ─── §1. Модель данных ──────────────────────────────────────────────────────

export type SourceType =
  | "video-source"
  | "video-theses"
  | "topic-hub"
  | "article"
  | "course-review"
  | "event"
  | "company"
  | "synthesis"
  | "primary";

export type Stance = "нейтральный" | "организатор-провайдер" | "подрядчик";

export type Profile = "public" | "pilot";

export type Language = "ru" | "en" | "fr" | "mixed" | "unknown";

export interface Source {
  id: string;
  path: string;
  type: SourceType;
  title: string;
  /** null, если URL первоисточника не указан. Заглушку не подставлять. */
  url: string | null;
  language: Language;
  published: string | null;
  duration_minutes: number | null;
  speaker_org: string | null;
  stance: Stance;
  derived: boolean;
  derived_of: string | null;
  /** Извлечённый из корпуса текст, не сгенерированный. */
  reliability_note: string | null;
  visibility: Profile;
  content_hash: string;
}

export interface Chunk {
  id: string;
  source_id: string;
  ordinal: number;
  heading_path: string[];
  text: string;
  char_len: number;
  token_estimate: number;
  atomic: boolean;
  quote_anchor: string;
}

// ─── §7. Контракт ответа ────────────────────────────────────────────────────

export interface Citation {
  /** Номер [1], [2] в answer_markdown. */
  n: number;
  chunk_id: string;
  source_id: string;
  source_title: string;
  source_type: SourceType;
  heading_path: string[];
  /** Дословная подстрока chunk.text, ≤300 символов. */
  quote: string;
  /** Только из source.url; модель URL не пишет. */
  url: string | null;
}

export type LimitationKind =
  | "single_org"
  | "provider_stance"
  | "reliability_note"
  | "no_data"
  | "derived_only";

export interface Limitation {
  kind: LimitationKind;
  /** Готовый текст с бэкенда. Фронт его не сочиняет и не переписывает. */
  text: string;
  source_ids: string[];
}

export interface AnswerPayload {
  answer_markdown: string;
  citations: Citation[];
  limitations: Limitation[];
  /** true — корректный отказ, а не сбой. */
  refused: boolean;
  /** corpus_version, retrieval_mode, prompt_version, model, latency_ms, cost_usd (§10). */
  meta: Record<string, unknown>;
}

// ─── §8. HTTP API ───────────────────────────────────────────────────────────

export type RetrievalMode = "fts" | "vector" | "hybrid";

export interface AskRequest {
  question: string;
  profile?: Profile;
  retrieval_mode?: RetrievalMode;
}

export interface ChunkView {
  chunk: Chunk;
  source: Source;
}

export type FeedbackVerdict = "useful" | "wrong";

export interface FeedbackRequest {
  query_id: string;
  verdict: FeedbackVerdict;
  comment?: string;
}

export interface Health {
  status: string;
  corpus_version: string;
  chunks: number;
  sources: number;
}

/** Экран «Проверка качества». Форма ответа контрактом не описана — наполняет T-31. */
export type EvalSummary = Record<string, unknown>;

export type ApiErrorCode = "bad_request" | "not_found" | "upstream" | "internal";

export interface ApiErrorBody {
  error: { code: ApiErrorCode; message: string };
}
