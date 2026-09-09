-- Схема — docs/CONTRACTS.md §5, нормативно. Не менять на месте: расхождение эскалируется
-- тех-лиду, правка контракта не делается из тикета.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE sources (
  id text PRIMARY KEY, path text NOT NULL, type text NOT NULL, title text NOT NULL,
  url text, language text NOT NULL, published text, duration_minutes int,
  speaker_org text, stance text NOT NULL, derived bool NOT NULL DEFAULT false,
  derived_of text REFERENCES sources(id), reliability_note text,
  visibility text NOT NULL DEFAULT 'public', content_hash text NOT NULL,
  corpus_version text NOT NULL
);

CREATE TABLE chunks (
  id text PRIMARY KEY, source_id text NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
  ordinal int NOT NULL, heading_path text[] NOT NULL, text text NOT NULL,
  char_len int NOT NULL, token_estimate int NOT NULL, atomic bool NOT NULL,
  quote_anchor text NOT NULL, corpus_version text NOT NULL,
  tsv_ru tsvector GENERATED ALWAYS AS (to_tsvector('russian', text)) STORED,
  tsv_simple tsvector GENERATED ALWAYS AS (to_tsvector('simple', text)) STORED,
  embedding vector(1024)
);
CREATE INDEX chunks_tsv_ru_idx     ON chunks USING gin (tsv_ru);
CREATE INDEX chunks_tsv_simple_idx ON chunks USING gin (tsv_simple);
CREATE INDEX chunks_trgm_idx       ON chunks USING gin (text gin_trgm_ops);
CREATE INDEX chunks_embedding_idx  ON chunks USING hnsw (embedding vector_cosine_ops);

CREATE TABLE query_log (
  id uuid PRIMARY KEY, created_at timestamptz NOT NULL DEFAULT now(),
  question text NOT NULL, profile text NOT NULL, retrieval_mode text NOT NULL,
  corpus_version text NOT NULL, prompt_version text NOT NULL, model text NOT NULL,
  retrieved jsonb NOT NULL,     -- [{chunk_id, score, rank}]
  answer jsonb,                 -- AnswerPayload целиком
  latency_ms int, cost_usd numeric(10,6), error text,
  feedback text                 -- null | 'useful' | 'wrong'
);
