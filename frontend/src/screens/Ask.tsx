// Экран «Вопросы по исследованию» (спека раздел 3).
// Процента уверенности здесь нет и быть не может — требование спеки.

import { useRef, useState } from "react";
import { AnswerBody } from "../components/AnswerBody";
import { CitationList } from "../components/CitationList";
import { Limitations } from "../components/Limitations";
import { ApiError, ask, queryIdOf, sendFeedback } from "../api/client";
import type { AnswerPayload, FeedbackVerdict } from "../api/types";
import { EXAMPLES } from "../examples";
import { usePersistentState } from "../storage";

type FeedbackState =
  | { kind: "idle" }
  | { kind: "sending"; verdict: FeedbackVerdict }
  | { kind: "sent"; verdict: FeedbackVerdict }
  | { kind: "failed"; message: string };

// Проверки формы для записей из localStorage: запись могла остаться от старой версии
// или быть испорчена руками. Неподходящая запись заменяется значением по умолчанию.
const isString = (v: unknown): v is string => typeof v === "string";

const isOpenedN = (v: unknown): v is number | null => v === null || typeof v === "number";

const isAnswer = (v: unknown): v is AnswerPayload | null => {
  if (v === null) return true;
  if (typeof v !== "object") return false;
  const a = v as Partial<AnswerPayload>;
  return (
    typeof a.answer_markdown === "string" &&
    Array.isArray(a.citations) &&
    Array.isArray(a.limitations) &&
    typeof a.refused === "boolean" &&
    typeof a.meta === "object" &&
    a.meta !== null
  );
};

// После перезагрузки имеет смысл только «отзыв отправлен». Незавершённая отправка
// или сбой не восстанавливаются, иначе кнопки остались бы заблокированными.
const isStoredFeedback = (v: unknown): v is FeedbackState => {
  if (typeof v !== "object" || v === null) return false;
  const f = v as { kind?: unknown; verdict?: unknown };
  return f.kind === "idle" || (f.kind === "sent" && (f.verdict === "useful" || f.verdict === "wrong"));
};

function Skeleton() {
  return (
    <section className="panel" data-testid="skeleton" aria-busy="true">
      <h2>Ответ</h2>
      <p className="muted">Ищу в материалах исследования…</p>
      <div className="skeleton-line" />
      <div className="skeleton-line" />
      <div className="skeleton-line short" />
    </section>
  );
}

export function Ask() {
  const [question, setQuestion] = usePersistentState("ask:question:v1", "", isString);
  const [answer, setAnswer] = usePersistentState<AnswerPayload | null>("ask:answer:v1", null, isAnswer);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [openedN, setOpenedN] = usePersistentState<number | null>("ask:opened:v1", null, isOpenedN);
  const [feedback, setFeedback] = usePersistentState<FeedbackState>(
    "ask:feedback:v1",
    { kind: "idle" },
    isStoredFeedback,
  );
  const [comment, setComment] = usePersistentState("ask:comment:v1", "", isString);
  const inFlight = useRef<AbortController | null>(null);
  const input = useRef<HTMLInputElement>(null);

  function reset() {
    inFlight.current?.abort();
    setLoading(false);
    setQuestion("");
    setAnswer(null);
    setError(null);
    setOpenedN(null);
    setFeedback({ kind: "idle" });
    setComment("");
    input.current?.focus();
  }

  async function run(text: string) {
    const q = text.trim();
    if (q.length === 0 || loading) return;

    inFlight.current?.abort();
    const controller = new AbortController();
    inFlight.current = controller;

    setLoading(true);
    setError(null);
    setAnswer(null);
    setOpenedN(null);
    setFeedback({ kind: "idle" });
    setComment("");

    try {
      const payload = await ask({ question: q }, controller.signal);
      setAnswer(payload);
    } catch (e) {
      if (controller.signal.aborted) return;
      setError(
        e instanceof ApiError
          ? e.message
          : "Не удалось получить ответ. Попробуйте повторить вопрос.",
      );
    } finally {
      if (!controller.signal.aborted) setLoading(false);
    }
  }

  async function vote(verdict: FeedbackVerdict) {
    if (!answer) return;
    const queryId = queryIdOf(answer);
    if (queryId === null) return;

    setFeedback({ kind: "sending", verdict });
    try {
      // Комментарий необязателен: пустое поле уходит как отсутствие текста, а не пустая строка.
      const text = comment.trim();
      await sendFeedback({ query_id: queryId, verdict, ...(text === "" ? {} : { comment: text }) });
      setFeedback({ kind: "sent", verdict });
    } catch (e) {
      setFeedback({
        kind: "failed",
        message: e instanceof ApiError ? e.message : "Отзыв не отправился.",
      });
    }
  }

  const queryId = answer ? queryIdOf(answer) : null;

  const compact = answer !== null || loading;

  return (
    <>
      <section className={compact ? "hero compact" : "hero"}>
        <div className="hero-inner">
          <form
            className="ask-form"
            onSubmit={(e) => {
              e.preventDefault();
              void run(question);
            }}
          >
            <label htmlFor="question" className="ask-title">
              Что хотите выяснить?
            </label>
            <p className="ask-lead">
              Ответ собирается только из материалов исследования о международных выставках — с
              цитатами и оговорками об источниках.
            </p>
            <div className="ask-row">
              <input
                ref={input}
                id="question"
                name="question"
                type="text"
                autoComplete="off"
                placeholder="Спросите о выставках"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
              />
              <button type="submit" className="primary" disabled={loading || question.trim() === ""}>
                {loading ? "Ищу…" : "Спросить"}
              </button>
            </div>
          </form>

          {answer === null && !loading && (
            <section className="examples">
              <h2>Можно начать с примера</h2>
              <ul>
                {EXAMPLES.map((s) => (
                  <li key={s.id}>
                    <button
                      type="button"
                      className="example"
                      data-testid={`example-${s.id}`}
                      onClick={() => {
                        setQuestion(s.question);
                        void run(s.question);
                      }}
                    >
                      {s.question}
                    </button>
                    <span className="example-hint">{s.hint}</span>
                  </li>
                ))}
              </ul>
            </section>
          )}
        </div>
      </section>

      <div className="screen">
        {loading && <Skeleton />}

        {error !== null && !loading && (
          <section className="panel panel-error" data-testid="error">
            <h2>Ответ не получен</h2>
            <p>{error}</p>
            <button type="button" onClick={() => void run(question)}>
              Повторить
            </button>
          </section>
        )}

        {answer !== null && !loading && (
          <>
            <section
              className={answer.refused ? "panel panel-refused" : "panel"}
              data-testid={answer.refused ? "answer-refused" : "answer"}
            >
              <div className="answer-head">
                <h2>{answer.refused ? "Данных для ответа не нашлось" : "Ответ"}</h2>
                <button type="button" className="link-button" onClick={reset}>
                  Новый вопрос
                </button>
              </div>
              {answer.refused && (
                <p className="refused-lead">
                  Это полноценный ответ: система не придумывает недостающие сведения, а называет,
                  чего именно нет в материалах.
                </p>
              )}
              <AnswerBody
                markdown={answer.answer_markdown}
                citations={answer.citations}
                onCitationClick={(n) => setOpenedN(n)}
              />
            </section>

            <CitationList
              citations={answer.citations}
              openedN={openedN}
              onToggle={(n) => setOpenedN((current) => (current === n ? null : n))}
            />

            <Limitations limitations={answer.limitations} />

            <section className="panel feedback" data-testid="feedback">
              <h2>Оценка ответа</h2>
              {queryId === null ? (
                <p className="muted" data-testid="feedback-unavailable">
                  Отзыв недоступен: ответ пришёл без идентификатора запроса.
                </p>
              ) : feedback.kind === "sent" ? (
                <p data-testid="feedback-sent">
                  Спасибо, отзыв записан: «{feedback.verdict === "useful" ? "Полезно" : "Есть ошибка"}».
                </p>
              ) : (
                <>
                  <label className="feedback-comment" htmlFor="feedback-comment">
                    Что именно не так или чем ответ помог (необязательно)
                  </label>
                  <textarea
                    id="feedback-comment"
                    data-testid="feedback-comment"
                    rows={2}
                    value={comment}
                    onChange={(e) => setComment(e.target.value)}
                    placeholder="Например: цитата [2] не подтверждает вывод"
                  />
                  <div className="feedback-buttons">
                  <button
                    type="button"
                    data-testid="feedback-useful"
                    disabled={feedback.kind === "sending"}
                    onClick={() => void vote("useful")}
                  >
                    Полезно
                  </button>
                  <button
                    type="button"
                    data-testid="feedback-wrong"
                    disabled={feedback.kind === "sending"}
                    onClick={() => void vote("wrong")}
                  >
                    Есть ошибка
                  </button>
                  {feedback.kind === "failed" && <span className="error-text">{feedback.message}</span>}
                  </div>
                </>
              )}
            </section>
          </>
        )}
      </div>
    </>
  );
}
