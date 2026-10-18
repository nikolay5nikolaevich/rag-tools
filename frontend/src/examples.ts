// Примеры запросов для экрана «Вопросы по исследованию» (спека раздел 3).
// Живут отдельно от моков: экран не должен импортировать mocks/, иначе моки
// уедут в боевую сборку. Моки, наоборот, отталкиваются от этого списка.

export type ExampleId =
  | "normal"
  | "single_org"
  | "provider_stance"
  | "refused"
  | "upstream_error"
  | "slow";

export interface Example {
  id: ExampleId;
  question: string;
  /** Подпись у примера: что он проверяет. Результат не обещает — его покажет ответ. */
  hint: string;
  /** Сценарий существует только в моках; живому API такой вопрос не отправляется. */
  mockOnly?: boolean;
}

const ALL_EXAMPLES: Example[] = [
  {
    id: "normal",
    question: "Как организовать работу с контактами после выставки?",
    hint: "подготовка к выставке: ответ с цитатами из нескольких источников",
  },
  {
    id: "single_org",
    question: "Какие ошибки работы на стенде повторяются в разных источниках?",
    hint: "сопоставление источников: сколько организаций на самом деле подтверждают",
  },
  {
    id: "provider_stance",
    question: "Есть ли доказательства, что участие в выставке окупается?",
    hint: "проверка доказательств: кто автор цифр и есть ли независимые измерения",
  },
  {
    id: "refused",
    question: "Сколько стоит стенд на выставке в Дубае в 2027 году?",
    hint: "вопрос вне базы: ожидается отказ, а не выдуманная цена",
  },
  {
    id: "upstream_error",
    question: "Демо: сбой генерации",
    hint: "ошибка upstream — сервис генерации недоступен",
    mockOnly: true,
  },
  {
    id: "slow",
    question: "Демо: долгий ответ",
    hint: "долгий ответ — скелетон ожидания",
    mockOnly: true,
  },
];

export const EXAMPLES: Example[] =
  import.meta.env.VITE_USE_MOCKS === "on"
    ? ALL_EXAMPLES
    : ALL_EXAMPLES.filter((e) => !e.mockOnly);

export function exampleQuestion(id: ExampleId): string {
  const found = ALL_EXAMPLES.find((e) => e.id === id);
  if (!found) throw new Error(`Неизвестный пример запроса: ${id}`);
  return found.question;
}
