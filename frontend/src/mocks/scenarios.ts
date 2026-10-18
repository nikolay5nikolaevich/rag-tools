// Шесть сценариев моков. Выбор сценария — по ключевым словам вопроса,
// чтобы каждый сценарий открывался кликом по примеру запроса и воспроизводился в тесте.

import type { AnswerPayload, Citation } from "../api/types";
import { type ExampleId, exampleQuestion } from "../examples";

export type ScenarioId = ExampleId;

export interface Scenario {
  id: ScenarioId;
  /** Текст примера запроса — берётся из общего списка src/examples.ts. */
  question: string;
  /** Ключевые слова в вопросе, по которым мок узнаёт сценарий. */
  keywords: string[];
  /** Задержка ответа, мс. */
  delayMs: number;
  /** null для сценария с ошибкой upstream. */
  answer: AnswerPayload | null;
}

const META = {
  query_id: "00000000-0000-4000-8000-0000000000",
  corpus_version: "v1",
  retrieval_mode: "hybrid",
  prompt_version: "p1",
  model: "claude-opus-5",
  latency_ms: 2840,
  cost_usd: 0.0143,
};

function meta(suffix: string, patch: Record<string, unknown> = {}) {
  return { ...META, query_id: META.query_id + suffix, ...patch };
}

const CIT_FOLLOWUP: Citation = {
  n: 1,
  chunk_id: "istochniki-video--trade-show-and-conference-lead-follow-up-secrets-featuring-h#004",
  source_id: "istochniki-video--trade-show-and-conference-lead-follow-up-secrets-featuring-h",
  source_title: "Trade Show and Conference Lead Follow Up Secrets featuring Harriet Mellor",
  source_type: "video-source",
  heading_path: ["Основные тезисы", "Работа с контактами после выставки"],
  quote:
    "The first follow-up should go out within 48 hours, while the conversation is still fresh — after a week the lead is effectively cold.",
  url: "https://www.youtube.com/watch?v=mock-salesgravy",
};

const CIT_CRM: Citation = {
  n: 2,
  chunk_id: "posle-meropriyatiya--posle-meropriyatiya#002",
  source_id: "posle-meropriyatiya--posle-meropriyatiya",
  source_title: "После мероприятия",
  source_type: "topic-hub",
  heading_path: ["Обработка контактов", "Порядок разбора визиток"],
  quote:
    "Контакты разбираются в тот же вечер: карточка, пометка о договорённости и срок следующего касания. Без этого через неделю половина визиток не опознаётся.",
  url: null,
};

const CIT_PREP: Citation = {
  n: 3,
  chunk_id: "istochniki-video--vystavochnyy-marketing-kak-pravilno-uchastvovat-v-vystavkah#007",
  source_id: "istochniki-video--vystavochnyy-marketing-kak-pravilno-uchastvovat-v-vystavkah",
  source_title: "Выставочный маркетинг — как правильно участвовать в выставках",
  source_type: "video-source",
  heading_path: ["Подготовка", "Назначение встреч до выставки"],
  quote:
    "Встречи назначаются за три-четыре недели до выставки: на площадке договориться уже почти не с кем, все расписаны.",
  url: "https://www.youtube.com/watch?v=mock-eventrocks",
};

const CIT_STAND_1: Citation = {
  n: 1,
  chunk_id: "istochniki-video--kak-rabotat-na-mezhdunarodnyh-vystavkah-rusglobaleksport#009",
  source_id: "istochniki-video--kak-rabotat-na-mezhdunarodnyh-vystavkah-rusglobaleksport",
  source_title: "Как работать на международных выставках — РусГлобалЭкспорт",
  source_type: "video-source",
  heading_path: ["Ошибки на стенде", "Персонал"],
  quote:
    "Самая частая ошибка — стенд, на котором сидят и смотрят в телефон. Посетитель проходит мимо: к нему никто не вышел.",
  url: "https://www.youtube.com/watch?v=mock-rge-1",
};

const CIT_STAND_2: Citation = {
  n: 2,
  chunk_id: "istochniki-video--mezhdunarodnye-vystavki-i-kak-na-nih-rabotat#005",
  source_id: "istochniki-video--mezhdunarodnye-vystavki-i-kak-na-nih-rabotat",
  source_title: "Международные выставки и как на них работать",
  source_type: "video-source",
  heading_path: ["Работа на стенде", "Типовые провалы"],
  quote:
    "Второй провал — материалы только на русском. На международной площадке это отсекает большую часть посетителей.",
  url: "https://www.youtube.com/watch?v=mock-rge-2",
};

const CIT_ROI_1: Citation = {
  n: 1,
  chunk_id: "istochniki-video--podgotovka-k-organic-natural-v-dubae-vebinar-rusglobalekspor#011",
  source_id: "istochniki-video--podgotovka-k-organic-natural-v-dubae-vebinar-rusglobalekspor",
  source_title: "Подготовка к Organic & Natural в Дубае — вебинар Русглобалэкспорт",
  source_type: "video-source",
  heading_path: ["Бюджет участия", "Окупаемость"],
  quote:
    "По нашему опыту стенд окупается с двух-трёх контрактов, но это оценка, а не посчитанная статистика.",
  url: null,
};

const CIT_ROI_2: Citation = {
  n: 2,
  chunk_id: "wiki--analitika-statey#003",
  source_id: "wiki--analitika-statey",
  source_title: "Аналитика статей",
  source_type: "article",
  heading_path: ["Что говорят публикации", "Цифры отдачи"],
  quote:
    "Публикации приводят показатели отдачи со ссылкой на самих организаторов; независимых измерений в разобранных материалах нет.",
  url: null,
};

export const SCENARIOS: Scenario[] = [
  {
    id: "normal",
    question: exampleQuestion("normal"),
    keywords: ["контакт", "после выставки", "follow"],
    delayMs: 350,
    answer: {
      answer_markdown: `Работа с контактами разбивается на три шага.

1. **Разобрать контакты в тот же день.** Карточка, договорённость, срок следующего касания [2].
2. **Первое касание — в течение 48 часов**, пока разговор ещё помнят обе стороны [1].
3. **Назначать встречи заранее**: часть контактов появляется до выставки, а не на ней [3].

Порядок шагов — обобщение по материалам; отдельного источника, который описывал бы всю
последовательность целиком, в корпусе нет.`,
      citations: [CIT_FOLLOWUP, CIT_CRM, CIT_PREP],
      limitations: [
        {
          kind: "reliability_note",
          text: "Раздел «После мероприятия» — сводная заметка исследования; формулировки обобщены составителем, а не взяты дословно из первоисточника.",
          source_ids: ["posle-meropriyatiya--posle-meropriyatiya"],
        },
      ],
      refused: false,
      meta: meta("01"),
    },
  },
  {
    id: "single_org",
    question: exampleQuestion("single_org"),
    keywords: ["ошибк"],
    delayMs: 350,
    answer: {
      answer_markdown: `В материалах повторяются две ошибки.

1. **Персонал не работает с проходящим потоком** — сидят на стенде и ждут [1].
2. **Материалы только на русском языке** на международной площадке [2].

Оба фрагмента происходят из материалов одной организации, то есть это одно
подтверждение, а не два.`,
      citations: [CIT_STAND_1, CIT_STAND_2],
      limitations: [
        {
          kind: "single_org",
          text: "Все процитированные материалы принадлежат одной организации — RusGlobalExport. Независимого подтверждения в корпусе нет.",
          source_ids: [
            "istochniki-video--kak-rabotat-na-mezhdunarodnyh-vystavkah-rusglobaleksport",
            "istochniki-video--mezhdunarodnye-vystavki-i-kak-na-nih-rabotat",
          ],
        },
        {
          kind: "provider_stance",
          text: "Источник продаёт участие в выставках (организатор-провайдер), его рекомендации не нейтральны.",
          source_ids: ["istochniki-video--kak-rabotat-na-mezhdunarodnyh-vystavkah-rusglobaleksport"],
        },
      ],
      refused: false,
      meta: meta("02"),
    },
  },
  {
    id: "provider_stance",
    question: exampleQuestion("provider_stance"),
    keywords: ["окуп", "доказательств", "roi"],
    delayMs: 350,
    answer: {
      answer_markdown: `Измеренных независимых данных об окупаемости в материалах нет.

Названные цифры — оценки участников рынка: «окупается с двух-трёх контрактов» звучит как
опыт, а не как расчёт [1]. Разобранные публикации приводят показатели отдачи со ссылкой
на самих организаторов выставок [2].

То есть на этом корпусе корректный ответ: свидетельства окупаемости есть, доказательств — нет.`,
      citations: [CIT_ROI_1, CIT_ROI_2],
      limitations: [
        {
          kind: "provider_stance",
          text: "Оценка окупаемости получена от организатора участия — стороны, заинтересованной в продаже участия.",
          source_ids: [
            "istochniki-video--podgotovka-k-organic-natural-v-dubae-vebinar-rusglobalekspor",
          ],
        },
        {
          kind: "reliability_note",
          text: "Цифры окупаемости названы организатором участия и не подтверждены независимым измерением.",
          source_ids: [
            "istochniki-video--podgotovka-k-organic-natural-v-dubae-vebinar-rusglobalekspor",
          ],
        },
      ],
      refused: false,
      meta: meta("03"),
    },
  },
  {
    id: "refused",
    question: exampleQuestion("refused"),
    keywords: ["сколько стоит", "2027", "цена стенда"],
    delayMs: 350,
    answer: {
      answer_markdown: `В материалах исследования нет данных, чтобы ответить на этот вопрос.

Чего не хватает:

- прайсов площадей и тарифов организаторов — в корпусе нет ни одного;
- материалов позднее 2024 года: самый свежий разобранный источник относится к 2024-му;
- сметы участия по конкретной выставке в Дубае — есть только описание подготовки
  к Organic & Natural без стоимости.

Цифру можно получить только у организатора выставки напрямую.`,
      citations: [],
      limitations: [
        {
          kind: "no_data",
          text: "В корпусе исследования нет материалов, содержащих запрошенные сведения.",
          source_ids: [],
        },
      ],
      refused: true,
      meta: meta("04", { latency_ms: 1620, cost_usd: 0.0061 }),
    },
  },
  {
    id: "upstream_error",
    question: exampleQuestion("upstream_error"),
    keywords: ["сбой генерации", "upstream"],
    delayMs: 250,
    answer: null,
  },
  {
    id: "slow",
    question: exampleQuestion("slow"),
    keywords: ["долгий ответ"],
    delayMs: 3500,
    answer: {
      answer_markdown: `Ответ пришёл после долгой генерации — сценарий существует, чтобы было видно
состояние ожидания.

Содержательно он повторяет разбор работы с контактами после выставки [1].`,
      citations: [{ ...CIT_FOLLOWUP, n: 1 }],
      limitations: [],
      refused: false,
      meta: meta("06", { latency_ms: 3500 }),
    },
  },
];

export function pickScenario(question: string): Scenario {
  const q = question.toLowerCase();
  const matched = SCENARIOS.find((s) => s.keywords.some((k) => q.includes(k)));
  return matched ?? SCENARIOS[0];
}
