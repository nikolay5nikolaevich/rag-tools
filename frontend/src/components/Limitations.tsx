// Блок «Ограничения». Текст приходит с бэкенда готовым (контракт §7, спека 5.3):
// фронт его не сочиняет и не переписывает. Фронт добавляет только заголовок вида
// ограничения и значок — это подпись к kind, а не переформулировка text.

import type { Limitation, LimitationKind } from "../api/types";

const KIND_LABEL: Record<LimitationKind, { icon: string; label: string }> = {
  single_org: { icon: "◆", label: "Одна организация" },
  provider_stance: { icon: "▲", label: "Заинтересованная сторона" },
  reliability_note: { icon: "●", label: "Оговорка источника" },
  no_data: { icon: "○", label: "Данных нет" },
  derived_only: { icon: "◇", label: "Только пересказ" },
};

export function Limitations({ limitations }: { limitations: Limitation[] }) {
  if (limitations.length === 0) return null;

  return (
    <section className="panel" data-testid="limitations">
      <h2>Ограничения</h2>
      <ul className="limitations">
        {limitations.map((limitation, i) => {
          const kind = KIND_LABEL[limitation.kind];
          return (
            <li key={`${limitation.kind}-${i}`} data-testid={`limitation-${limitation.kind}`}>
              <span className="limitation-icon" aria-hidden="true">
                {kind?.icon ?? "•"}
              </span>
              <div>
                <div className="limitation-kind">{kind?.label ?? limitation.kind}</div>
                <p className="limitation-text">{limitation.text}</p>
                {limitation.source_ids.length > 0 && (
                  <p className="limitation-sources">
                    Материалы: {limitation.source_ids.join(", ")}
                  </p>
                )}
              </div>
            </li>
          );
        })}
      </ul>
    </section>
  );
}
