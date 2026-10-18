// Рендер answer_markdown: абзацы, нумерованные и маркированные списки, **жирный**
// и кликабельные ссылки [n]. Полноценный markdown-движок ради этого не нужен.
//
// [n] без соответствующей Citation отображается как обычный текст: контракт §7 гарантирует,
// что таких не бывает, но фронт не имитирует ссылку на несуществующий фрагмент.

import type { JSX } from "react";
import type { Citation } from "../api/types";

interface Props {
  markdown: string;
  citations: Citation[];
  onCitationClick: (n: number) => void;
}

function inline(text: string, citations: Citation[], onClick: (n: number) => void): JSX.Element[] {
  const nodes: JSX.Element[] = [];
  const pattern = /\*\*(.+?)\*\*|\[(\d+)\]/g;
  let last = 0;
  let match: RegExpExecArray | null;
  let key = 0;

  while ((match = pattern.exec(text)) !== null) {
    if (match.index > last) {
      nodes.push(<span key={key++}>{text.slice(last, match.index)}</span>);
    }
    if (match[1] !== undefined) {
      nodes.push(<strong key={key++}>{match[1]}</strong>);
    } else {
      const n = Number(match[2]);
      const known = citations.some((c) => c.n === n);
      nodes.push(
        known ? (
          <button
            key={key++}
            type="button"
            className="cite-ref"
            data-testid={`cite-ref-${n}`}
            onClick={() => onClick(n)}
            aria-label={`Открыть фрагмент ${n}`}
          >
            [{n}]
          </button>
        ) : (
          <span key={key++}>[{n}]</span>
        ),
      );
    }
    last = match.index + match[0].length;
  }
  if (last < text.length) nodes.push(<span key={key++}>{text.slice(last)}</span>);
  return nodes;
}

type Block =
  | { type: "p"; lines: string[] }
  | { type: "ol"; items: string[] }
  | { type: "ul"; items: string[] };

function parse(markdown: string): Block[] {
  const blocks: Block[] = [];
  for (const raw of markdown.split(/\n{2,}/)) {
    const chunk = raw.trim();
    if (!chunk) continue;
    const lines = chunk.split("\n").map((l) => l.trim());

    if (lines.every((l) => /^\d+\.\s+/.test(l))) {
      blocks.push({ type: "ol", items: lines.map((l) => l.replace(/^\d+\.\s+/, "")) });
    } else if (lines.every((l) => /^[-*]\s+/.test(l))) {
      blocks.push({ type: "ul", items: lines.map((l) => l.replace(/^[-*]\s+/, "")) });
    } else {
      blocks.push({ type: "p", lines });
    }
  }
  return blocks;
}

export function AnswerBody({ markdown, citations, onCitationClick }: Props) {
  return (
    <div className="answer-body" data-testid="answer-body">
      {parse(markdown).map((block, i) => {
        if (block.type === "p") {
          return <p key={i}>{inline(block.lines.join(" "), citations, onCitationClick)}</p>;
        }
        const items = block.items.map((item, j) => (
          <li key={j}>{inline(item, citations, onCitationClick)}</li>
        ));
        return block.type === "ol" ? <ol key={i}>{items}</ol> : <ul key={i}>{items}</ul>;
      })}
    </div>
  );
}
