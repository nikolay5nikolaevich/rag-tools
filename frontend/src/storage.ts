// Состояние экрана в localStorage: после закрытия вкладки вопрос и ответ остаются на месте.
// Хранилище может быть недоступно (приватный режим, запрет сайта) — тогда экран просто
// работает без памяти. Ключ версионирован: при смене формы данных старую запись игнорируем.

import { useEffect, useState } from "react";

const PREFIX = "ergovault:";

function load<T>(key: string, fallback: T, isValid: (value: unknown) => value is T): T {
  try {
    const raw = localStorage.getItem(PREFIX + key);
    if (raw === null) return fallback;
    const value: unknown = JSON.parse(raw);
    return isValid(value) ? value : fallback;
  } catch {
    return fallback;
  }
}

function save(key: string, value: unknown) {
  try {
    localStorage.setItem(PREFIX + key, JSON.stringify(value));
  } catch {
    // Переполнение или запрет хранилища не должны ломать экран.
  }
}

/** useState, значение которого переживает перезагрузку страницы. */
export function usePersistentState<T>(
  key: string,
  fallback: T,
  isValid: (value: unknown) => value is T,
) {
  const [value, setValue] = useState<T>(() => load(key, fallback, isValid));
  useEffect(() => save(key, value), [key, value]);
  return [value, setValue] as const;
}
