// Точка включения моков. Импорт msw — динамический: пока VITE_USE_MOCKS не равен "on",
// ни msw, ни фикстуры не попадают в сборку и в сеть.

/**
 * Моки выключены по умолчанию: обычный запуск ходит в живой API (T-22).
 * Демонстрационные сценарии без бэкенда — `npm run dev:mock` (VITE_USE_MOCKS=on).
 */
export const mocksEnabled = (import.meta.env.VITE_USE_MOCKS ?? "off") === "on";

export async function startMocks(): Promise<void> {
  if (!mocksEnabled) return;
  const [{ setupWorker }, { handlers }] = await Promise.all([
    import("msw/browser"),
    import("./handlers"),
  ]);
  await setupWorker(...handlers).start({ onUnhandledRequest: "bypass", quiet: true });
}
