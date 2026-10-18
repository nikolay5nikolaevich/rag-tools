import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Прокси нужен, чтобы обычный запуск работал без настройки: фронт ходит на свой origin,
// dev-сервер передаёт /api и /healthz бэкенду. Другой адрес API задаётся переменной
// VITE_API_BASE_URL — тогда запрос идёт напрямую и его разрешает CORS на бэкенде.
const API_TARGET = process.env.EV_API_URL ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target: API_TARGET, changeOrigin: true },
      "/healthz": { target: API_TARGET, changeOrigin: true },
    },
  },
  preview: { port: 5173 },
});
