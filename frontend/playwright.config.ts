import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  fullyParallel: true,
  reporter: [["list"]],
  use: {
    baseURL: "http://127.0.0.1:5173",
    trace: "retain-on-failure",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  // Браузерные сценарии идут на моках (`dev:mock`): проверяется поведение экранов,
  // включая отказ, сбой генерации и долгий ответ, — без вызовов модели и денег на них.
  webServer: {
    command: "npm run dev:mock -- --host 127.0.0.1 --port 5173",
    url: "http://127.0.0.1:5173",
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
});
