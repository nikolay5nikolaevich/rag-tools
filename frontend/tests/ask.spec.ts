import { expect, test } from "@playwright/test";

// Сценарий приёмки: задал вопрос → увидел ответ → раскрыл источник → отправил фидбек.
// Плюс проверки требований отображения ответа и источников.

test("задал вопрос → увидел ответ → раскрыл источник → отправил фидбек", async ({ page }) => {
  await page.goto("/");

  await page.getByLabel("Что хотите выяснить?").fill("Как организовать работу с контактами после выставки?");
  await page.getByRole("button", { name: "Спросить" }).click();

  const answer = page.getByTestId("answer");
  await expect(answer).toBeVisible();
  await expect(page.getByTestId("citations")).toBeVisible();

  // Клик по [n] в тексте ответа раскрывает конкретный фрагмент.
  await page.getByTestId("cite-ref-1").click();
  const body = page.getByTestId("citation-body-1");
  await expect(body).toBeVisible();
  await expect(page.getByTestId("heading-path-1")).toContainText("Работа с контактами после выставки");
  await expect(page.getByTestId("quote-1")).toContainText("within 48 hours");
  await expect(page.getByTestId("citation-url-1")).toHaveAttribute("href", /youtube/);

  // Фидбек уходит и подтверждается; текст комментария необязателен, но принимается.
  await page.getByTestId("feedback-comment").fill("Цитата отвечает на вопрос");
  await page.getByTestId("feedback-useful").click();
  await expect(page.getByTestId("feedback-sent")).toContainText("отзыв записан");
});

test("источник без url показан без ссылки и не спрятан", async ({ page }) => {
  await page.goto("/");
  await page.getByTestId("example-normal").click();
  await expect(page.getByTestId("answer")).toBeVisible();

  await page.getByTestId("cite-ref-2").click();
  await expect(page.getByTestId("citation-2")).toContainText("После мероприятия");
  await expect(page.getByTestId("citation-nourl-2")).toBeVisible();
  await expect(page.getByTestId("citation-url-2")).toHaveCount(0);
});

test("ограничения рендерятся из limitations, каждый kind со своей формулировкой", async ({ page }) => {
  await page.goto("/");
  await page.getByTestId("example-single_org").click();

  await expect(page.getByTestId("limitation-single_org")).toContainText("RusGlobalExport");
  await expect(page.getByTestId("limitation-single_org")).toContainText("Одна организация");
  await expect(page.getByTestId("limitation-provider_stance")).toContainText(
    "Заинтересованная сторона",
  );
});

test("организатор-провайдер: позиция источника названа в ограничениях", async ({ page }) => {
  await page.goto("/");
  await page.getByTestId("example-provider_stance").click();

  await expect(page.getByTestId("answer")).toBeVisible();
  await expect(page.getByTestId("limitation-provider_stance")).toContainText("организатора участия");
});

test("отказ оформлен как полноценный ответ, а не как ошибка", async ({ page }) => {
  await page.goto("/");
  await page.getByTestId("example-refused").click();

  const refused = page.getByTestId("answer-refused");
  await expect(refused).toBeVisible();
  await expect(refused).toContainText("Чего не хватает");
  await expect(page.getByTestId("limitation-no_data")).toBeVisible();
  await expect(page.getByTestId("error")).toHaveCount(0);
  // Отказ — тоже ответ: оценить его можно.
  await expect(page.getByTestId("feedback-useful")).toBeVisible();
});

test("ошибка upstream показана как ошибка сервиса с возможностью повторить", async ({ page }) => {
  await page.goto("/");
  await page.getByTestId("example-upstream_error").click();

  const error = page.getByTestId("error");
  await expect(error).toBeVisible();
  await expect(error).toContainText("Сервис генерации ответа временно недоступен");
  await expect(page.getByRole("button", { name: "Повторить" })).toBeVisible();
  await expect(page.getByTestId("answer")).toHaveCount(0);
});

test("долгий ответ показывает скелетон ожидания", async ({ page }) => {
  await page.goto("/");
  await page.getByTestId("example-slow").click();

  await expect(page.getByTestId("skeleton")).toBeVisible();
  await expect(page.getByTestId("answer")).toBeVisible({ timeout: 15000 });
  await expect(page.getByTestId("skeleton")).toHaveCount(0);
});

test("процент уверенности не показывается ни на одном экране", async ({ page }) => {
  const forbidden = /уверенност|confidence|\b\d{1,3}\s?%\s?(уверен|точн)/i;

  await page.goto("/");
  await page.getByTestId("example-normal").click();
  await expect(page.getByTestId("answer")).toBeVisible();
  expect(await page.locator("body").innerText()).not.toMatch(forbidden);

  await page.getByRole("link", { name: "Библиотека" }).click();
  await expect(page.getByTestId("library-table")).toBeVisible();
  expect(await page.locator("body").innerText()).not.toMatch(forbidden);
});

test("библиотека показывает метаданные, включая пересказы и пустые значения", async ({ page }) => {
  await page.goto("/library");

  const table = page.getByTestId("library-table");
  await expect(table).toBeVisible();
  await expect(table).toContainText("Организация спикера");

  const derived = page.getByTestId("source-tezisy-video--mezhdunarodnye-vystavki-i-kak-na-nih-rabotat");
  await expect(derived).toContainText("пересказ");

  // Источник без speaker_org показан честно, а не с догадкой.
  await expect(page.getByTestId("source-istochniki-video--z-ro-1-hebdo-vivatech-2021")).toContainText(
    "не указано",
  );
});

test("вопрос, ответ и раскрытый источник сохраняются после перезагрузки", async ({ page }) => {
  await page.goto("/");
  await page.getByTestId("example-normal").click();
  await expect(page.getByTestId("answer")).toBeVisible();
  await page.getByTestId("cite-ref-1").click();
  await expect(page.getByTestId("citation-body-1")).toBeVisible();
  await page.getByTestId("feedback-comment").fill("черновик комментария");

  await page.reload();

  await expect(page.getByLabel("Что хотите выяснить?")).toHaveValue(
    "Как организовать работу с контактами после выставки?",
  );
  await expect(page.getByTestId("answer")).toBeVisible();
  await expect(page.getByTestId("citation-body-1")).toBeVisible();
  await expect(page.getByTestId("feedback-comment")).toHaveValue("черновик комментария");
});

test("испорченная запись в хранилище не ломает экран", async ({ page }) => {
  await page.addInitScript(() => {
    localStorage.setItem("ergovault:ask:answer:v1", '{"answer_markdown": 42}');
    localStorage.setItem("ergovault:ask:feedback:v1", "не json");
  });
  await page.goto("/");
  await expect(page.getByLabel("Что хотите выяснить?")).toBeVisible();
  await expect(page.getByTestId("answer")).toHaveCount(0);
});

test("поиск в библиотеке сужает список и переживает перезагрузку", async ({ page }) => {
  await page.goto("/library");
  await expect(page.getByTestId("library-table")).toBeVisible();

  await page.getByLabel("Поиск по библиотеке").fill("GITEX");
  const rows = page.getByTestId("library-table").locator("> li");
  await expect(rows).toHaveCount(1);
  await expect(rows.first()).toContainText("GITEX");

  await page.reload();
  await expect(page.getByLabel("Поиск по библиотеке")).toHaveValue("GITEX");
  await expect(rows).toHaveCount(1);
});
