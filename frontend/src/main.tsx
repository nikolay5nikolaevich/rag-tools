import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import { App } from "./App";
import { startMocks } from "./mocks/browser";
import "./styles.css";

async function bootstrap() {
  await startMocks();
  const container = document.getElementById("root");
  if (!container) throw new Error("Не найден #root");
  createRoot(container).render(
    <StrictMode>
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </StrictMode>,
  );
}

void bootstrap();
