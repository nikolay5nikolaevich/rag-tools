import { NavLink, Route, Routes } from "react-router-dom";
import { Ask } from "./screens/Ask";
import { Library } from "./screens/Library";

export function App() {
  return (
    <>
      <header className="top">
        <NavLink to="/" end className="brand">
          ErgoVault
        </NavLink>
        <nav aria-label="Разделы">
          <NavLink to="/" end>
            Вопросы
          </NavLink>
          <NavLink to="/library">Библиотека</NavLink>
        </nav>
      </header>

      <main>
        <Routes>
          <Route path="/" element={<Ask />} />
          <Route path="/library" element={<Library />} />
          <Route path="*" element={<p className="screen">Такой страницы нет.</p>} />
        </Routes>
      </main>
    </>
  );
}
