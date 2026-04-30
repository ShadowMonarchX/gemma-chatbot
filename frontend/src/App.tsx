import { useEffect, useState } from "react";
import Admin from "./Admin";
import Chat from "./Chat";
import { SKILLS } from "./skills";

type Tab = "chat" | "admin";

const THEME_KEY = "gemma-chatbot-theme";

function resolveInitialTheme(): "light" | "dark" {
  const saved = localStorage.getItem(THEME_KEY);
  if (saved === "light" || saved === "dark") {
    return saved;
  }
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

export default function App() {
  const [activeTab, setActiveTab] = useState<Tab>("chat");
  const [theme, setTheme] = useState<"light" | "dark">(() => resolveInitialTheme());

  useEffect(() => {
    document.documentElement.classList.toggle("dark", theme === "dark");
    localStorage.setItem(THEME_KEY, theme);
  }, [theme]);

  return (
    <main className="relative min-h-screen overflow-hidden bg-slate-100 px-3 py-6 font-body text-slate-900 transition-colors dark:bg-slate-950 dark:text-slate-100 sm:px-6">
      <div className="pointer-events-none absolute inset-0">
        <div className="absolute -left-20 top-[-120px] h-72 w-72 rounded-full bg-cyan-300/40 blur-3xl dark:bg-cyan-600/25" />
        <div className="absolute -right-24 bottom-[-80px] h-80 w-80 rounded-full bg-blue-300/30 blur-3xl dark:bg-indigo-600/25" />
      </div>

      <div className="relative mx-auto w-full max-w-6xl">
        <header className="mb-4 flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-slate-200/80 bg-white/75 px-3 py-3 backdrop-blur dark:border-slate-800 dark:bg-slate-900/70 sm:px-4">
          <div className="inline-flex items-center rounded-full border border-slate-200 bg-slate-50 p-1 dark:border-slate-700 dark:bg-slate-800">
            <button
              type="button"
              onClick={() => setActiveTab("chat")}
              className={`rounded-full px-4 py-1.5 text-sm font-semibold transition ${
                activeTab === "chat"
                  ? "bg-gradient-to-r from-cyan-500 to-blue-500 text-white"
                  : "text-slate-600 hover:text-slate-900 dark:text-slate-300 dark:hover:text-slate-100"
              }`}
            >
              Chat
            </button>
            <button
              type="button"
              onClick={() => setActiveTab("admin")}
              className={`rounded-full px-4 py-1.5 text-sm font-semibold transition ${
                activeTab === "admin"
                  ? "bg-gradient-to-r from-cyan-500 to-blue-500 text-white"
                  : "text-slate-600 hover:text-slate-900 dark:text-slate-300 dark:hover:text-slate-100"
              }`}
            >
              Admin
            </button>
          </div>

          <button
            type="button"
            onClick={() => setTheme((prev) => (prev === "light" ? "dark" : "light"))}
            className="rounded-xl border border-slate-300 bg-white px-3 py-1.5 text-xs font-semibold text-slate-700 transition hover:border-cyan-400 hover:text-cyan-700 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100 dark:hover:border-cyan-400 dark:hover:text-cyan-200"
          >
            {theme === "dark" ? "Light mode" : "Dark mode"}
          </button>
        </header>

        {activeTab === "chat" ? <Chat skills={SKILLS} /> : <Admin />}
      </div>
    </main>
  );
}
