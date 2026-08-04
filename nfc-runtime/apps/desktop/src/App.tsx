import { NavLink, Route, Routes } from "react-router-dom";
import { RuntimeProvider, useRuntime } from "./lib/RuntimeContext";
import Dashboard from "./pages/Dashboard";
import ModelManager from "./pages/ModelManager";
import TaskCompilerPage from "./pages/TaskCompiler";
import RuntimeConsole from "./pages/RuntimeConsole";
import ChatPlayground from "./pages/ChatPlayground";
import DeveloperTools from "./pages/DeveloperTools";

const nav = [
  { to: "/", label: "Dashboard" },
  { to: "/models", label: "Models" },
  { to: "/compiler", label: "Task Compiler" },
  { to: "/console", label: "Console" },
  { to: "/chat", label: "Chat" },
  { to: "/devtools", label: "Dev Tools" },
];

function Shell() {
  const { snapshot, refresh } = useRuntime();
  const status = snapshot?.status ?? "starting";

  return (
    <div className="flex h-full min-h-screen">
      <aside className="flex w-56 shrink-0 flex-col border-r border-ink-800 bg-ink-900/60 px-4 py-6">
        <div className="mb-8">
          <div className="flex items-center gap-3">
            <img
              src="/logo.png"
              alt="NFCM"
              className="h-10 w-10 rounded-lg shadow-panel"
            />
            <div>
              <div className="font-display text-xl font-bold tracking-tight text-white">
                NFCM
              </div>
              <div className="text-xs text-mist-dim">Neural Foundation Runtime</div>
            </div>
          </div>
          <div className="mt-4 flex items-center gap-2 text-xs">
            <span
              className={`h-2 w-2 rounded-full ${
                status === "running"
                  ? "bg-signal"
                  : status === "compiling"
                    ? "bg-amber-soft"
                    : "bg-mist-dim"
              }`}
            />
            <span className="capitalize text-mist">{status}</span>
          </div>
        </div>
        <nav className="flex flex-1 flex-col gap-1">
          {nav.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === "/"}
              className={({ isActive }) =>
                `rounded-md px-3 py-2 text-sm transition ${
                  isActive
                    ? "bg-ink-800 text-signal"
                    : "text-mist-dim hover:bg-ink-800/60 hover:text-mist"
                }`
              }
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
        <button type="button" className="btn-ghost mt-4 w-full text-xs" onClick={() => refresh()}>
          Refresh
        </button>
        <p className="mt-4 text-[10px] leading-relaxed text-mist-dim">
          Phase 1 platform. Mock generator only — no trained LLM claims.
        </p>
      </aside>
      <main className="flex-1 overflow-auto p-8">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/models" element={<ModelManager />} />
          <Route path="/compiler" element={<TaskCompilerPage />} />
          <Route path="/console" element={<RuntimeConsole />} />
          <Route path="/chat" element={<ChatPlayground />} />
          <Route path="/devtools" element={<DeveloperTools />} />
        </Routes>
      </main>
    </div>
  );
}

export default function App() {
  return (
    <RuntimeProvider>
      <Shell />
    </RuntimeProvider>
  );
}
