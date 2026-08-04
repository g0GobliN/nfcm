import { NavLink, Route, Routes, useLocation } from "react-router-dom";
import { RuntimeProvider, useRuntime } from "./lib/RuntimeContext";
import Dashboard from "./pages/Dashboard";
import ModelManager from "./pages/ModelManager";
import TaskCompilerPage from "./pages/TaskCompiler";
import RuntimeConsole from "./pages/RuntimeConsole";
import ChatPlayground from "./pages/ChatPlayground";
import DeveloperTools from "./pages/DeveloperTools";
import SettingsPage from "./pages/Settings";

const nav = [
  { to: "/", label: "Dashboard" },
  { to: "/models", label: "Models" },
  { to: "/compiler", label: "Compiler" },
  { to: "/chat", label: "Chat" },
  { to: "/console", label: "Console" },
  { to: "/settings", label: "Settings" },
  { to: "/devtools", label: "Dev Tools" },
];

function Shell() {
  const { snapshot, refresh } = useRuntime();
  const location = useLocation();
  const status = snapshot?.status ?? "starting";
  const brain = snapshot?.active_model?.name;
  const isChat = location.pathname === "/chat";

  return (
    <div className="flex h-screen overflow-hidden">
      <aside className="relative flex w-56 shrink-0 flex-col overflow-y-auto border-r border-ink-800/60 bg-ink-900/30 px-3 py-5 backdrop-blur-xl">
        <div className="mb-6 flex items-center gap-2.5 px-2">
          <img
            src="/logo.png"
            alt="NFCM"
            className="h-9 w-9 rounded-lg ring-1 ring-signal/20"
          />
          <div className="min-w-0">
            <div className="font-display text-lg font-bold tracking-tight text-white">NFCM</div>
            <div className="truncate text-[10px] text-mist-dim">Local runtime</div>
          </div>
        </div>

        <div className="mb-4 flex items-center gap-2 px-2">
          <span
            className={`h-1.5 w-1.5 shrink-0 rounded-full ${
              status === "running"
                ? "bg-signal"
                : status === "compiling"
                  ? "bg-amber-soft"
                  : "bg-mist-dim"
            }`}
          />
          <span className="truncate text-xs capitalize text-mist-dim">{status}</span>
        </div>
        {brain && (
          <p className="mb-4 truncate px-2 text-[11px] text-mist-dim" title={brain}>
            {brain}
          </p>
        )}

        <nav className="flex flex-1 flex-col gap-0.5">
          {nav.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === "/"}
              className={({ isActive }) =>
                `nav-link ${
                  isActive
                    ? "bg-ink-800/80 text-white"
                    : "text-mist-dim hover:bg-ink-800/40 hover:text-mist"
                }`
              }
            >
              {item.label}
            </NavLink>
          ))}
        </nav>

        <div className="mt-4 border-t border-ink-800/60 pt-3">
          <button type="button" className="btn-ghost w-full text-xs" onClick={() => refresh()}>
            Refresh
          </button>
        </div>
      </aside>

      <main
        className={`flex min-h-0 min-w-0 flex-1 flex-col ${
          isChat ? "overflow-hidden" : "overflow-y-auto"
        }`}
      >
        <div
          className={
            isChat
              ? "flex min-h-0 flex-1 flex-col"
              : "mx-auto w-full max-w-5xl px-6 py-6 pb-10 md:px-10 md:py-8"
          }
        >
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/models" element={<ModelManager />} />
            <Route path="/compiler" element={<TaskCompilerPage />} />
            <Route path="/console" element={<RuntimeConsole />} />
            <Route path="/chat" element={<ChatPlayground />} />
            <Route path="/settings" element={<SettingsPage />} />
            <Route path="/devtools" element={<DeveloperTools />} />
          </Routes>
        </div>
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
