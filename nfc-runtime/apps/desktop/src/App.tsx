import { NavLink, Route, Routes } from "react-router-dom";
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
  const status = snapshot?.status ?? "starting";
  const brain = snapshot?.active_model?.name;

  return (
    <div className="flex h-full min-h-screen">
      <aside className="relative flex w-[15.5rem] shrink-0 flex-col border-r border-ink-800/80 bg-ink-900/40 px-4 py-6 backdrop-blur-xl">
        <div className="pointer-events-none absolute inset-x-0 top-0 h-32 bg-gradient-to-b from-signal/10 to-transparent" />

        <div className="relative mb-8">
          <div className="flex items-center gap-3">
            <img
              src="/logo.png"
              alt="NFCM"
              className="h-11 w-11 rounded-xl shadow-glow ring-1 ring-signal/20"
            />
            <div>
              <div className="font-display text-2xl font-extrabold tracking-tight text-white">
                NFCM
              </div>
              <div className="text-[11px] tracking-wide text-mist-dim">Local neural runtime</div>
            </div>
          </div>

          <div className="mt-5 flex items-center gap-2 rounded-lg border border-ink-700/80 bg-ink-950/40 px-3 py-2">
            <span
              className={`h-2 w-2 rounded-full ${
                status === "running"
                  ? "animate-pulse-dot bg-signal shadow-[0_0_8px_rgba(62,207,142,0.8)]"
                  : status === "compiling"
                    ? "animate-pulse-dot bg-amber-soft"
                    : "bg-mist-dim"
              }`}
            />
            <span className="text-xs capitalize text-mist">{status}</span>
          </div>
          {brain && (
            <p className="mt-2 truncate px-1 text-[11px] text-mist-dim" title={brain}>
              Brain · {brain}
            </p>
          )}
        </div>

        <nav className="relative flex flex-1 flex-col gap-0.5">
          {nav.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === "/"}
              className={({ isActive }) =>
                `nav-link ${
                  isActive
                    ? "bg-ink-800/90 text-signal shadow-[inset_3px_0_0_0_#3ecf8e]"
                    : "text-mist-dim hover:bg-ink-800/40 hover:text-mist"
                }`
              }
            >
              {item.label}
            </NavLink>
          ))}
        </nav>

        <div className="relative mt-4 space-y-3 border-t border-ink-800/80 pt-4">
          <button type="button" className="btn-ghost w-full text-xs" onClick={() => refresh()}>
            Refresh
          </button>
          <p className="px-1 text-[10px] leading-relaxed text-mist-dim">
            Runs on your machine. GGUF / latent — never cloud.
          </p>
        </div>
      </aside>

      <main className="flex-1 overflow-auto p-6 md:p-9">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/models" element={<ModelManager />} />
          <Route path="/compiler" element={<TaskCompilerPage />} />
          <Route path="/console" element={<RuntimeConsole />} />
          <Route path="/chat" element={<ChatPlayground />} />
          <Route path="/settings" element={<SettingsPage />} />
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
