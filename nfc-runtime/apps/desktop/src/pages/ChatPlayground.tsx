import { type FormEvent, useEffect, useRef, useState } from "react";
import { runInference } from "../lib/api";
import { useRuntime } from "../lib/RuntimeContext";
import { formatMb } from "../lib/types";

interface Msg {
  role: "user" | "assistant";
  text: string;
  meta?: string;
}

export default function ChatPlayground() {
  const { snapshot } = useRuntime();
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  const model = snapshot?.active_model;
  const mem = snapshot?.memory;
  const backend = snapshot?.inference_backend;

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, busy]);

  async function onSend(e: FormEvent) {
    e.preventDefault();
    const prompt = input.trim();
    if (!prompt || busy) return;
    setInput("");
    setMessages((m) => [...m, { role: "user", text: prompt }]);
    setBusy(true);
    try {
      const resp = await runInference(prompt, 256);
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          text: resp.text,
          meta: `${resp.backend} · ${resp.tokens_in}→${resp.tokens_out} tokens · mock=${resp.is_mock}`,
        },
      ]);
    } catch (err) {
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          text: err instanceof Error ? err.message : String(err),
        },
      ]);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="page-enter mx-auto flex h-[calc(100vh-3.5rem)] max-w-3xl flex-col">
      <header className="mb-5 flex flex-wrap items-end justify-between gap-3">
        <div>
          <p className="label mb-1">Conversation</p>
          <h1 className="font-display text-4xl font-extrabold tracking-tight text-white">Chat</h1>
          <p className="mt-1 text-sm text-mist-dim">
            {backend?.is_mock === false
              ? "Local weights — private to this machine."
              : "Mock path — load a real GGUF for live answers."}
          </p>
        </div>
        <div className="rounded-xl border border-ink-700/80 bg-ink-900/50 px-4 py-3 text-right text-xs backdrop-blur-sm">
          <div className="font-medium text-mist">{model?.name ?? "No model loaded"}</div>
          <div className="mt-1 text-mist-dim">
            {backend?.name ?? "—"}
            {backend?.ready ? " · ready" : " · idle"}
          </div>
          <div className="mt-1 font-mono text-mist-dim">
            {mem ? `${formatMb(mem.used_bytes)} / ${formatMb(mem.max_ram_bytes)}` : "—"}
          </div>
        </div>
      </header>

      <div className="panel flex flex-1 flex-col overflow-hidden">
        <div className="flex-1 space-y-4 overflow-auto p-5">
          {messages.length === 0 && (
            <div className="flex h-full min-h-[12rem] flex-col items-center justify-center text-center">
              <div className="mb-3 h-12 w-12 rounded-2xl bg-signal/15 ring-1 ring-signal/30" />
              <p className="font-display text-lg font-semibold text-white">Ready when you are</p>
              <p className="mt-2 max-w-sm text-sm text-mist-dim">
                {model
                  ? "Ask anything. Answers come from your local GGUF / probe — not the cloud."
                  : "Load a brain in Models first, then come back to chat."}
              </p>
            </div>
          )}
          {messages.map((m, i) => (
            <div
              key={i}
              className={`animate-fade-up max-w-[88%] px-4 py-3 text-sm leading-relaxed whitespace-pre-wrap ${
                m.role === "user"
                  ? "ml-auto rounded-2xl rounded-br-md bg-signal/15 text-mist ring-1 ring-signal/25"
                  : "rounded-2xl rounded-bl-md bg-ink-950/80 text-mist ring-1 ring-ink-700"
              }`}
            >
              {m.text}
              {m.meta && (
                <div className="mt-2 font-mono text-[10px] text-mist-dim">{m.meta}</div>
              )}
            </div>
          ))}
          {busy && (
            <div className="animate-fade-up max-w-[60%] rounded-2xl rounded-bl-md bg-ink-950/80 px-4 py-3 text-sm text-mist-dim ring-1 ring-ink-700">
              <span className="inline-flex gap-1">
                <span className="animate-pulse-dot h-1.5 w-1.5 rounded-full bg-signal" />
                <span className="animation-delay-150 animate-pulse-dot h-1.5 w-1.5 rounded-full bg-signal" />
                <span className="animation-delay-300 animate-pulse-dot h-1.5 w-1.5 rounded-full bg-signal" />
              </span>
              <span className="ml-2">Thinking locally…</span>
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        <form
          onSubmit={(e) => void onSend(e)}
          className="flex gap-2 border-t border-ink-800/80 bg-ink-950/30 p-3"
        >
          <input
            className="field flex-1 disabled:opacity-50"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={model ? "Message your local brain…" : "Load a model first (Models → Load)"}
            disabled={!model || busy}
          />
          <button type="submit" className="btn-primary px-5" disabled={!model || busy}>
            Send
          </button>
        </form>
        {!model && (
          <p className="border-t border-ink-800/80 px-4 py-2.5 text-xs text-amber-soft">
            Chat is locked until a brain is loaded — open <strong>Models</strong> and click{" "}
            <strong>Load</strong>.
          </p>
        )}
      </div>
    </div>
  );
}
