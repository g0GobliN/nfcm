import { type FormEvent, type KeyboardEvent, useEffect, useRef, useState } from "react";
import { flushSync } from "react-dom";
import { runInference } from "../lib/api";
import { useRuntime } from "../lib/RuntimeContext";

interface Msg {
  role: "user" | "assistant";
  text: string;
  meta?: string;
}

const STATUS_STEPS = ["Thinking", "Reading", "Writing", "Finishing"] as const;

function paintFrame(): Promise<void> {
  return new Promise((resolve) => {
    requestAnimationFrame(() => requestAnimationFrame(() => resolve()));
  });
}

export default function ChatPlayground() {
  const { snapshot } = useRuntime();
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [statusIdx, setStatusIdx] = useState(0);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const model = snapshot?.active_model;
  const backend = snapshot?.inference_backend;

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, busy, statusIdx]);

  useEffect(() => {
    if (!busy) {
      setStatusIdx(0);
      return;
    }
    const id = window.setInterval(() => {
      setStatusIdx((i) => (i + 1) % STATUS_STEPS.length);
    }, 2400);
    return () => window.clearInterval(id);
  }, [busy]);

  async function onSend(e?: FormEvent) {
    e?.preventDefault();
    const prompt = input.trim();
    if (!prompt || busy || !model) return;
    setInput("");
    if (inputRef.current) inputRef.current.style.height = "auto";
    flushSync(() => {
      setMessages((m) => [...m, { role: "user", text: prompt }]);
      setBusy(true);
      setStatusIdx(0);
    });
    await paintFrame();
    try {
      const resp = await runInference(prompt, 256);
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          text: resp.text,
          meta: `${resp.backend} · ${resp.tokens_in}→${resp.tokens_out}`,
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
      inputRef.current?.focus();
    }
  }

  function onKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void onSend();
    }
  }

  function onInput(e: React.ChangeEvent<HTMLTextAreaElement>) {
    setInput(e.target.value);
    const el = e.target;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  }

  return (
    <div className="page-enter flex h-full min-h-0 flex-col">
      {/* Slim top bar */}
      <header className="flex shrink-0 items-center justify-between gap-4 border-b border-ink-800/50 px-6 py-3 md:px-10">
        <div className="min-w-0">
          <h1 className="font-display text-base font-semibold text-white">Chat</h1>
          <p className="truncate text-xs text-mist-dim">
            {model?.name ?? "No model"}
            {backend?.name ? ` · ${backend.name}` : ""}
          </p>
        </div>
        {!model && (
          <p className="shrink-0 text-xs text-amber-soft">Load a model in Models first</p>
        )}
      </header>

      {/* Scrollable messages — padding-bottom keeps last line above composer */}
      <div className="min-h-0 flex-1 overflow-y-auto">
        <div className="mx-auto flex min-h-full max-w-2xl flex-col px-5 pb-8 pt-6 md:px-6">
          {messages.length === 0 && !busy && (
            <div className="flex flex-1 flex-col items-center justify-center py-16 text-center">
              <p className="font-display text-xl font-semibold text-white/90">How can I help?</p>
              <p className="mt-2 max-w-sm text-sm text-mist-dim">
                {model
                  ? "Messages stay on this machine."
                  : "Open Models, load a brain, then come back."}
              </p>
            </div>
          )}

          <div className="mt-auto space-y-6">
            {messages.map((m, i) =>
              m.role === "user" ? (
                <div key={i} className="animate-fade-up flex justify-end">
                  <div className="max-w-[85%] rounded-2xl bg-ink-800/80 px-4 py-2.5 text-[15px] leading-relaxed text-mist">
                    {m.text}
                  </div>
                </div>
              ) : (
                <div key={i} className="animate-fade-up">
                  <div className="whitespace-pre-wrap text-[15px] leading-7 text-mist">
                    {m.text}
                  </div>
                  {m.meta && (
                    <p className="mt-2 font-mono text-[10px] text-mist-dim/70">{m.meta}</p>
                  )}
                </div>
              ),
            )}

            {busy && (
              <div className="animate-fade-up py-1" aria-live="polite">
                <span
                  key={statusIdx}
                  className="claude-shimmer text-[15px] font-medium tracking-tight"
                >
                  {STATUS_STEPS[statusIdx]}
                </span>
              </div>
            )}
            <div ref={bottomRef} className="h-2" />
          </div>
        </div>
      </div>

      {/* Composer — always fully visible */}
      <div className="shrink-0 border-t border-ink-800/50 bg-ink-950/40 px-4 pb-5 pt-3 md:px-10">
        <form
          onSubmit={(e) => void onSend(e)}
          className="mx-auto flex max-w-2xl items-end gap-2 rounded-2xl border border-ink-700/80 bg-ink-900/60 px-3 py-2 shadow-sm backdrop-blur-md"
        >
          <textarea
            ref={inputRef}
            rows={1}
            className="max-h-40 min-h-[44px] flex-1 resize-none bg-transparent px-2 py-2.5 text-[15px] leading-relaxed text-mist outline-none placeholder:text-mist-dim/60 disabled:opacity-40"
            value={input}
            onChange={onInput}
            onKeyDown={onKeyDown}
            placeholder={model ? "Message…" : "Load a model first"}
            disabled={!model || busy}
          />
          <button
            type="submit"
            className="mb-1 shrink-0 rounded-xl bg-signal px-4 py-2 text-sm font-semibold text-ink-950 transition hover:bg-signal-dim disabled:cursor-not-allowed disabled:opacity-35"
            disabled={!model || busy || !input.trim()}
          >
            Send
          </button>
        </form>
        <p className="mx-auto mt-2 max-w-2xl text-center text-[11px] text-mist-dim/60">
          Enter to send · Shift+Enter for newline
        </p>
      </div>
    </div>
  );
}
