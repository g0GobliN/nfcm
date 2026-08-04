import { type FormEvent, useState } from "react";
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

  const model = snapshot?.active_model;
  const mem = snapshot?.memory;

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
          meta: `${resp.tokens_in}→${resp.tokens_out} tokens · mock=${resp.is_mock}`,
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
    <div className="mx-auto flex h-[calc(100vh-4rem)] max-w-3xl flex-col">
      <header className="mb-4 flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="font-display text-3xl font-bold text-white">Chat Playground</h1>
          <p className="mt-1 text-sm text-mist-dim">
            Local mock inference — not a production LLM.
          </p>
        </div>
        <div className="text-right text-xs text-mist-dim">
          <div className="text-mist">{model?.name ?? "No model loaded"}</div>
          <div>
            Memory {mem ? `${formatMb(mem.used_bytes)} / ${formatMb(mem.max_ram_bytes)}` : "—"}
          </div>
        </div>
      </header>

      <div className="panel flex flex-1 flex-col overflow-hidden">
        <div className="flex-1 space-y-4 overflow-auto p-4">
          {messages.length === 0 && (
            <p className="text-sm text-mist-dim">
              Compile or load a brain, then send a prompt. Responses are labeled mock.
            </p>
          )}
          {messages.map((m, i) => (
            <div
              key={i}
              className={`max-w-[90%] rounded-md px-3 py-2 text-sm whitespace-pre-wrap ${
                m.role === "user"
                  ? "ml-auto bg-ink-800 text-mist"
                  : "bg-ink-950 text-mist border border-ink-700"
              }`}
            >
              {m.text}
              {m.meta && <div className="mt-2 text-[10px] text-mist-dim">{m.meta}</div>}
            </div>
          ))}
        </div>
        <form
          onSubmit={(e) => void onSend(e)}
          className="flex gap-2 border-t border-ink-800 p-3"
        >
          <input
            className="flex-1 rounded-md border border-ink-700 bg-ink-950 px-3 py-2 text-sm outline-none focus:border-signal/50"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask the active brain…"
            disabled={!model || busy}
          />
          <button type="submit" className="btn-primary" disabled={!model || busy}>
            Send
          </button>
        </form>
      </div>
    </div>
  );
}
