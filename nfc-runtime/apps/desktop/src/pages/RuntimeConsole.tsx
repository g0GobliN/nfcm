import { type FormEvent, useEffect, useRef, useState } from "react";
import { consoleCommand } from "../lib/api";
import { useRuntime } from "../lib/RuntimeContext";

export default function RuntimeConsole() {
  const { snapshot, refresh } = useRuntime();
  const [lines, setLines] = useState<string[]>([]);
  const [input, setInput] = useState("");
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (snapshot?.console_lines) {
      setLines(snapshot.console_lines);
    }
  }, [snapshot?.console_lines]);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [lines]);

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    const cmd = input.trim();
    if (!cmd) return;
    setInput("");
    try {
      const out = await consoleCommand(cmd);
      setLines(out);
      await refresh();
    } catch (err) {
      setLines((prev) => [
        ...prev,
        `> ${cmd}`,
        err instanceof Error ? err.message : String(err),
      ]);
    }
  }

  return (
    <div className="mx-auto flex h-[calc(100vh-4rem)] max-w-4xl flex-col">
      <header className="mb-4">
        <h1 className="font-display text-2xl font-semibold text-white">Runtime Console</h1>
        <p className="mt-2 text-sm text-mist-dim">
          Commands: <code className="text-signal">compile …</code>,{" "}
          <code className="text-signal">status</code>,{" "}
          <code className="text-signal">unload</code>,{" "}
          <code className="text-signal">optimize</code>,{" "}
          <code className="text-signal">help</code>
        </p>
      </header>
      <div className="panel flex flex-1 flex-col overflow-hidden font-mono text-sm">
        <div className="flex-1 overflow-auto p-4 space-y-1">
          {lines.map((line, i) => (
            <div
              key={`${i}-${line.slice(0, 24)}`}
              className={line.startsWith(">") ? "text-signal" : "text-mist"}
            >
              {line}
            </div>
          ))}
          <div ref={endRef} />
        </div>
        <form
          onSubmit={(e) => void onSubmit(e)}
          className="flex border-t border-ink-800"
        >
          <span className="px-3 py-3 text-signal">›</span>
          <input
            className="flex-1 bg-transparent py-3 pr-4 outline-none"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="compile coding assistant"
            autoFocus
          />
        </form>
      </div>
    </div>
  );
}
