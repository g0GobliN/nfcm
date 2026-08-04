import { useState } from "react";
import { compileBrain, previewTask } from "../lib/api";
import { useRuntime } from "../lib/RuntimeContext";
import type { TaskProfile, TaskType } from "../lib/types";

const STEPS = [
  "Analyzing task",
  "Selecting components",
  "Generating model",
  "Optimizing memory",
] as const;

const CATEGORIES: { id: TaskType; label: string }[] = [
  { id: "coding", label: "Coding" },
  { id: "math", label: "Math" },
  { id: "writing", label: "Writing" },
  { id: "research", label: "Research" },
  { id: "custom", label: "Custom" },
];

export default function TaskCompilerPage() {
  const { refresh } = useRuntime();
  const [category, setCategory] = useState<TaskType>("coding");
  const [language, setLanguage] = useState("python");
  const [memoryMb, setMemoryMb] = useState(1024);
  const [customPrompt, setCustomPrompt] = useState("");
  const [preview, setPreview] = useState<TaskProfile | null>(null);
  const [step, setStep] = useState(-1);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);

  async function onPreview() {
    const prompt =
      category === "custom" && customPrompt
        ? customPrompt
        : `I need a ${language} ${category} assistant`;
    const profile = await previewTask(prompt);
    setPreview(profile);
  }

  async function onCompile() {
    setBusy(true);
    setErr(null);
    setResult(null);
    setStep(0);
    const timers = STEPS.map((_, i) =>
      window.setTimeout(() => setStep(i), 400 + i * 500),
    );
    try {
      const model = await compileBrain(
        category,
        category === "coding" ? language : null,
        memoryMb,
        true,
      );
      setStep(STEPS.length - 1);
      setResult(`Compiled & loaded: ${model.name}`);
      await refresh();
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      timers.forEach(clearTimeout);
      setBusy(false);
      setTimeout(() => setStep(-1), 800);
    }
  }

  return (
    <div className="mx-auto max-w-3xl space-y-8">
      <header>
        <h1 className="font-display text-2xl font-semibold text-white">Task Compiler</h1>
        <p className="mt-2 text-mist-dim">
          Intent → TaskProfile → mock weight generation. Swap generator later.
        </p>
      </header>

      <section className="panel space-y-5 p-6">
        <div className="label">Choose intelligence</div>
        <div className="flex flex-wrap gap-2">
          {CATEGORIES.map((c) => (
            <button
              key={c.id}
              type="button"
              onClick={() => setCategory(c.id)}
              className={`rounded-md px-4 py-2 text-sm transition ${
                category === c.id
                  ? "bg-signal text-ink-950"
                  : "border border-ink-700 text-mist hover:border-signal/40"
              }`}
            >
              {c.label}
            </button>
          ))}
        </div>

        {category === "coding" && (
          <div>
            <div className="label mb-2">Language</div>
            <select
              className="rounded-md border border-ink-700 bg-ink-950 px-3 py-2 text-sm"
              value={language}
              onChange={(e) => setLanguage(e.target.value)}
            >
              {["python", "rust", "javascript", "typescript"].map((l) => (
                <option key={l} value={l}>
                  {l}
                </option>
              ))}
            </select>
          </div>
        )}

        {category === "custom" && (
          <textarea
            className="h-24 w-full rounded-md border border-ink-700 bg-ink-950 px-3 py-2 text-sm outline-none focus:border-signal/50"
            placeholder="Describe the specialist you need…"
            value={customPrompt}
            onChange={(e) => setCustomPrompt(e.target.value)}
          />
        )}

        <div>
          <div className="label mb-2">Memory limit (MB)</div>
          <input
            type="number"
            min={64}
            className="w-36 rounded-md border border-ink-700 bg-ink-950 px-3 py-2 text-sm"
            value={memoryMb}
            onChange={(e) => setMemoryMb(Number(e.target.value))}
          />
        </div>

        <div className="flex gap-3">
          <button type="button" className="btn-ghost" onClick={() => void onPreview()}>
            Preview profile
          </button>
          <button
            type="button"
            className="btn-primary"
            disabled={busy}
            onClick={() => void onCompile()}
          >
            Compile Brain
          </button>
        </div>
      </section>

      {preview && (
        <pre className="panel overflow-auto p-4 font-mono text-xs text-mist">
          {JSON.stringify(preview, null, 2)}
        </pre>
      )}

      {step >= 0 && (
        <ul className="panel space-y-2 p-5">
          {STEPS.map((s, i) => (
            <li
              key={s}
              className={`flex items-center gap-3 text-sm ${
                i <= step ? "text-signal" : "text-mist-dim"
              }`}
            >
              <span className="font-mono text-xs">{i <= step ? "●" : "○"}</span>
              {s}
            </li>
          ))}
        </ul>
      )}

      {result && <p className="text-signal">{result}</p>}
      {err && <p className="text-amber-soft">{err}</p>}
    </div>
  );
}
