import { useState } from "react";
import { deleteModel, importModel, loadModel, unloadModel } from "../lib/api";
import { useRuntime } from "../lib/RuntimeContext";
import { formatMb, type TaskType } from "../lib/types";

export default function ModelManager() {
  const { snapshot, refresh } = useRuntime();
  const [name, setName] = useState("");
  const [taskType, setTaskType] = useState<TaskType>("coding");
  const [memoryMb, setMemoryMb] = useState(64);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const models = snapshot?.models ?? [];
  const activeId = snapshot?.active_model?.id;

  async function onImport() {
    if (!name.trim()) return;
    setBusy(true);
    setErr(null);
    try {
      await importModel(name.trim(), taskType, memoryMb);
      setName("");
      await refresh();
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  async function onDelete(id: string) {
    setBusy(true);
    try {
      await deleteModel(id);
      await refresh();
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  async function onLoad(id: string) {
    setBusy(true);
    try {
      await loadModel(id);
      await refresh();
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="page-enter mx-auto max-w-5xl space-y-8">
      <header>
        <p className="label mb-1">Registry</p>
        <h1 className="font-display text-2xl font-semibold tracking-tight text-white">Models</h1>
        <p className="mt-2 text-mist-dim">
          Installed and generated brains. Load one before chatting.
        </p>
      </header>

      <section className="panel space-y-4 p-6">
        <div className="label">Import model stub</div>
        <div className="flex flex-wrap gap-3">
          <input
            className="min-w-[200px] flex-1 rounded-md border border-ink-700 bg-ink-950 px-3 py-2 text-sm outline-none focus:border-signal/50"
            placeholder="Name"
            value={name}
            onChange={(e) => setName(e.target.value)}
          />
          <select
            className="rounded-md border border-ink-700 bg-ink-950 px-3 py-2 text-sm"
            value={taskType}
            onChange={(e) => setTaskType(e.target.value as TaskType)}
          >
            {(["coding", "math", "writing", "research", "medical", "custom"] as TaskType[]).map(
              (t) => (
                <option key={t} value={t}>
                  {t}
                </option>
              ),
            )}
          </select>
          <input
            type="number"
            min={16}
            className="w-28 rounded-md border border-ink-700 bg-ink-950 px-3 py-2 text-sm"
            value={memoryMb}
            onChange={(e) => setMemoryMb(Number(e.target.value))}
          />
          <button type="button" className="btn-primary" disabled={busy} onClick={() => void onImport()}>
            Import
          </button>
        </div>
        {err && <p className="text-sm text-amber-soft">{err}</p>}
      </section>

      <section className="space-y-3">
        {models.length === 0 && (
          <p className="text-mist-dim">No models yet — compile a brain or import a stub.</p>
        )}
        {models.map((m) => (
          <article key={m.id} className="panel flex flex-wrap items-start justify-between gap-4 p-5">
            <div>
              <div className="font-medium text-white">
                {m.name}
                {activeId === m.id && (
                  <span className="ml-2 text-xs text-signal">loaded</span>
                )}
              </div>
              <div className="mt-1 text-xs text-mist-dim">
                {m.task_type} · {m.architecture} · {formatMb(m.memory_requirement_bytes)} ·{" "}
                {m.status}
              </div>
              <p className="mt-2 max-w-xl text-sm text-mist-dim">{m.description}</p>
              {m.skills.length > 0 && (
                <p className="mt-1 text-xs text-mist-dim">Skills: {m.skills.join(", ")}</p>
              )}
            </div>
            <div className="flex gap-2">
              {activeId === m.id ? (
                <button
                  type="button"
                  className="btn-ghost"
                  disabled={busy}
                  onClick={() => void unloadModel().then(refresh)}
                >
                  Unload
                </button>
              ) : (
                <button
                  type="button"
                  className="btn-ghost"
                  disabled={busy}
                  onClick={() => void onLoad(m.id)}
                >
                  Load
                </button>
              )}
              <button
                type="button"
                className="btn-ghost"
                disabled={busy}
                onClick={() => void onDelete(m.id)}
              >
                Delete
              </button>
            </div>
          </article>
        ))}
      </section>
    </div>
  );
}
