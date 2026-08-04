import { useState } from "react";
import { importGgufModel, setBackend } from "../lib/api";
import { useRuntime } from "../lib/RuntimeContext";

const BACKENDS = ["mock", "latent", "gguf", "candle"] as const;

export default function SettingsPage() {
  const { snapshot, refresh } = useRuntime();
  const [backend, setBackendLocal] = useState(
    snapshot?.inference_backend.id ?? "latent",
  );
  const [ggufPath, setGgufPath] = useState("");
  const [ggufName, setGgufName] = useState("");
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const current = snapshot?.inference_backend;

  async function onSetBackend() {
    setBusy(true);
    setErr(null);
    setMsg(null);
    try {
      const info = await setBackend(backend);
      setMsg(`Backend → ${info.name} (ready=${info.ready})`);
      await refresh();
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  async function onImportGguf() {
    if (!ggufPath.trim()) return;
    setBusy(true);
    setErr(null);
    setMsg(null);
    try {
      const model = await importGgufModel(
        ggufPath.trim(),
        ggufName.trim() || null,
        512,
      );
      setMsg(`Imported ${model.name}. Switch backend to gguf, then Load it.`);
      await refresh();
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="page-enter mx-auto max-w-3xl space-y-8">
      <header>
        <p className="label mb-1">Runtime</p>
        <h1 className="font-display text-4xl font-extrabold tracking-tight text-white">Settings</h1>
        <p className="mt-2 text-mist-dim">
          Inference backend and GGUF import. GGUF needs{" "}
          <code className="text-signal">llama-completion</code> (or{" "}
          <code className="text-signal">NFCM_LLAMA_CLI</code>).
        </p>
      </header>

      <section className="panel space-y-4 p-6">
        <div className="label">Inference backend</div>
        <p className="text-sm text-mist-dim">
          Current: {current?.name ?? "—"}
          {current?.is_mock ? " · mock" : ""}
          {current?.ready ? " · ready" : " · idle"}
        </p>
        <div className="flex flex-wrap gap-3">
          <select
            className="rounded-md border border-ink-700 bg-ink-950 px-3 py-2 text-sm"
            value={backend}
            onChange={(e) => setBackendLocal(e.target.value)}
          >
            {BACKENDS.map((b) => (
              <option key={b} value={b}>
                {b}
              </option>
            ))}
          </select>
          <button
            type="button"
            className="btn-primary"
            disabled={busy}
            onClick={() => void onSetBackend()}
          >
            Apply
          </button>
        </div>
        <p className="text-xs text-mist-dim">
          Switching backends unloads the active brain. Latent needs a compiled NFCM brain;
          GGUF needs an imported <code>.gguf</code> + CLI.
        </p>
      </section>

      <section className="panel space-y-4 p-6">
        <div className="label">Import GGUF</div>
        <input
          className="w-full rounded-md border border-ink-700 bg-ink-950 px-3 py-2 text-sm outline-none focus:border-signal/50"
          placeholder="/path/to/model.gguf (or use models/tinyllama… already downloaded)"
          value={ggufPath}
          onChange={(e) => setGgufPath(e.target.value)}
        />
        <input
          className="w-full rounded-md border border-ink-700 bg-ink-950 px-3 py-2 text-sm outline-none focus:border-signal/50"
          placeholder="Display name (optional)"
          value={ggufName}
          onChange={(e) => setGgufName(e.target.value)}
        />
        <button
          type="button"
          className="btn-primary"
          disabled={busy || !ggufPath.trim()}
          onClick={() => void onImportGguf()}
        >
          Import GGUF
        </button>
        <p className="text-xs text-mist-dim">
          Optional env: <code>NFCM_GGUF_MODEL</code>, <code>NFCM_LLAMA_CLI</code>
        </p>
      </section>

      {msg && <p className="text-sm text-signal">{msg}</p>}
      {err && <p className="text-sm text-amber-soft">{err}</p>}
    </div>
  );
}
