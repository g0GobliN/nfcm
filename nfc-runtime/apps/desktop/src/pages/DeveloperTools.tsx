import { optimizeMemory } from "../lib/api";
import { useRuntime } from "../lib/RuntimeContext";
import { formatBytes, formatMb } from "../lib/types";

export default function DeveloperTools() {
  const { snapshot, refresh } = useRuntime();

  if (!snapshot) {
    return <p className="text-mist-dim">Loading…</p>;
  }

  return (
    <div className="mx-auto max-w-5xl space-y-8">
      <header>
        <h1 className="font-display text-2xl font-semibold text-white">Developer Tools</h1>
        <p className="mt-2 text-mist-dim">Runtime logs, API status, model metadata.</p>
      </header>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <div className="panel p-4">
          <div className="label">API status</div>
          <div className="mt-2 text-signal capitalize">{snapshot.status}</div>
        </div>
        <div className="panel p-4">
          <div className="label">Models registered</div>
          <div className="mt-2 font-mono text-white">{snapshot.models.length}</div>
        </div>
        <div className="panel p-4">
          <div className="label">Weight generator</div>
          <div className="mt-2 truncate text-white">{snapshot.weight_generator.name}</div>
          <div className="mt-1 text-xs text-mist-dim">
            {snapshot.weight_generator.kind} ·{" "}
            {snapshot.weight_generator.is_mock ? "mock" : "proto"}
          </div>
        </div>
        <div className="panel p-4">
          <div className="label">Inference backend</div>
          <div className="mt-2 truncate text-white">{snapshot.inference_backend.name}</div>
          <div className="mt-1 text-xs text-mist-dim">
            {snapshot.inference_backend.is_mock ? "mock" : "pluggable"} ·{" "}
            {snapshot.inference_backend.ready ? "ready" : "idle"}
          </div>
        </div>
      </div>

      <section className="panel p-5">
        <div className="label mb-3">Weight generator</div>
        <pre className="overflow-auto font-mono text-xs text-mist">
          {JSON.stringify(snapshot.weight_generator, null, 2)}
        </pre>
      </section>

      <section className="panel p-5">
        <div className="label mb-3">Backend capabilities</div>
        <pre className="overflow-auto font-mono text-xs text-mist">
          {JSON.stringify(snapshot.inference_backend, null, 2)}
        </pre>
      </section>

      <section className="panel p-5">
        <div className="mb-3 flex items-center justify-between">
          <div className="label">Memory breakdown</div>
          <button
            type="button"
            className="btn-ghost text-xs"
            onClick={() => void optimizeMemory().then(refresh)}
          >
            Optimize
          </button>
        </div>
        <pre className="font-mono text-xs text-mist">
          {JSON.stringify(
            {
              max: formatMb(snapshot.memory.max_ram_bytes),
              used: formatMb(snapshot.memory.used_bytes),
              generator: formatBytes(snapshot.memory.generator_bytes),
              active_model: formatBytes(snapshot.memory.active_model_bytes),
              cache: formatBytes(snapshot.memory.cache_bytes),
            },
            null,
            2,
          )}
        </pre>
      </section>

      <section className="panel p-5">
        <div className="label mb-3">Active model</div>
        <pre className="overflow-auto font-mono text-xs text-mist">
          {JSON.stringify(snapshot.active_model, null, 2)}
        </pre>
      </section>

      <section className="panel p-5">
        <div className="label mb-3">Runtime logs</div>
        <div className="max-h-80 overflow-auto font-mono text-xs space-y-1">
          {snapshot.logs.length === 0 && <p className="text-mist-dim">No logs yet.</p>}
          {snapshot.logs.map((line, i) => (
            <div key={i} className="text-mist">
              {line}
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
