import { useRuntime } from "../lib/RuntimeContext";
import { formatMb } from "../lib/types";

export default function Dashboard() {
  const { snapshot } = useRuntime();
  if (!snapshot) {
    return <p className="text-mist-dim">Loading runtime…</p>;
  }

  const { hardware, memory, active_model, status, inference_backend } = snapshot;
  const pct = Math.min(100, Math.round((memory.used_bytes / memory.max_ram_bytes) * 100));
  const gpu = hardware.gpus[0];

  return (
    <div className="mx-auto max-w-5xl space-y-8">
      <header className="flex items-center gap-4">
        <img
          src="/logo.png"
          alt="NFCM"
          className="h-14 w-14 rounded-xl shadow-panel"
        />
        <div>
          <h1 className="font-display text-3xl font-bold text-white">NFCM Runtime</h1>
          <p className="mt-1 text-mist-dim">Local-first workstation · Phase 1 foundation</p>
        </div>
      </header>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Stat label="Status" value={status} accent />
        <Stat label="CPU" value={`${hardware.cpu.cores} cores`} sub={hardware.cpu.brand} />
        <Stat
          label="RAM"
          value={formatMb(hardware.ram.available_bytes) + " free"}
          sub={`${formatMb(hardware.ram.total_bytes)} total`}
        />
        <Stat
          label="GPU"
          value={gpu?.available ? gpu.vendor : "CPU mode"}
          sub={gpu?.name}
        />
      </div>

      <section className="panel p-6">
        <div className="label">Current Brain</div>
        <div className="mt-2 font-display text-2xl font-semibold text-white">
          {active_model?.name ?? "None loaded"}
        </div>
        {active_model && (
          <p className="mt-2 text-sm text-mist-dim">
            {active_model.task_type} · {active_model.skills.join(", ") || "no skills"} ·{" "}
            {active_model.architecture}
          </p>
        )}
        <p className="mt-3 text-sm text-mist-dim">
          Backend:{" "}
          <span className="text-mist">{inference_backend.name}</span>
          {inference_backend.is_mock ? " · mock" : ""}
          {inference_backend.ready ? " · ready" : " · idle"}
        </p>
      </section>

      <section className="panel p-6">
        <div className="flex items-end justify-between">
          <div>
            <div className="label">Memory</div>
            <div className="mt-2 font-mono text-xl text-white">
              {formatMb(memory.used_bytes)} / {formatMb(memory.max_ram_bytes)}
            </div>
          </div>
          <div className="text-sm text-mist-dim">{pct}%</div>
        </div>
        <div className="mt-4 h-2 overflow-hidden rounded-full bg-ink-800">
          <div
            className="h-full rounded-full bg-signal transition-all duration-500"
            style={{ width: `${pct}%` }}
          />
        </div>
        <div className="mt-4 grid grid-cols-3 gap-3 text-sm">
          <MemPart label="Generator" bytes={memory.generator_bytes} />
          <MemPart label="Active model" bytes={memory.active_model_bytes} />
          <MemPart label="Cache" bytes={memory.cache_bytes} />
        </div>
      </section>
    </div>
  );
}

function Stat({
  label,
  value,
  sub,
  accent,
}: {
  label: string;
  value: string;
  sub?: string;
  accent?: boolean;
}) {
  return (
    <div className="panel p-4">
      <div className="label">{label}</div>
      <div className={`mt-2 truncate font-medium capitalize ${accent ? "text-signal" : "text-white"}`}>
        {value}
      </div>
      {sub && <div className="mt-1 truncate text-xs text-mist-dim">{sub}</div>}
    </div>
  );
}

function MemPart({ label, bytes }: { label: string; bytes: number }) {
  return (
    <div>
      <div className="text-mist-dim">{label}</div>
      <div className="font-mono text-white">{formatMb(bytes)}</div>
    </div>
  );
}
