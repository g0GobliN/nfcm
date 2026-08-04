import { Link } from "react-router-dom";
import { useRuntime } from "../lib/RuntimeContext";
import { formatMb } from "../lib/types";

export default function Dashboard() {
  const { snapshot } = useRuntime();
  if (!snapshot) {
    return <p className="text-mist-dim">Loading runtime…</p>;
  }

  const { hardware, memory, active_model, status, inference_backend, weight_generator } =
    snapshot;
  const pct = Math.min(100, Math.round((memory.used_bytes / memory.max_ram_bytes) * 100));
  const gpu = hardware.gpus[0];

  return (
    <div className="page-enter mx-auto max-w-5xl space-y-8">
      <header className="relative overflow-hidden rounded-2xl border border-ink-700/70 bg-ink-900/50 p-8 shadow-panel backdrop-blur-md">
        <div className="pointer-events-none absolute -right-16 -top-20 h-56 w-56 rounded-full bg-signal/15 blur-3xl" />
        <div className="pointer-events-none absolute -bottom-24 left-1/3 h-40 w-40 rounded-full bg-amber-soft/10 blur-3xl" />

        <div className="relative flex flex-wrap items-end justify-between gap-6">
          <div className="flex items-center gap-5">
            <img
              src="/logo.png"
              alt="NFCM"
              className="h-16 w-16 rounded-2xl shadow-glow ring-1 ring-signal/25"
            />
            <div>
              <p className="label mb-2">Neural Foundation</p>
              <h1 className="font-display text-3xl font-semibold tracking-tight text-white md:text-4xl">
                NFCM
              </h1>
              <p className="mt-2 max-w-md text-sm text-mist-dim">
                Local-first runtime — compile specialists, run GGUF brains, stay on-device.
              </p>
            </div>
          </div>
          <div className="flex flex-wrap gap-2">
            <Link to="/chat" className="btn-primary">
              Open Chat
            </Link>
            <Link to="/models" className="btn-ghost">
              Models
            </Link>
          </div>
        </div>
      </header>

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <Stat label="Status" value={status} accent />
        <Stat label="CPU" value={`${hardware.cpu.cores} cores`} sub={hardware.cpu.brand} />
        <Stat
          label="RAM"
          value={formatMb(hardware.ram.available_bytes) + " free"}
          sub={`${formatMb(hardware.ram.total_bytes)} total`}
        />
        <Stat label="GPU" value={gpu?.available ? gpu.vendor : "CPU mode"} sub={gpu?.name} />
      </div>

      <section className="panel relative overflow-hidden p-7">
        <div className="pointer-events-none absolute right-0 top-0 h-28 w-28 bg-signal/10 blur-2xl" />
        <div className="label">Active brain</div>
        <div className="mt-3 font-display text-2xl font-semibold tracking-tight text-white">
          {active_model?.name ?? "None loaded"}
        </div>
        {active_model ? (
          <p className="mt-2 text-sm text-mist-dim">
            {active_model.task_type} · {active_model.skills.join(", ") || "no skills"} ·{" "}
            {active_model.architecture}
          </p>
        ) : (
          <p className="mt-2 text-sm text-mist-dim">
            Import a GGUF in Settings, then Load it from Models — or compile a latent brain.
          </p>
        )}
        <div className="mt-5 flex flex-wrap gap-x-6 gap-y-2 text-sm">
          <Meta k="Generator" v={weight_generator.name} />
          <Meta
            k="Backend"
            v={`${inference_backend.name}${inference_backend.ready ? " · ready" : " · idle"}`}
          />
        </div>
      </section>

      <section className="panel p-7">
        <div className="flex items-end justify-between">
          <div>
            <div className="label">Memory budget</div>
            <div className="mt-2 font-mono text-xl text-white">
              {formatMb(memory.used_bytes)}{" "}
              <span className="text-mist-dim">/ {formatMb(memory.max_ram_bytes)}</span>
            </div>
          </div>
          <div className="font-mono text-sm text-signal">{pct}%</div>
        </div>
        <div className="mt-5 h-2.5 overflow-hidden rounded-full bg-ink-800">
          <div
            className="h-full rounded-full bg-gradient-to-r from-signal-muted via-signal to-signal transition-all duration-700"
            style={{ width: `${pct}%` }}
          />
        </div>
        <div className="mt-5 grid grid-cols-3 gap-3 text-sm">
          <MemPart label="Generator" bytes={memory.generator_bytes} />
          <MemPart label="Active model" bytes={memory.active_model_bytes} />
          <MemPart label="Cache" bytes={memory.cache_bytes} />
        </div>
      </section>
    </div>
  );
}

function Meta({ k, v }: { k: string; v: string }) {
  return (
    <div>
      <span className="text-mist-dim">{k} · </span>
      <span className="text-mist">{v}</span>
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
    <div className="panel p-4 transition duration-200 hover:border-signal/25">
      <div className="label">{label}</div>
      <div
        className={`mt-2 truncate font-display text-lg font-semibold capitalize ${
          accent ? "text-signal" : "text-white"
        }`}
      >
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
