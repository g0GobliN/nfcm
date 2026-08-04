export type TaskType =
  | "coding"
  | "math"
  | "writing"
  | "research"
  | "medical"
  | "custom";

export type ModelStatus =
  | "registered"
  | "generating"
  | "ready"
  | "loaded"
  | "unloaded"
  | "failed";

export interface Model {
  id: string;
  name: string;
  size_bytes: number;
  architecture: string;
  task_type: TaskType;
  memory_requirement_bytes: number;
  status: ModelStatus;
  skills: string[];
  description: string;
  path?: string | null;
  created_at: string;
  updated_at: string;
}

export interface HardwareProfile {
  cpu: { brand: string; cores: number; frequency_mhz: number };
  ram: { total_bytes: number; available_bytes: number; used_bytes: number };
  gpus: { name: string; vendor: string; vram_bytes: number; available: boolean }[];
  hostname: string;
}

export interface MemorySnapshot {
  max_ram_bytes: number;
  used_bytes: number;
  generator_bytes: number;
  active_model_bytes: number;
  cache_bytes: number;
  other_bytes: number;
}

export interface BackendInfo {
  id: string;
  name: string;
  is_mock: boolean;
  ready: boolean;
  capabilities: {
    always_available: boolean;
    loads_external_weights: boolean;
    notes: string;
  };
}

export interface GeneratorInfo {
  kind: string;
  name: string;
  is_mock: boolean;
  notes: string;
}

export interface RuntimeSnapshot {
  status: string;
  hardware: HardwareProfile;
  memory: MemorySnapshot;
  active_model: Model | null;
  models: Model[];
  logs: string[];
  console_lines: string[];
  inference_backend: BackendInfo;
  weight_generator: GeneratorInfo;
}

export interface TaskProfile {
  domain: TaskType;
  skills: string[];
  language?: string | null;
  memory_limit_bytes: number;
  raw_prompt: string;
}

export interface InferenceResponse {
  model_id: string;
  model_name: string;
  text: string;
  tokens_in: number;
  tokens_out: number;
  is_mock: boolean;
  backend: string;
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const mb = bytes / (1024 * 1024);
  if (mb < 1024) return `${mb.toFixed(0)} MB`;
  return `${(mb / 1024).toFixed(2)} GB`;
}

export function formatMb(bytes: number): string {
  return `${Math.round(bytes / (1024 * 1024))} MB`;
}
