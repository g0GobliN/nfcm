import { invoke } from "@tauri-apps/api/core";
import type {
  InferenceResponse,
  Model,
  RuntimeSnapshot,
  TaskProfile,
  TaskType,
} from "./types";

const isTauri = () =>
  typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;

/** Browser-dev fallback so Vite UI can be previewed without the Rust shell. */
function mockSnapshot(): RuntimeSnapshot {
  return {
    status: "running",
    hardware: {
      cpu: { brand: "Dev Preview CPU", cores: 8, frequency_mhz: 3200 },
      ram: {
        total_bytes: 16 * 1024 ** 3,
        available_bytes: 8 * 1024 ** 3,
        used_bytes: 8 * 1024 ** 3,
      },
      gpus: [
        {
          name: "CPU fallback",
          vendor: "none",
          vram_bytes: 0,
          available: false,
        },
      ],
      hostname: "dev-preview",
    },
    memory: {
      max_ram_bytes: 1024 * 1024 * 1024,
      used_bytes: 820 * 1024 * 1024,
      generator_bytes: 300 * 1024 * 1024,
      active_model_bytes: 420 * 1024 * 1024,
      cache_bytes: 100 * 1024 * 1024,
      other_bytes: 0,
    },
    active_model: {
      id: "00000000-0000-0000-0000-000000000001",
      name: "Coding Specialist",
      size_bytes: 420 * 1024 * 1024,
      architecture: "mock",
      task_type: "coding",
      memory_requirement_bytes: 420 * 1024 * 1024,
      status: "loaded",
      skills: ["python", "debugging"],
      description: "Preview stub — launch via Tauri for real runtime",
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    },
    models: [],
    logs: ["Preview mode: start with `npm run tauri dev` for live runtime"],
    console_lines: ["> runtime preview", "Ready."],
  };
}

async function call<T>(cmd: string, args?: Record<string, unknown>): Promise<T> {
  if (!isTauri()) {
    throw new Error("tauri_unavailable");
  }
  return invoke<T>(cmd, args);
}

export async function getSnapshot(): Promise<RuntimeSnapshot> {
  try {
    return await call<RuntimeSnapshot>("get_snapshot");
  } catch {
    return mockSnapshot();
  }
}

export async function listModels(): Promise<Model[]> {
  try {
    return await call<Model[]>("list_models");
  } catch {
    return (await getSnapshot()).models;
  }
}

export async function deleteModel(id: string): Promise<void> {
  await call("delete_model", { id });
}

export async function importModel(
  name: string,
  taskType: TaskType,
  memoryMb: number,
): Promise<Model> {
  return call("import_model", {
    name,
    taskType,
    memoryMb,
  });
}

export async function compileBrain(
  category: TaskType,
  language: string | null,
  memoryLimitMb: number,
  load: boolean,
): Promise<Model> {
  return call("compile_brain", {
    category,
    language,
    memoryLimitMb,
    load,
  });
}

export async function compileFromPrompt(prompt: string, load: boolean): Promise<Model> {
  return call("compile_from_prompt", { prompt, load });
}

export async function loadModel(id: string): Promise<Model> {
  return call("load_model", { id });
}

export async function unloadModel(): Promise<void> {
  await call("unload_model");
}

export async function runInference(
  prompt: string,
  maxTokens: number,
): Promise<InferenceResponse> {
  return call("run_inference", { prompt, maxTokens });
}

export async function optimizeMemory(): Promise<number> {
  return call("optimize_memory");
}

export async function consoleCommand(line: string): Promise<string[]> {
  try {
    return await call<string[]>("console_command", { line });
  } catch {
    return [`> ${line}`, "Preview mode — use Tauri for live console"];
  }
}

export async function previewTask(prompt: string): Promise<TaskProfile> {
  try {
    return await call("preview_task", { prompt });
  } catch {
    return {
      domain: "coding",
      skills: ["python", "debugging"],
      language: "python",
      memory_limit_bytes: 1024 * 1024 * 1024,
      raw_prompt: prompt,
    };
  }
}
