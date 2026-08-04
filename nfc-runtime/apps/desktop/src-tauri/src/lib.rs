//! Tauri command surface over `nfc-runtime`.

use nfc_generator::{TaskCategory, TaskProfile};
use nfc_runtime::{
    InferenceRequest, Model, RuntimeConfig, RuntimeEngine, RuntimeSnapshot, TaskType,
};
use std::path::PathBuf;
use std::sync::Arc;
use tauri::State;
use uuid::Uuid;

pub struct AppState {
    pub engine: Arc<RuntimeEngine>,
}

fn data_dir() -> PathBuf {
    dirs_fallback()
}

fn dirs_fallback() -> PathBuf {
    if let Some(home) = std::env::var_os("HOME") {
        return PathBuf::from(home).join(".local/share/nfcm");
    }
    PathBuf::from("./nfcm-data")
}

fn parse_category(s: &str) -> TaskCategory {
    match s.to_lowercase().as_str() {
        "coding" => TaskCategory::Coding,
        "math" => TaskCategory::Math,
        "writing" => TaskCategory::Writing,
        "research" => TaskCategory::Research,
        "medical" => TaskCategory::Medical,
        _ => TaskCategory::Custom,
    }
}

fn parse_task_type(s: &str) -> TaskType {
    match s.to_lowercase().as_str() {
        "coding" => TaskType::Coding,
        "math" => TaskType::Math,
        "writing" => TaskType::Writing,
        "research" => TaskType::Research,
        "medical" => TaskType::Medical,
        _ => TaskType::Custom,
    }
}

#[tauri::command]
fn get_snapshot(state: State<'_, AppState>) -> Result<RuntimeSnapshot, String> {
    state.engine.snapshot().map_err(|e| e.to_string())
}

#[tauri::command]
fn list_models(state: State<'_, AppState>) -> Result<Vec<Model>, String> {
    state.engine.list_models().map_err(|e| e.to_string())
}

#[tauri::command]
fn delete_model(state: State<'_, AppState>, id: String) -> Result<(), String> {
    let id = Uuid::parse_str(&id).map_err(|e| e.to_string())?;
    state.engine.delete_model(id).map_err(|e| e.to_string())
}

#[tauri::command]
fn import_model(
    state: State<'_, AppState>,
    name: String,
    task_type: String,
    memory_mb: u64,
) -> Result<Model, String> {
    state
        .engine
        .import_model_stub(name, parse_task_type(&task_type), memory_mb)
        .map_err(|e| e.to_string())
}

#[tauri::command]
fn compile_brain(
    state: State<'_, AppState>,
    category: String,
    language: Option<String>,
    memory_limit_mb: u64,
    load: bool,
) -> Result<Model, String> {
    let profile =
        state
            .engine
            .compile_category(parse_category(&category), language, Some(memory_limit_mb));
    state
        .engine
        .compile_brain(profile, load, |_| {})
        .map_err(|e| e.to_string())
}

#[tauri::command]
fn compile_from_prompt(
    state: State<'_, AppState>,
    prompt: String,
    load: bool,
) -> Result<Model, String> {
    let profile = state.engine.compile_prompt(&prompt);
    state
        .engine
        .compile_brain(profile, load, |_| {})
        .map_err(|e| e.to_string())
}

#[tauri::command]
fn preview_task(state: State<'_, AppState>, prompt: String) -> TaskProfile {
    state.engine.compile_prompt(&prompt)
}

#[tauri::command]
fn load_model(state: State<'_, AppState>, id: String) -> Result<Model, String> {
    let id = Uuid::parse_str(&id).map_err(|e| e.to_string())?;
    state.engine.load_model(id).map_err(|e| e.to_string())
}

#[tauri::command]
fn unload_model(state: State<'_, AppState>) -> Result<(), String> {
    state.engine.unload_model().map_err(|e| e.to_string())
}

#[tauri::command]
fn run_inference(
    state: State<'_, AppState>,
    prompt: String,
    max_tokens: u32,
) -> Result<nfc_runtime::InferenceResponse, String> {
    state
        .engine
        .run_inference(InferenceRequest { prompt, max_tokens })
        .map_err(|e| e.to_string())
}

#[tauri::command]
fn optimize_memory(state: State<'_, AppState>) -> Result<u64, String> {
    state.engine.optimize_memory().map_err(|e| e.to_string())
}

#[tauri::command]
fn console_command(state: State<'_, AppState>, line: String) -> Result<Vec<String>, String> {
    let trimmed = line.trim();
    let lower = trimmed.to_lowercase();

    if lower == "help" {
        state.engine.append_console("> help");
        state.engine.append_console(
            "Commands: compile <prompt> | status | unload | optimize | help | clear",
        );
    } else if lower == "status" {
        let snap = state.engine.snapshot().map_err(|e| e.to_string())?;
        state.engine.append_console("> status");
        state
            .engine
            .append_console(format!("status: {:?}", snap.status));
        if let Some(m) = snap.active_model {
            state.engine.append_console(format!(
                "brain: {} ({} MB)",
                m.name,
                m.memory_requirement_mb()
            ));
        } else {
            state.engine.append_console("brain: none");
        }
        state.engine.append_console(format!(
            "memory: {} / {} MB",
            snap.memory.used_mb(),
            snap.memory.max_mb()
        ));
    } else if lower == "unload" {
        state.engine.append_console("> unload");
        state.engine.unload_model().map_err(|e| e.to_string())?;
    } else if lower == "optimize" {
        state.engine.append_console("> optimize");
        let freed = state.engine.optimize_memory().map_err(|e| e.to_string())?;
        state.engine.append_console(format!("freed {freed} bytes"));
    } else if lower == "clear" {
        // Soft clear: just note it; snapshot still returns full history.
        state.engine.append_console("> clear");
        state
            .engine
            .append_console("(history retained in snapshot; UI may filter)");
    } else if let Some(rest) = lower.strip_prefix("compile ") {
        let prompt = trimmed[trimmed.len() - rest.len()..].to_string();
        // Use original casing for prompt body after "compile "
        let prompt = trimmed
            .strip_prefix("compile ")
            .or_else(|| trimmed.strip_prefix("Compile "))
            .unwrap_or(prompt.as_str())
            .to_string();
        let profile = state.engine.compile_prompt(&prompt);
        state
            .engine
            .compile_brain(profile, true, |_| {})
            .map_err(|e| e.to_string())?;
    } else if !trimmed.is_empty() {
        state.engine.append_console(format!("> {trimmed}"));
        state.engine.append_console("Unknown command. Type `help`.");
    }

    let snap = state.engine.snapshot().map_err(|e| e.to_string())?;
    Ok(snap.console_lines)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tracing_subscriber::fmt()
        .with_env_filter("nfc_runtime=info,nfc_desktop=info")
        .init();

    let engine =
        RuntimeEngine::start(RuntimeConfig::new(data_dir())).expect("failed to start NFCM runtime");

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(AppState {
            engine: Arc::new(engine),
        })
        .invoke_handler(tauri::generate_handler![
            get_snapshot,
            list_models,
            delete_model,
            import_model,
            compile_brain,
            compile_from_prompt,
            preview_task,
            load_model,
            unload_model,
            run_inference,
            optimize_memory,
            console_command,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
