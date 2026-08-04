//! GGUF / llama.cpp compatibility backend.
//!
//! Shells out to an external `llama-cli` (or `llama-cpp`) binary when configured.
//! No binary / no model path → clear `NotConfigured` errors (not silent fake output).

use super::{
    BackendCapabilities, BackendId, InferenceBackend, InferenceError, InferenceRequest,
    InferenceResponse, ModelContext,
};
use std::path::{Path, PathBuf};
use std::process::Command;
use uuid::Uuid;

#[derive(Debug, Clone)]
pub struct GgufConfig {
    pub cli_path: PathBuf,
    pub model_path: Option<PathBuf>,
}

impl GgufConfig {
    pub fn from_env() -> Self {
        let cli = std::env::var("NFCM_LLAMA_CLI")
            .map(PathBuf::from)
            .unwrap_or_else(|_| default_llama_cli());
        let model = std::env::var("NFCM_GGUF_MODEL")
            .ok()
            .filter(|s| !s.is_empty())
            .map(PathBuf::from)
            .filter(|p| p.is_file())
            .or_else(default_gguf_model);
        Self {
            cli_path: cli,
            model_path: model,
        }
    }
}

fn default_llama_cli() -> PathBuf {
    // Prefer project-bundled tools/ when present.
    let candidates = [
        PathBuf::from("tools/llama-b10250/llama-completion"),
        PathBuf::from("nfc-runtime/tools/llama-b10250/llama-completion"),
        PathBuf::from("llama-completion"),
        PathBuf::from("llama-cli"),
    ];
    if let Ok(cwd) = std::env::current_dir() {
        for rel in &candidates {
            let p = cwd.join(rel);
            if p.is_file() {
                return p;
            }
            // walk up a few levels
            let mut dir = cwd.clone();
            for _ in 0..5 {
                let p = dir.join(rel);
                if p.is_file() {
                    return p;
                }
                if !dir.pop() {
                    break;
                }
            }
        }
    }
    PathBuf::from("llama-cli")
}

fn default_gguf_model() -> Option<PathBuf> {
    let names = [
        "models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
        "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    ];
    if let Ok(cwd) = std::env::current_dir() {
        let mut dir = cwd;
        for _ in 0..6 {
            for name in &names {
                let p = dir.join(name);
                if p.is_file() {
                    return Some(p);
                }
                let p2 = dir.join("nfc-runtime").join(name);
                if p2.is_file() {
                    return Some(p2);
                }
            }
            if !dir.pop() {
                break;
            }
        }
    }
    None
}

pub struct GgufInferenceBackend {
    config: GgufConfig,
    attached: Option<(Uuid, String)>,
    effective_model: Option<PathBuf>,
}

impl GgufInferenceBackend {
    pub fn new(config: GgufConfig) -> Self {
        Self {
            config,
            attached: None,
            effective_model: None,
        }
    }

    pub fn from_env() -> Self {
        Self::new(GgufConfig::from_env())
    }

    fn resolve_model(&self, ctx: &ModelContext) -> Result<PathBuf, InferenceError> {
        // Prefer the loaded registry model's path (per-brain), then env default.
        if let Some(path) = &ctx.weights_path {
            let p = PathBuf::from(path);
            if p.extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
                && p.is_file()
            {
                return Ok(p);
            }
        }
        if let Some(p) = &self.config.model_path {
            if p.is_file() {
                return Ok(p.clone());
            }
            // Stale env — don't hard-fail if we might still recover (already tried ctx).
            return Err(InferenceError::NotConfigured(format!(
                "GGUF not found. Loaded model path missing/invalid, and NFCM_GGUF_MODEL \
                 does not exist: {}. Import/Load Qwen: \
                 nfc-runtime/models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
                p.display()
            )));
        }
        Err(InferenceError::NotConfigured(
            "set NFCM_GGUF_MODEL to a .gguf file, or Load a Model with a .gguf path \
             (e.g. models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf)"
                .into(),
        ))
    }

    fn cli_available(cli: &Path) -> bool {
        Command::new(cli)
            .arg("--version")
            .output()
            .map(|o| o.status.success() || !o.stdout.is_empty() || !o.stderr.is_empty())
            .unwrap_or(false)
            || which_exists(cli)
    }
}

fn which_exists(cli: &Path) -> bool {
    if cli.is_absolute() {
        return cli.exists();
    }
    let Some(path) = std::env::var_os("PATH") else {
        return false;
    };
    for dir in std::env::split_paths(&path) {
        let candidate = dir.join(cli);
        if candidate.exists() {
            return true;
        }
    }
    // also try llama-cpp common names
    false
}

impl InferenceBackend for GgufInferenceBackend {
    fn id(&self) -> BackendId {
        BackendId::Gguf
    }

    fn name(&self) -> &str {
        "gguf-llamacpp-v1"
    }

    fn is_mock(&self) -> bool {
        false
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            always_available: false,
            loads_external_weights: true,
            notes: "Requires llama.cpp CLI on PATH (or NFCM_LLAMA_CLI) and a .gguf model \
                    (NFCM_GGUF_MODEL). See docs/inference-backends.md."
                .into(),
        }
    }

    fn attach(&mut self, ctx: &ModelContext) -> Result<(), InferenceError> {
        if !Self::cli_available(&self.config.cli_path) {
            return Err(InferenceError::NotConfigured(format!(
                "llama.cpp CLI not found at '{}' — install llama-cli or set NFCM_LLAMA_CLI",
                self.config.cli_path.display()
            )));
        }
        let model = self.resolve_model(ctx)?;
        self.effective_model = Some(model);
        self.attached = Some((ctx.model_id, ctx.model_name.clone()));
        Ok(())
    }

    fn detach(&mut self) -> Result<(), InferenceError> {
        self.attached = None;
        self.effective_model = None;
        Ok(())
    }

    fn is_ready(&self) -> bool {
        self.attached.is_some() && self.effective_model.is_some()
    }

    fn infer(&self, request: &InferenceRequest) -> Result<InferenceResponse, InferenceError> {
        let (model_id, model_name) = self.attached.clone().ok_or(InferenceError::NotAttached)?;
        let model_path = self
            .effective_model
            .as_ref()
            .ok_or_else(|| InferenceError::NotReady("no gguf model attached".into()))?;

        let n = request.max_tokens.clamp(1, 512);
        let system = std::env::var("NFCM_SYSTEM_PROMPT").unwrap_or_else(|_| {
            "You are a helpful local assistant. Answer clearly and briefly. \
             Do not invent long unrelated examples unless asked."
                .into()
        });
        let mut cmd = Command::new(&self.config.cli_path);
        if let Some(dir) = self.config.cli_path.parent() {
            cmd.current_dir(dir);
            let mut path = dir.display().to_string();
            if let Ok(existing) = std::env::var("LD_LIBRARY_PATH") {
                path = format!("{path}:{existing}");
            }
            cmd.env("LD_LIBRARY_PATH", path);
        }
        // Use model chat template (single turn) instead of raw prompt stuffing.
        let output = cmd
            .arg("-m")
            .arg(model_path)
            .arg("-sys")
            .arg(&system)
            .arg("-p")
            .arg(request.prompt.trim())
            .arg("-n")
            .arg(n.to_string())
            .arg("-st")
            .arg("--simple-io")
            .output()
            .map_err(|e| InferenceError::Failed(format!("failed to spawn llama CLI: {e}")))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(InferenceError::Failed(format!(
                "llama CLI exited {}: {stderr}",
                output.status
            )));
        }

        let raw = String::from_utf8_lossy(&output.stdout);
        let text = clean_llama_stdout(&raw);
        let tokens_in = request.prompt.split_whitespace().count() as u32;
        let tokens_out = text.split_whitespace().count() as u32;
        Ok(InferenceResponse {
            model_id,
            model_name,
            text,
            tokens_in,
            tokens_out: tokens_out.min(request.max_tokens.max(1)),
            is_mock: false,
            backend: self.name().to_string(),
        })
    }
}

fn clean_llama_stdout(raw: &str) -> String {
    let mut s = raw.trim().to_string();
    for marker in ["> EOF by user", "\n> ", "[end of text]"] {
        if let Some(idx) = s.find(marker) {
            s.truncate(idx);
            s = s.trim().to_string();
        }
    }
    // Prefer text after the last "assistant" role line (ChatML / Zephyr dumps).
    for sep in ["\nassistant\n", "\n<|assistant|>\n", "<|assistant|>"] {
        if let Some(idx) = s.rfind(sep) {
            s = s[idx + sep.len()..].trim().to_string();
            break;
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gguf_attach_fails_without_cli_and_model() {
        let mut b = GgufInferenceBackend::new(GgufConfig {
            cli_path: PathBuf::from("/nonexistent/llama-cli-nfcm-test"),
            model_path: None,
        });
        let err = b
            .attach(&ModelContext {
                model_id: Uuid::nil(),
                model_name: "x".into(),
                architecture: "gguf".into(),
                weights_path: None,
                memory_requirement_bytes: 0,
                skills: vec![],
                latent_values: vec![],
                weights: vec![],
            })
            .unwrap_err();
        assert!(matches!(err, InferenceError::NotConfigured(_)));
    }
}
