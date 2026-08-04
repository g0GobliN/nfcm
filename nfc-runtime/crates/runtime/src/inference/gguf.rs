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
            .unwrap_or_else(|_| PathBuf::from("llama-cli"));
        let model = std::env::var("NFCM_GGUF_MODEL")
            .ok()
            .filter(|s| !s.is_empty())
            .map(PathBuf::from);
        Self {
            cli_path: cli,
            model_path: model,
        }
    }
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
        if let Some(p) = &self.config.model_path {
            if p.exists() {
                return Ok(p.clone());
            }
            return Err(InferenceError::NotConfigured(format!(
                "NFCM_GGUF_MODEL does not exist: {}",
                p.display()
            )));
        }
        if let Some(path) = &ctx.weights_path {
            let p = PathBuf::from(path);
            if p.extension().and_then(|e| e.to_str()) == Some("gguf") && p.exists() {
                return Ok(p);
            }
        }
        Err(InferenceError::NotConfigured(
            "set NFCM_GGUF_MODEL to a .gguf file, or load a Model with a .gguf path".into(),
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
        let output = Command::new(&self.config.cli_path)
            .arg("-m")
            .arg(model_path)
            .arg("-p")
            .arg(&request.prompt)
            .arg("-n")
            .arg(n.to_string())
            .arg("--log-disable")
            .output()
            .map_err(|e| InferenceError::Failed(format!("failed to spawn llama CLI: {e}")))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(InferenceError::Failed(format!(
                "llama CLI exited {}: {stderr}",
                output.status
            )));
        }

        let text = String::from_utf8_lossy(&output.stdout).trim().to_string();
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
            })
            .unwrap_err();
        assert!(matches!(err, InferenceError::NotConfigured(_)));
    }
}
