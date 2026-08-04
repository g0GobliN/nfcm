//! Pluggable inference backends.
//!
//! Phase 1 default is [`MockInferenceBackend`]. Candle and GGUF/llama.cpp are
//! real extension points — they do not claim trained-model quality unless a
//! concrete weight file is loaded and the backend reports `is_mock: false`.

mod candle_backend;
mod gguf;
mod mock;
mod types;

pub use candle_backend::CandleInferenceBackend;
pub use gguf::GgufInferenceBackend;
pub use mock::MockInferenceBackend;
pub use types::{
    BackendCapabilities, BackendId, BackendInfo, InferenceError, InferenceRequest,
    InferenceResponse, ModelContext,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum BackendKind {
    #[default]
    Mock,
    Candle,
    Gguf,
}

impl BackendKind {
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "mock" => Some(Self::Mock),
            "candle" => Some(Self::Candle),
            "gguf" | "llama" | "llama.cpp" | "llamacpp" => Some(Self::Gguf),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Mock => "mock",
            Self::Candle => "candle",
            Self::Gguf => "gguf",
        }
    }
}

/// Contract for all inference engines plugged into [`crate::RuntimeEngine`].
pub trait InferenceBackend: Send + Sync {
    fn id(&self) -> BackendId;
    fn name(&self) -> &str;
    fn is_mock(&self) -> bool;
    fn capabilities(&self) -> BackendCapabilities;

    /// Bind to a loaded registry model (weights path may be absent for mocks).
    fn attach(&mut self, ctx: &ModelContext) -> Result<(), InferenceError>;

    fn detach(&mut self) -> Result<(), InferenceError>;

    fn is_ready(&self) -> bool;

    fn infer(&self, request: &InferenceRequest) -> Result<InferenceResponse, InferenceError>;

    fn info(&self) -> BackendInfo {
        BackendInfo {
            id: self.id(),
            name: self.name().to_string(),
            is_mock: self.is_mock(),
            ready: self.is_ready(),
            capabilities: self.capabilities(),
        }
    }
}

/// Build a backend instance. Candle/GGUF may still need env/path config to become ready.
pub fn create_backend(kind: BackendKind) -> Box<dyn InferenceBackend> {
    match kind {
        BackendKind::Mock => Box::new(MockInferenceBackend::new()),
        BackendKind::Candle => Box::new(CandleInferenceBackend::new()),
        BackendKind::Gguf => Box::new(GgufInferenceBackend::from_env()),
    }
}

pub fn backend_kind_from_env() -> BackendKind {
    std::env::var("NFCM_INFERENCE_BACKEND")
        .ok()
        .and_then(|s| BackendKind::parse(&s))
        .unwrap_or_default()
}
