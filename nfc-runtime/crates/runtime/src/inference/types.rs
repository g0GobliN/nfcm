//! Shared inference types.

use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendId {
    Mock,
    Candle,
    Gguf,
}

impl BackendId {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Mock => "mock",
            Self::Candle => "candle",
            Self::Gguf => "gguf",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendCapabilities {
    /// Can run without an external binary / feature.
    pub always_available: bool,
    /// Intended for GGUF / Candle weight files.
    pub loads_external_weights: bool,
    pub notes: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendInfo {
    pub id: BackendId,
    pub name: String,
    pub is_mock: bool,
    pub ready: bool,
    pub capabilities: BackendCapabilities,
}

/// Context passed when a registry model is loaded into the backend.
#[derive(Debug, Clone)]
pub struct ModelContext {
    pub model_id: Uuid,
    pub model_name: String,
    pub architecture: String,
    pub weights_path: Option<String>,
    pub memory_requirement_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceRequest {
    pub prompt: String,
    pub max_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResponse {
    pub model_id: Uuid,
    pub model_name: String,
    pub text: String,
    pub tokens_in: u32,
    pub tokens_out: u32,
    pub is_mock: bool,
    /// Which backend produced this response.
    pub backend: String,
}

#[derive(Debug, Error)]
pub enum InferenceError {
    #[error("backend not attached to a model")]
    NotAttached,
    #[error("backend not ready: {0}")]
    NotReady(String),
    #[error("backend not configured: {0}")]
    NotConfigured(String),
    #[error("inference failed: {0}")]
    Failed(String),
    #[error("feature disabled: {0}")]
    FeatureDisabled(String),
}
