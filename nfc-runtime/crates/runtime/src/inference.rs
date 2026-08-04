//! Placeholder inference — clearly labeled as mock, no fake capability claims.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

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
}

/// Extremely small deterministic stub used until a real backend (Candle/ONNX/llama.cpp) is wired.
pub fn mock_infer(
    model_id: Uuid,
    model_name: &str,
    request: &InferenceRequest,
) -> InferenceResponse {
    let tokens_in = request.prompt.split_whitespace().count() as u32;
    let reply = format!(
        "[NFCM mock inference — not a trained model]\n\
         Active brain: {model_name}\n\
         Prompt received ({tokens_in} tokens).\n\
         Real generation will plug in via WeightGenerator + inference backend."
    );
    let tokens_out = reply.split_whitespace().count() as u32;
    InferenceResponse {
        model_id,
        model_name: model_name.to_string(),
        text: reply,
        tokens_in,
        tokens_out: tokens_out.min(request.max_tokens),
        is_mock: true,
    }
}
