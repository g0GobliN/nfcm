//! Default mock backend — honest placeholder, not a trained model.

use super::{
    BackendCapabilities, BackendId, InferenceBackend, InferenceError, InferenceRequest,
    InferenceResponse, ModelContext,
};
use uuid::Uuid;

pub struct MockInferenceBackend {
    attached: Option<(Uuid, String)>,
}

impl MockInferenceBackend {
    pub fn new() -> Self {
        Self { attached: None }
    }
}

impl Default for MockInferenceBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl InferenceBackend for MockInferenceBackend {
    fn id(&self) -> BackendId {
        BackendId::Mock
    }

    fn name(&self) -> &str {
        "mock-v1"
    }

    fn is_mock(&self) -> bool {
        true
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            always_available: true,
            loads_external_weights: false,
            notes: "Deterministic placeholder. Swap via NFCM_INFERENCE_BACKEND=candle|gguf.".into(),
        }
    }

    fn attach(&mut self, ctx: &ModelContext) -> Result<(), InferenceError> {
        self.attached = Some((ctx.model_id, ctx.model_name.clone()));
        Ok(())
    }

    fn detach(&mut self) -> Result<(), InferenceError> {
        self.attached = None;
        Ok(())
    }

    fn is_ready(&self) -> bool {
        self.attached.is_some()
    }

    fn infer(&self, request: &InferenceRequest) -> Result<InferenceResponse, InferenceError> {
        let (model_id, model_name) = self.attached.clone().ok_or(InferenceError::NotAttached)?;
        let tokens_in = request.prompt.split_whitespace().count() as u32;
        let reply = format!(
            "[NFCM mock inference — not a trained model]\n\
             Backend: mock-v1\n\
             Active brain: {model_name}\n\
             Prompt received ({tokens_in} tokens).\n\
             Plug in Candle (feature) or GGUF/llama.cpp via InferenceBackend."
        );
        let tokens_out = reply.split_whitespace().count() as u32;
        Ok(InferenceResponse {
            model_id,
            model_name,
            text: reply,
            tokens_in,
            tokens_out: tokens_out.min(request.max_tokens.max(1)),
            is_mock: true,
            backend: self.name().to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_requires_attach() {
        let b = MockInferenceBackend::new();
        let err = b
            .infer(&InferenceRequest {
                prompt: "hi".into(),
                max_tokens: 16,
            })
            .unwrap_err();
        assert!(matches!(err, InferenceError::NotAttached));
    }
}
