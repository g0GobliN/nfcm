//! Candle backend seam.
//!
//! - Default build: compile-time stub that explains how to enable the feature.
//! - `candle` feature: uses `candle-core` for a tiny health-check tensor op.
//!   Still does **not** load an LLM unless weights are provided later.

use super::{
    BackendCapabilities, BackendId, InferenceBackend, InferenceError, InferenceRequest,
    InferenceResponse, ModelContext,
};
use uuid::Uuid;

pub struct CandleInferenceBackend {
    attached: Option<(Uuid, String)>,
    #[cfg(feature = "candle")]
    candle_ok: bool,
}

impl CandleInferenceBackend {
    pub fn new() -> Self {
        Self {
            attached: None,
            #[cfg(feature = "candle")]
            candle_ok: true,
        }
    }
}

impl Default for CandleInferenceBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl InferenceBackend for CandleInferenceBackend {
    fn id(&self) -> BackendId {
        BackendId::Candle
    }

    fn name(&self) -> &str {
        "candle-v1"
    }

    fn is_mock(&self) -> bool {
        // Scaffold only until generative weights / decode are wired.
        true
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            always_available: cfg!(feature = "candle"),
            loads_external_weights: true,
            notes: if cfg!(feature = "candle") {
                "Candle feature enabled. Generative weights not wired yet — health tensor only."
                    .into()
            } else {
                "Build with `--features candle` to enable candle-core. Then load weights.".into()
            },
        }
    }

    fn attach(&mut self, ctx: &ModelContext) -> Result<(), InferenceError> {
        #[cfg(not(feature = "candle"))]
        {
            let _ = ctx;
            Err(InferenceError::FeatureDisabled(
                "rebuild nfc-runtime with `--features candle` (candle-core)".into(),
            ))
        }
        #[cfg(feature = "candle")]
        {
            // Prove the Candle stack is linked: allocate a tiny tensor.
            use candle_core::{Device, Tensor};
            let t = Tensor::zeros((2, 2), candle_core::DType::F32, &Device::Cpu)
                .map_err(|e| InferenceError::Failed(format!("candle health tensor failed: {e}")))?;
            let _ = t
                .sum_all()
                .map_err(|e| InferenceError::Failed(e.to_string()))?;
            self.candle_ok = true;
            self.attached = Some((ctx.model_id, ctx.model_name.clone()));
            Ok(())
        }
    }

    fn detach(&mut self) -> Result<(), InferenceError> {
        self.attached = None;
        Ok(())
    }

    fn is_ready(&self) -> bool {
        #[cfg(feature = "candle")]
        {
            self.attached.is_some() && self.candle_ok
        }
        #[cfg(not(feature = "candle"))]
        {
            false
        }
    }

    fn infer(&self, request: &InferenceRequest) -> Result<InferenceResponse, InferenceError> {
        #[cfg(not(feature = "candle"))]
        {
            let _ = request;
            Err(InferenceError::FeatureDisabled(
                "candle feature not enabled in this build".into(),
            ))
        }
        #[cfg(feature = "candle")]
        {
            let (model_id, model_name) =
                self.attached.clone().ok_or(InferenceError::NotAttached)?;
            let tokens_in = request.prompt.split_whitespace().count() as u32;
            // Honest scaffold: Candle is linked; no generative weights yet.
            let reply = format!(
                "[NFCM Candle backend — scaffold, not a trained LLM]\n\
                 Backend: candle-v1 (candle-core health check OK)\n\
                 Active brain: {model_name}\n\
                 Prompt ({tokens_in} tokens). Generative decode not implemented yet.\n\
                 Next: load safetensors / hypernetwork weights into this backend."
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
}
