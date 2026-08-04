//! Latent decode inference — forward pass on TARC tensors + greedy text decode.
//!
//! Not an LLM. Emits real decoded tokens from activations over a fixed vocab,
//! plus a short diagnostic footer.

use super::decode::decode_tokens;
use super::{
    BackendCapabilities, BackendId, InferenceBackend, InferenceError, InferenceRequest,
    InferenceResponse, ModelContext,
};
use nfc_tensor::{DType, Tensor};
use uuid::Uuid;

struct LoadedSpec {
    model_id: Uuid,
    model_name: String,
    skills: Vec<String>,
    latent: Vec<f32>,
    /// Row-major f32 matrices as (rows, cols, data) where W is [rows × cols].
    matrices: Vec<(usize, usize, Vec<f32>)>,
}

pub struct LatentProbeBackend {
    loaded: Option<LoadedSpec>,
}

impl Default for LatentProbeBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl LatentProbeBackend {
    pub fn new() -> Self {
        Self { loaded: None }
    }

    fn parse_f32_matrix(t: &Tensor) -> Result<(usize, usize, Vec<f32>), InferenceError> {
        if t.dtype != DType::F32 {
            return Err(InferenceError::Failed(
                "latent probe expects f32 weights".into(),
            ));
        }
        if t.shape.dims.len() != 2 {
            return Err(InferenceError::Failed(format!(
                "expected rank-2 weight, got {:?}",
                t.shape.dims
            )));
        }
        let rows = t.shape.dims[0];
        let cols = t.shape.dims[1];
        let mut vals = Vec::with_capacity(rows * cols);
        for chunk in t.data.chunks_exact(4) {
            vals.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        if vals.len() != rows * cols {
            return Err(InferenceError::Failed("weight byte length mismatch".into()));
        }
        Ok((rows, cols, vals))
    }

    fn embed_prompt(prompt: &str, dim: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; dim];
        for (i, tok) in prompt.split_whitespace().enumerate() {
            let h = fnv1a(tok.as_bytes());
            let idx = (h as usize) % dim;
            v[idx] += 1.0;
            v[(idx + i + 1) % dim] += 0.25;
        }
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
        for x in &mut v {
            *x /= norm;
        }
        v
    }

    /// x ∈ R^{rows}, W[rows, cols] → y ∈ R^{cols}, then ReLU.
    fn matvec(rows: usize, cols: usize, w: &[f32], x: &[f32]) -> Vec<f32> {
        let mut outv = vec![0.0f32; cols];
        for r in 0..rows {
            let xr = x.get(r).copied().unwrap_or(0.0);
            for c in 0..cols {
                outv[c] += xr * w[r * cols + c];
            }
        }
        for v in &mut outv {
            if *v < 0.0 {
                *v = 0.0;
            }
        }
        outv
    }

    fn fit_len(mut x: Vec<f32>, n: usize) -> Vec<f32> {
        if x.len() == n {
            return x;
        }
        x.resize(n, 0.0);
        x
    }

    fn forward(matrices: &[(usize, usize, Vec<f32>)], prompt: &str) -> (Vec<f32>, f32) {
        if matrices.is_empty() {
            return (vec![], 0.0);
        }
        let (in_rows, _, _) = matrices[0];
        let mut x = Self::embed_prompt(prompt, in_rows);
        for (rows, cols, w) in matrices {
            x = Self::fit_len(x, *rows);
            x = Self::matvec(*rows, *cols, w, &x);
        }
        let energy = x.iter().map(|v| v.abs()).sum::<f32>() / x.len().max(1) as f32;
        (x, energy)
    }

    /// Activation energy for the current loaded brain (eval / routing metrics).
    pub fn activation_energy(&self, prompt: &str) -> Result<f32, InferenceError> {
        let spec = self.loaded.as_ref().ok_or(InferenceError::NotAttached)?;
        let (_acts, energy) = Self::forward(&spec.matrices, prompt);
        Ok(energy)
    }
}

impl InferenceBackend for LatentProbeBackend {
    fn id(&self) -> BackendId {
        BackendId::Latent
    }

    fn name(&self) -> &str {
        "latent-decode-v1"
    }

    fn is_mock(&self) -> bool {
        false
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            always_available: true,
            loads_external_weights: true,
            notes: "TARC forward + greedy fixed-vocab decode — not a trained LLM.".into(),
        }
    }

    fn attach(&mut self, ctx: &ModelContext) -> Result<(), InferenceError> {
        if ctx.weights.is_empty() {
            return Err(InferenceError::NotConfigured(
                "latent-decode needs GeneratedModel weights (use NFCM_WEIGHT_GENERATOR=latent)"
                    .into(),
            ));
        }
        let mut matrices = Vec::new();
        for t in &ctx.weights {
            matrices.push(Self::parse_f32_matrix(t)?);
        }
        self.loaded = Some(LoadedSpec {
            model_id: ctx.model_id,
            model_name: ctx.model_name.clone(),
            skills: ctx.skills.clone(),
            latent: ctx.latent_values.clone(),
            matrices,
        });
        Ok(())
    }

    fn detach(&mut self) -> Result<(), InferenceError> {
        self.loaded = None;
        Ok(())
    }

    fn is_ready(&self) -> bool {
        self.loaded.is_some()
    }

    fn infer(&self, request: &InferenceRequest) -> Result<InferenceResponse, InferenceError> {
        let spec = self.loaded.as_ref().ok_or(InferenceError::NotAttached)?;
        let (acts, energy) = Self::forward(&spec.matrices, &request.prompt);
        let tokens_in = request.prompt.split_whitespace().count() as u32;

        let decoded = decode_tokens(
            &acts,
            &spec.latent,
            &spec.skills,
            &request.prompt,
            request.max_tokens as usize,
        );

        let mut skill_bits = Vec::new();
        for (i, skill) in spec.skills.iter().enumerate().take(6) {
            let score = if spec.latent.is_empty() || acts.is_empty() {
                0.0
            } else {
                let a = spec.latent[i % spec.latent.len()].abs();
                let b = acts[i % acts.len()].abs();
                (a * 0.5 + b * 0.5).min(1.0)
            };
            skill_bits.push(format!("{skill}={score:.2}"));
        }

        let text = format!(
            "{decoded}\n\n\
             —\n\
             [NFCM latent-decode · NOT a trained LLM]\n\
             Brain: {} · energy={energy:.3} · layers={}\n\
             Skills: {}",
            spec.model_name,
            spec.matrices.len(),
            if skill_bits.is_empty() {
                "(none)".into()
            } else {
                skill_bits.join(", ")
            }
        );

        let tokens_out = decoded.split_whitespace().count() as u32;
        Ok(InferenceResponse {
            model_id: spec.model_id,
            model_name: spec.model_name.clone(),
            text,
            tokens_in,
            tokens_out: tokens_out.min(request.max_tokens.max(1)),
            is_mock: false,
            backend: self.name().to_string(),
        })
    }
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for b in bytes {
        hash ^= u64::from(*b);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use nfc_generator::{LatentWeightGenerator, TaskCategory, TaskProfile, WeightGenerator};

    #[test]
    fn probe_runs_on_latent_weights() {
        let gen = LatentWeightGenerator::new();
        let task = TaskProfile {
            domain: TaskCategory::Coding,
            skills: vec!["python".into(), "debugging".into()],
            language: Some("python".into()),
            memory_limit_bytes: 128 * 1024 * 1024,
            raw_prompt: "python help".into(),
        };
        let model = gen.generate(task).unwrap();
        let mut backend = LatentProbeBackend::new();
        backend
            .attach(&ModelContext {
                model_id: model.id,
                model_name: model.name.clone(),
                architecture: "nfcm".into(),
                weights_path: None,
                memory_requirement_bytes: model.memory_size_bytes,
                skills: model.task.skills.clone(),
                latent_values: model.latent.values.clone(),
                weights: model.weights.clone(),
            })
            .unwrap();
        let resp = backend
            .infer(&InferenceRequest {
                prompt: "fix my rust borrow checker error".into(),
                max_tokens: 32,
            })
            .unwrap();
        assert!(!resp.is_mock);
        assert_eq!(resp.backend, "latent-decode-v1");
        assert!(resp.text.contains("latent-decode"));
        let main = resp.text.split("\n\n—").next().unwrap_or("");
        assert!(main.split_whitespace().count() >= 4);
    }

    #[test]
    fn attach_fails_without_weights() {
        let mut backend = LatentProbeBackend::new();
        let err = backend
            .attach(&ModelContext {
                model_id: Uuid::nil(),
                model_name: "x".into(),
                architecture: "x".into(),
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
