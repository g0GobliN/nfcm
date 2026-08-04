//! Optional trained hypernetwork checkpoint for latent decode.
//!
//! Format: JSON with remix matrices `P`, `Q` and scale `alpha`:
//!   W[r,c] = alpha * (P @ z)[r % rows_p] * (Q @ z)[c % rows_q]
//!
//! Train with `experiments/neural-generation/hypernetwork/train_toy.py`.

use serde::{Deserialize, Serialize};
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CheckpointError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("invalid checkpoint: {0}")]
    Invalid(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperCheckpoint {
    pub version: u32,
    pub latent_dim: usize,
    pub alpha: f32,
    /// Row-major f32, shape [rows_p, latent_dim]
    pub p_rows: usize,
    pub p_cols: usize,
    pub p: Vec<f32>,
    /// Row-major f32, shape [rows_q, latent_dim]
    pub q_rows: usize,
    pub q_cols: usize,
    pub q: Vec<f32>,
    pub notes: String,
}

impl HyperCheckpoint {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, CheckpointError> {
        let bytes = std::fs::read(path.as_ref())?;
        let ckpt: Self = serde_json::from_slice(&bytes)?;
        ckpt.validate()?;
        Ok(ckpt)
    }

    pub fn validate(&self) -> Result<(), CheckpointError> {
        if self.version != 1 {
            return Err(CheckpointError::Invalid(format!(
                "unsupported version {}",
                self.version
            )));
        }
        if self.p.len() != self.p_rows * self.p_cols {
            return Err(CheckpointError::Invalid("P size mismatch".into()));
        }
        if self.q.len() != self.q_rows * self.q_cols {
            return Err(CheckpointError::Invalid("Q size mismatch".into()));
        }
        if self.p_cols != self.latent_dim || self.q_cols != self.latent_dim {
            return Err(CheckpointError::Invalid("latent_dim mismatch".into()));
        }
        Ok(())
    }

    /// Project latent z through P → length rows_p.
    pub fn project_p(&self, z: &[f32]) -> Vec<f32> {
        matvec(self.p_rows, self.p_cols, &self.p, z)
    }

    pub fn project_q(&self, z: &[f32]) -> Vec<f32> {
        matvec(self.q_rows, self.q_cols, &self.q, z)
    }
}

fn matvec(rows: usize, cols: usize, w: &[f32], x: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; rows];
    for r in 0..rows {
        let mut s = 0.0f32;
        for c in 0..cols {
            let xv = x.get(c).copied().unwrap_or(0.0);
            s += w[r * cols + c] * xv;
        }
        out[r] = s;
    }
    out
}

/// Resolve checkpoint path: env `NFCM_HYPERNET_CHECKPOINT`, else default under repo/experiments.
pub fn resolve_checkpoint_path() -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var("NFCM_HYPERNET_CHECKPOINT") {
        let path = std::path::PathBuf::from(p);
        if path.is_file() {
            return Some(path);
        }
    }
    // Prefer task-metric checkpoint, then toy-v1.
    let mut dir = std::env::current_dir().ok()?;
    for _ in 0..6 {
        for rel in [
            "experiments/neural-generation/hypernetwork/checkpoints/task-v1.json",
            "nfc-runtime/experiments/neural-generation/hypernetwork/checkpoints/task-v1.json",
            "experiments/neural-generation/hypernetwork/checkpoints/toy-v1.json",
            "nfc-runtime/experiments/neural-generation/hypernetwork/checkpoints/toy-v1.json",
        ] {
            let candidate = dir.join(rel);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
        if !dir.pop() {
            break;
        }
    }
    None
}
