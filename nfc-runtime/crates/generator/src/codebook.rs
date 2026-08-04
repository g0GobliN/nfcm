//! Phase 3 skill codebook — compressed skill residuals activated per task.
//!
//! Stores tiny fixed-size vectors keyed by skill name. At compile time the
//! generator sums the matching entries into the latent (additive residual).
//! Default bank is deterministic / untrained; swap the JSON to plug in trained
//! residuals later without changing the runtime seam.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use thiserror::Error;

pub const DEFAULT_CODEBOOK_ID: &str = "skill-codebook-v1";
pub const DEFAULT_DIM: usize = 128;

#[derive(Debug, Error)]
pub enum CodebookError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("invalid codebook: {0}")]
    Invalid(String),
}

/// On-disk compressed skill bank.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillCodebook {
    pub id: String,
    pub dim: usize,
    pub version: u32,
    /// Honest labels (trained vs deterministic stub).
    pub notes: String,
    pub entries: BTreeMap<String, Vec<f32>>,
}

impl SkillCodebook {
    pub fn empty(id: impl Into<String>, dim: usize) -> Self {
        Self {
            id: id.into(),
            dim,
            version: 1,
            notes: "empty codebook".into(),
            entries: BTreeMap::new(),
        }
    }

    /// Built-in deterministic bank so Phase 3 works offline with zero setup.
    pub fn builtin() -> Self {
        let mut cb = Self {
            id: DEFAULT_CODEBOOK_ID.into(),
            dim: DEFAULT_DIM,
            version: 1,
            notes: "Deterministic skill residuals (untrained) — replace via NFCM_CODEBOOK.".into(),
            entries: BTreeMap::new(),
        };
        for (skill, salt) in BUILTIN_SKILLS {
            cb.entries
                .insert((*skill).into(), synth_entry(DEFAULT_DIM, *salt));
        }
        cb
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self, CodebookError> {
        let raw = std::fs::read_to_string(path)?;
        let cb: Self = serde_json::from_str(&raw)?;
        cb.validate()?;
        Ok(cb)
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), CodebookError> {
        self.validate()?;
        if let Some(parent) = path.as_ref().parent() {
            std::fs::create_dir_all(parent)?;
        }
        let raw = serde_json::to_string_pretty(self)?;
        std::fs::write(path, raw)?;
        Ok(())
    }

    fn validate(&self) -> Result<(), CodebookError> {
        if self.dim == 0 {
            return Err(CodebookError::Invalid("dim must be > 0".into()));
        }
        for (k, v) in &self.entries {
            if v.len() != self.dim {
                return Err(CodebookError::Invalid(format!(
                    "skill `{k}` has len {}, expected {}",
                    v.len(),
                    self.dim
                )));
            }
        }
        Ok(())
    }

    /// Sum residuals for requested skills (case-insensitive key match).
    /// Returns (activated_skill_names, residual_vector).
    pub fn activate(&self, skills: &[String]) -> (Vec<String>, Vec<f32>) {
        let mut residual = vec![0.0f32; self.dim];
        let mut hit = Vec::new();
        for skill in skills {
            let key = skill.trim().to_ascii_lowercase();
            if key.is_empty() {
                continue;
            }
            if let Some(entry) = self.entries.get(&key) {
                for (i, v) in residual.iter_mut().enumerate() {
                    *v += entry[i];
                }
                hit.push(key);
            }
        }
        // Keep energy bounded so topology stays stable.
        let scale = 0.22f32;
        for v in &mut residual {
            *v *= scale;
        }
        (hit, residual)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

const BUILTIN_SKILLS: &[(&str, u64)] = &[
    ("python", 0x5079_7468),
    ("rust", 0x7275_7374),
    ("javascript", 0x6a73_6372),
    ("typescript", 0x7473_6372),
    ("debugging", 0x6465_6275),
    ("testing", 0x7465_7374),
    ("refactoring", 0x7265_6661),
    ("algorithms", 0x616c_676f),
    ("math", 0x6d61_7468),
    ("calculus", 0x6361_6c63),
    ("statistics", 0x7374_6174),
    ("writing", 0x7772_6974),
    ("research", 0x7265_7365),
    ("summarization", 0x7375_6d6d),
    ("medical", 0x6d65_6469),
];

fn synth_entry(dim: usize, salt: u64) -> Vec<f32> {
    let mut out = Vec::with_capacity(dim);
    for i in 0..dim {
        let x = mix(salt, i as u64);
        let v = ((x % 10_000) as f32 / 5_000.0) - 1.0;
        out.push(v * 0.85);
    }
    out
}

fn mix(seed: u64, i: u64) -> u64 {
    let mut x = seed ^ i.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// Resolve codebook path: `NFCM_CODEBOOK`, then trained/default experiment files, else builtin.
pub fn resolve_codebook_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("NFCM_CODEBOOK") {
        let pb = PathBuf::from(p);
        if pb.is_file() {
            return Some(pb);
        }
    }
    let candidates = [
        PathBuf::from("experiments/neural-generation/codebook/checkpoints/skill-task-v2.json"),
        PathBuf::from(
            "nfc-runtime/experiments/neural-generation/codebook/checkpoints/skill-task-v2.json",
        ),
        PathBuf::from("experiments/neural-generation/codebook/checkpoints/skill-task-v1.json"),
        PathBuf::from(
            "nfc-runtime/experiments/neural-generation/codebook/checkpoints/skill-task-v1.json",
        ),
        PathBuf::from("experiments/neural-generation/codebook/checkpoints/skill-trained-v1.json"),
        PathBuf::from(
            "nfc-runtime/experiments/neural-generation/codebook/checkpoints/skill-trained-v1.json",
        ),
        PathBuf::from("experiments/neural-generation/codebook/skill-v1.json"),
        PathBuf::from("nfc-runtime/experiments/neural-generation/codebook/skill-v1.json"),
    ];
    for c in candidates {
        if c.is_file() {
            return Some(c);
        }
    }
    if let Ok(manifest) = std::env::var("CARGO_MANIFEST_DIR") {
        let root = PathBuf::from(manifest).join("../../experiments/neural-generation/codebook");
        for name in [
            "checkpoints/skill-task-v2.json",
            "checkpoints/skill-task-v1.json",
            "checkpoints/skill-trained-v1.json",
            "skill-v1.json",
        ] {
            let p = root.join(name);
            if p.is_file() {
                return Some(p);
            }
        }
    }
    None
}

pub fn load_or_builtin() -> SkillCodebook {
    if let Some(path) = resolve_codebook_path() {
        match SkillCodebook::load(&path) {
            Ok(cb) => {
                tracing::info!(
                    path = %path.display(),
                    id = %cb.id,
                    skills = cb.len(),
                    "loaded skill codebook"
                );
                return cb;
            }
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    path = %path.display(),
                    "codebook load failed; using builtin"
                );
            }
        }
    }
    SkillCodebook::builtin()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_activates_known_skills() {
        let cb = SkillCodebook::builtin();
        let (hit, residual) = cb.activate(&["Python".into(), "debugging".into(), "nope".into()]);
        assert_eq!(hit, vec!["python".to_string(), "debugging".to_string()]);
        assert_eq!(residual.len(), DEFAULT_DIM);
        assert!(residual.iter().any(|v| *v != 0.0));
    }

    #[test]
    fn roundtrip_json() {
        let cb = SkillCodebook::builtin();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cb.json");
        cb.save(&path).unwrap();
        let loaded = SkillCodebook::load(&path).unwrap();
        assert_eq!(loaded.id, cb.id);
        assert_eq!(loaded.len(), cb.len());
    }
}
