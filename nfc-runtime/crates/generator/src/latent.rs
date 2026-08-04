//! Phase 2 latent weight generator.
//!
//! Deterministic default or optional trained remix checkpoint:
//! - Encodes [`TaskProfile`] → [`LatentCode`]
//! - Chooses a small subnetwork topology from the latent
//! - Emits real tiny f32 tensors via outer-product recipe or checkpoint decode

use crate::checkpoint::HyperCheckpoint;
use crate::codebook::SkillCodebook;
use crate::pager::{PagerStats, SkillPager};
use crate::types::{
    GeneratedModel, LatentCode, LayerSpec, OptimizationProfile, TaskCategory, TaskProfile,
    WeightGenerator, WeightGeneratorError,
};
use nfc_tensor::{DType, Tensor, TensorShape};
use std::sync::Mutex;
use tracing::info;
use uuid::Uuid;

const LATENT_DIM: usize = 64;
const MAX_ALLOC_BYTES: u64 = 4 * 1024 * 1024;

pub struct LatentWeightGenerator {
    min_memory_bytes: u64,
    latent_dim: usize,
    checkpoint: Option<HyperCheckpoint>,
    pager: Mutex<SkillPager>,
}

impl Default for LatentWeightGenerator {
    fn default() -> Self {
        Self {
            min_memory_bytes: 16 * 1024 * 1024,
            latent_dim: LATENT_DIM,
            checkpoint: None,
            pager: Mutex::new(SkillPager::from_bank(SkillCodebook::builtin())),
        }
    }
}

impl LatentWeightGenerator {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_checkpoint(mut self, ckpt: HyperCheckpoint) -> Self {
        self.checkpoint = Some(ckpt);
        self
    }

    pub fn with_codebook(self, codebook: SkillCodebook) -> Self {
        Self {
            min_memory_bytes: self.min_memory_bytes,
            latent_dim: self.latent_dim,
            checkpoint: self.checkpoint,
            pager: Mutex::new(SkillPager::from_bank(codebook)),
        }
    }

    pub fn with_pager_budget(self, max_resident_bytes: u64) -> Self {
        let bank = self
            .pager
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .bank()
            .clone();
        Self {
            min_memory_bytes: self.min_memory_bytes,
            latent_dim: self.latent_dim,
            checkpoint: self.checkpoint,
            pager: Mutex::new(SkillPager::new(bank, max_resident_bytes)),
        }
    }

    pub fn from_env_checkpoint() -> Self {
        let mut gen = Self::new().with_codebook(crate::codebook::load_or_builtin());
        if let Some(path) = crate::checkpoint::resolve_checkpoint_path() {
            match HyperCheckpoint::load(&path) {
                Ok(ckpt) => {
                    info!(path = %path.display(), "loaded hypernetwork checkpoint");
                    gen.checkpoint = Some(ckpt);
                }
                Err(e) => {
                    tracing::warn!(error = %e, path = %path.display(), "failed to load checkpoint");
                }
            }
        }
        gen
    }

    pub fn has_checkpoint(&self) -> bool {
        self.checkpoint.is_some()
    }

    fn lock_pager(&self) -> std::sync::MutexGuard<'_, SkillPager> {
        self.pager.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn target_claim_bytes(&self, task: &TaskProfile) -> Result<u64, WeightGeneratorError> {
        if task.memory_limit_bytes < self.min_memory_bytes {
            return Err(WeightGeneratorError::MemoryLimitTooLow {
                limit_bytes: task.memory_limit_bytes,
            });
        }
        let target = (task.memory_limit_bytes as f64 * 0.55) as u64;
        Ok(target
            .max(self.min_memory_bytes)
            .min(task.memory_limit_bytes))
    }

    pub fn encode_latent(&self, task: &TaskProfile) -> LatentCode {
        self.encode_latent_with_hits(task).0
    }

    fn encode_latent_with_hits(&self, task: &TaskProfile) -> (LatentCode, Vec<String>) {
        let mut values = vec![0.0f32; self.latent_dim];
        let mut seed = fnv1a(task.domain.as_str().as_bytes());
        for skill in &task.skills {
            seed ^= fnv1a(skill.as_bytes());
            seed = seed.wrapping_mul(0x0100_0000_01b3);
        }
        if let Some(lang) = &task.language {
            seed ^= fnv1a(lang.as_bytes());
        }
        seed ^= task.memory_limit_bytes / (1024 * 1024);

        for (i, v) in values.iter_mut().enumerate() {
            let x = mix(seed, i as u64);
            *v = ((x % 10_000) as f32 / 5_000.0) - 1.0;
        }

        match task.domain {
            TaskCategory::Coding => values[0] += 0.35,
            TaskCategory::Math => values[1] += 0.35,
            TaskCategory::Writing => values[2] += 0.35,
            TaskCategory::Research => values[3] += 0.35,
            TaskCategory::Medical => values[4] += 0.35,
            TaskCategory::Custom => values[5] += 0.35,
        }

        let mut pager = self.lock_pager();
        let (activated, residual) = pager.activate(&task.skills);
        let n = values.len().min(residual.len());
        for i in 0..n {
            values[i] += residual[i];
        }

        let codebook_id = if activated.is_empty() {
            if self.checkpoint.is_some() {
                format!("hypernet-toy-{}", task.domain.as_str())
            } else {
                format!("latent-proto-{}", task.domain.as_str())
            }
        } else {
            format!("{}+{}", pager.bank().id, activated.join(","))
        };

        (
            LatentCode {
                dim: self.latent_dim,
                values,
                codebook_id,
            },
            activated,
        )
    }

    fn topology(latent: &LatentCode, claim_bytes: u64, skill_hits: usize) -> (usize, usize, usize) {
        let energy: f32 = latent.values.iter().map(|v| v.abs()).sum::<f32>() / latent.dim as f32;
        // Dynamic subnetwork: more activated skills → slightly wider/deeper specialist
        // (still capped by MAX_ALLOC_BYTES).
        let skill_boost = (skill_hits as f32 * 24.0).min(96.0);
        let hidden = ((48.0 + energy * 160.0 + skill_boost) as usize).clamp(48, 256);
        let depth = match skill_hits {
            0 => 2,
            1..=2 => {
                if energy > 0.6 {
                    3
                } else {
                    2
                }
            }
            _ => 3,
        };
        let max_elems = (MAX_ALLOC_BYTES / 4).max(1024) as usize;
        let in_dim = (max_elems / (hidden * depth.max(1))).clamp(32, 128);
        let _ = claim_bytes;
        (in_dim, hidden, depth)
    }

    fn synth_matrix(&self, latent: &LatentCode, rows: usize, cols: usize, salt: u64) -> Tensor {
        if let Some(ckpt) = &self.checkpoint {
            return Self::synth_from_checkpoint(ckpt, latent, rows, cols, salt);
        }
        let mut data = Vec::with_capacity(rows * cols * 4);
        let n = latent.values.len().max(1);
        for r in 0..rows {
            for c in 0..cols {
                let a = latent.values[r % n];
                let b = latent.values[c % n];
                let noise = ((mix(salt, (r as u64) << 32 | c as u64) % 1000) as f32) / 1000.0 - 0.5;
                let v = (a * b) * 0.15 + noise * 0.02;
                data.extend_from_slice(&v.to_le_bytes());
            }
        }
        Tensor {
            shape: TensorShape::new([rows, cols]),
            dtype: DType::F32,
            data,
        }
    }

    fn synth_from_checkpoint(
        ckpt: &HyperCheckpoint,
        latent: &LatentCode,
        rows: usize,
        cols: usize,
        salt: u64,
    ) -> Tensor {
        let mut z = latent.values.clone();
        z.resize(ckpt.latent_dim, 0.0);
        let u = ckpt.project_p(&z);
        let v = ckpt.project_q(&z);
        let mut data = Vec::with_capacity(rows * cols * 4);
        for r in 0..rows {
            for c in 0..cols {
                let a = u[r % u.len()];
                let b = v[c % v.len()];
                let noise = ((mix(salt, (r as u64) << 32 | c as u64) % 1000) as f32) / 1000.0 - 0.5;
                let val = ckpt.alpha * a * b + noise * 0.005;
                data.extend_from_slice(&val.to_le_bytes());
            }
        }
        Tensor {
            shape: TensorShape::new([rows, cols]),
            dtype: DType::F32,
            data,
        }
    }
}

impl WeightGenerator for LatentWeightGenerator {
    fn name(&self) -> &str {
        if self.checkpoint.is_some() {
            "latent-hypernet-v1"
        } else {
            "latent-proto-v1"
        }
    }

    fn generate(&self, task: TaskProfile) -> Result<GeneratedModel, WeightGeneratorError> {
        let claim = self.target_claim_bytes(&task)?;
        let (latent, activated) = self.encode_latent_with_hits(&task);
        let (in_dim, hidden, depth) = Self::topology(&latent, claim, activated.len());
        let pager_stats = self.lock_pager().stats();

        info!(
            domain = task.domain.as_str(),
            latent_dim = latent.dim,
            skill_hits = activated.len(),
            hot_skills = pager_stats.hot_skills,
            resident_bytes = pager_stats.resident_bytes,
            in_dim,
            hidden,
            depth,
            claim_mb = claim / (1024 * 1024),
            trained = self.checkpoint.is_some(),
            "latent weight generation"
        );

        let mut layers = Vec::new();
        let mut weights = Vec::new();

        let embed = self.synth_matrix(&latent, in_dim, hidden, 1);
        layers.push(LayerSpec {
            name: "latent.embed.weight".into(),
            shape: vec![in_dim, hidden],
            dtype: DType::F32,
        });
        weights.push(embed);

        for d in 0..depth {
            let w = self.synth_matrix(&latent, hidden, hidden, 10 + d as u64);
            layers.push(LayerSpec {
                name: format!("latent.block{d}.weight"),
                shape: vec![hidden, hidden],
                dtype: DType::F32,
            });
            weights.push(w);
        }

        let head = self.synth_matrix(&latent, hidden, 128, 99);
        layers.push(LayerSpec {
            name: "latent.head.weight".into(),
            shape: vec![hidden, 128],
            dtype: DType::F32,
        });
        weights.push(head);

        let alloc_bytes: u64 = weights.iter().map(|t| t.memory_bytes() as u64).sum();
        let name = specialist_name(&task, self.checkpoint.is_some());
        let notes = if self.checkpoint.is_some() {
            format!(
                "Trained toy hypernetwork decode (checkpoint) — allocated {alloc_bytes} B; \
                 claimed {claim} B; skills_hit={}; hot={}/{} ({} B). Still not an LLM.",
                activated.len(),
                pager_stats.hot_skills,
                pager_stats.bank_skills,
                pager_stats.resident_bytes,
            )
        } else {
            format!(
                "Phase 3 paged codebook `{}` — hit {} skills, hot {}/{} ({} B / {} B cap) → \
                 depth={depth} hidden={hidden} (allocated {alloc_bytes} B; claimed {claim} B)",
                pager_stats.codebook_id,
                activated.len(),
                pager_stats.hot_skills,
                pager_stats.bank_skills,
                pager_stats.resident_bytes,
                pager_stats.max_resident_bytes,
            )
        };

        Ok(GeneratedModel {
            id: Uuid::new_v4(),
            name,
            task,
            layers,
            weights,
            latent,
            memory_size_bytes: claim,
            optimization_profile: OptimizationProfile {
                quantize: true,
                target_memory_bytes: claim,
                activation_sparsity: 0.4,
                notes,
            },
            is_mock: false,
        })
    }

    fn optimize_pages(&self) -> u64 {
        self.lock_pager().optimize()
    }

    fn resident_skill_bytes(&self) -> u64 {
        self.lock_pager().resident_bytes()
    }

    fn pager_stats(&self) -> Option<PagerStats> {
        Some(self.lock_pager().stats())
    }
}

fn specialist_name(task: &TaskProfile, trained: bool) -> String {
    let base = match task.domain {
        TaskCategory::Coding => {
            if let Some(lang) = &task.language {
                format!("{} Coding Specialist", capitalize(lang))
            } else {
                "Coding Specialist".into()
            }
        }
        TaskCategory::Math => "Math Specialist".into(),
        TaskCategory::Writing => "Writing Specialist".into(),
        TaskCategory::Research => "Research Specialist".into(),
        TaskCategory::Medical => "Medical Text Specialist".into(),
        TaskCategory::Custom => "Custom Specialist".into(),
    };
    if trained {
        format!("{base} (hypernet)")
    } else {
        format!("{base} (latent)")
    }
}

fn capitalize(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        None => String::new(),
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
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

fn mix(seed: u64, i: u64) -> u64 {
    let mut x = seed ^ i.wrapping_mul(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn latent_produces_nonzero_weights() {
        let gen = LatentWeightGenerator::new();
        let task = TaskProfile {
            domain: TaskCategory::Coding,
            skills: vec!["python".into(), "debugging".into()],
            language: Some("python".into()),
            memory_limit_bytes: 256 * 1024 * 1024,
            raw_prompt: "python coding".into(),
        };
        let model = gen.generate(task).unwrap();
        assert!(!model.is_mock);
        assert_eq!(model.latent.dim, 64);
        assert!(!model.weights.is_empty());
        assert!(model.weights.iter().any(|w| w.memory_bytes() > 0));
        assert!(model.weights.iter().any(|w| w.data.iter().any(|b| *b != 0)));
        assert!(model.name.contains("latent"));
        assert!(
            model.latent.codebook_id.contains("python")
                || model.latent.codebook_id.contains("skill"),
            "expected codebook activation, got {}",
            model.latent.codebook_id
        );
    }

    #[test]
    fn same_task_is_deterministic() {
        let gen = LatentWeightGenerator::new();
        let task = TaskProfile {
            domain: TaskCategory::Math,
            skills: vec!["algebra".into()],
            language: None,
            memory_limit_bytes: 128 * 1024 * 1024,
            raw_prompt: "math".into(),
        };
        let a = gen.generate(task.clone()).unwrap();
        let b = gen.generate(task).unwrap();
        assert_eq!(a.latent.values, b.latent.values);
        assert_eq!(a.layers.len(), b.layers.len());
        assert_eq!(a.weights[0].data, b.weights[0].data);
    }

    #[test]
    fn more_skills_grows_subnetwork() {
        let gen = LatentWeightGenerator::new();
        let slim = TaskProfile {
            domain: TaskCategory::Coding,
            skills: vec!["python".into()],
            language: Some("python".into()),
            memory_limit_bytes: 256 * 1024 * 1024,
            raw_prompt: "python".into(),
        };
        let wide = TaskProfile {
            domain: TaskCategory::Coding,
            skills: vec![
                "python".into(),
                "debugging".into(),
                "testing".into(),
                "refactoring".into(),
            ],
            language: Some("python".into()),
            memory_limit_bytes: 256 * 1024 * 1024,
            raw_prompt: "python full".into(),
        };
        let a = gen.generate(slim).unwrap();
        let b = gen.generate(wide).unwrap();
        assert!(
            b.layers.len() >= a.layers.len(),
            "slim={} wide={}",
            a.layers.len(),
            b.layers.len()
        );
        let alloc_a: usize = a.weights.iter().map(|w| w.memory_bytes()).sum();
        let alloc_b: usize = b.weights.iter().map(|w| w.memory_bytes()).sum();
        assert!(
            alloc_b >= alloc_a,
            "expected wider specialist to allocate >= slim ({alloc_a} vs {alloc_b})"
        );
    }
}
