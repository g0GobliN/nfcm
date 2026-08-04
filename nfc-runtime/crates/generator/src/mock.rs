//! Mock weight generator — deterministic placeholder, not a real model.
//!
//! Phase 1 does **not** allocate full weight buffers (that would defeat the
//! low-memory goal). It records layer shapes + a claimed memory footprint so
//! the runtime memory manager and UI can be exercised realistically.

use crate::types::{
    GeneratedModel, LatentCode, LayerSpec, OptimizationProfile, TaskCategory, TaskProfile,
    WeightGenerator, WeightGeneratorError,
};
use nfc_tensor::DType;
use tracing::info;
use uuid::Uuid;

/// Produces symbolic layer specs sized to fit the task memory budget.
pub struct MockWeightGenerator {
    min_memory_bytes: u64,
}

impl Default for MockWeightGenerator {
    fn default() -> Self {
        Self {
            min_memory_bytes: 16 * 1024 * 1024, // 16 MiB floor for mock artifacts
        }
    }
}

impl MockWeightGenerator {
    pub fn new() -> Self {
        Self::default()
    }

    fn target_bytes(
        task: &TaskProfile,
        min_memory_bytes: u64,
    ) -> Result<u64, WeightGeneratorError> {
        if task.memory_limit_bytes < min_memory_bytes {
            return Err(WeightGeneratorError::MemoryLimitTooLow {
                limit_bytes: task.memory_limit_bytes,
            });
        }
        // Use ~60% of the budget for the mock active model payload.
        let target = (task.memory_limit_bytes as f64 * 0.6) as u64;
        Ok(target.max(min_memory_bytes).min(task.memory_limit_bytes))
    }
}

impl WeightGenerator for MockWeightGenerator {
    fn name(&self) -> &str {
        "mock-v1"
    }

    fn generate(&self, task: TaskProfile) -> Result<GeneratedModel, WeightGeneratorError> {
        let target = Self::target_bytes(&task, self.min_memory_bytes)?;
        info!(
            domain = task.domain.as_str(),
            target_mb = target / (1024 * 1024),
            "mock weight generation (metadata only — no full tensor alloc)"
        );

        let elems = (target / 4).max(1024);
        let hidden = ((elems as f64).sqrt() as usize).clamp(64, 4096);
        let in_dim = (elems / hidden as u64).max(64) as usize;

        let layers = vec![
            LayerSpec {
                name: "embed.weight".into(),
                shape: vec![in_dim, hidden],
                dtype: DType::F32,
            },
            LayerSpec {
                name: "head.weight".into(),
                shape: vec![hidden, 256],
                dtype: DType::F32,
            },
        ];

        let memory_size_bytes = layers
            .iter()
            .map(|l| {
                let numel: u64 = l.shape.iter().map(|d| *d as u64).product();
                numel * l.dtype.size_bytes() as u64
            })
            .sum();

        let name = specialist_name(&task);
        let latent = LatentCode {
            dim: 32,
            values: pseudo_latent(&task),
            codebook_id: format!("mock-{}", task.domain.as_str()),
        };

        Ok(GeneratedModel {
            id: Uuid::new_v4(),
            name,
            task,
            layers,
            // Empty weights: real hypernetwork will fill these later.
            weights: vec![],
            latent,
            memory_size_bytes,
            optimization_profile: OptimizationProfile {
                quantize: true,
                target_memory_bytes: target,
                activation_sparsity: 0.35,
                notes: "Phase 1 mock generator — metadata only, not a trained model".into(),
            },
            is_mock: true,
        })
    }
}

fn specialist_name(task: &TaskProfile) -> String {
    match task.domain {
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
    }
}

fn capitalize(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        None => String::new(),
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
    }
}

fn pseudo_latent(task: &TaskProfile) -> Vec<f32> {
    let mut values = vec![0.0f32; 32];
    let seed = task.domain.as_str().bytes().map(|b| b as u32).sum::<u32>()
        + task.skills.iter().map(|s| s.len() as u32).sum::<u32>();
    for (i, v) in values.iter_mut().enumerate() {
        let x = ((seed as usize * 2654435761) ^ (i * 97)) as f32;
        *v = (x % 1000.0) / 1000.0 - 0.5;
    }
    values
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::TaskCategory;

    #[test]
    fn mock_generates_within_budget() {
        let gen = MockWeightGenerator::new();
        let task = TaskProfile {
            domain: TaskCategory::Coding,
            skills: vec!["python".into()],
            language: Some("python".into()),
            memory_limit_bytes: 256 * 1024 * 1024,
            raw_prompt: "python coding".into(),
        };
        let model = gen.generate(task).unwrap();
        assert!(model.is_mock);
        assert!(model.memory_size_bytes > 0);
        assert!(model.memory_size_bytes <= 256 * 1024 * 1024);
        assert_eq!(model.layers.len(), 2);
        assert!(model.weights.is_empty());
    }
}
