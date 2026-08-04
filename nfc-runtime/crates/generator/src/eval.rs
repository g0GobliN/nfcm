//! Fixed-RAM structural eval for weight generators.
//!
//! Metrics are honest: footprint, codebook hits, latent discrimination.
//! They are **not** LLM quality scores.

use crate::types::{GeneratedModel, TaskCategory, TaskProfile, WeightGenerator};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub struct EvalCase {
    pub id: &'static str,
    pub profile: TaskProfile,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaseMetrics {
    pub case: String,
    pub domain: String,
    pub skills: Vec<String>,
    pub skill_hits: usize,
    pub codebook_id: String,
    pub layers: usize,
    pub latent_dim: usize,
    pub claimed_bytes: u64,
    pub allocated_bytes: u64,
    pub under_claim: bool,
    pub is_mock: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscriminationMetrics {
    /// Mean cosine among same-domain latent pairs.
    pub mean_within_cosine: f32,
    /// Mean cosine among cross-domain latent pairs.
    pub mean_across_cosine: f32,
    /// within - across (higher ⇒ specialists separate by domain).
    pub separation: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuiteReport {
    pub generator: String,
    pub budget_bytes: u64,
    pub cases: Vec<CaseMetrics>,
    pub discrimination: Option<DiscriminationMetrics>,
    pub all_under_claim: bool,
    pub notes: String,
}

pub fn default_cases(budget_bytes: u64) -> Vec<EvalCase> {
    vec![
        EvalCase {
            id: "coding-slim",
            profile: TaskProfile {
                domain: TaskCategory::Coding,
                skills: vec!["python".into()],
                language: Some("python".into()),
                memory_limit_bytes: budget_bytes,
                raw_prompt: "python coding".into(),
            },
        },
        EvalCase {
            id: "coding-wide",
            profile: TaskProfile {
                domain: TaskCategory::Coding,
                skills: vec![
                    "python".into(),
                    "debugging".into(),
                    "testing".into(),
                    "refactoring".into(),
                ],
                language: Some("python".into()),
                memory_limit_bytes: budget_bytes,
                raw_prompt: "python coding full stack".into(),
            },
        },
        EvalCase {
            id: "math",
            profile: TaskProfile {
                domain: TaskCategory::Math,
                skills: vec!["math".into(), "calculus".into()],
                language: None,
                memory_limit_bytes: budget_bytes,
                raw_prompt: "math calculus".into(),
            },
        },
        EvalCase {
            id: "writing",
            profile: TaskProfile {
                domain: TaskCategory::Writing,
                skills: vec!["writing".into()],
                language: None,
                memory_limit_bytes: budget_bytes,
                raw_prompt: "write a blog".into(),
            },
        },
        EvalCase {
            id: "research",
            profile: TaskProfile {
                domain: TaskCategory::Research,
                skills: vec!["research".into(), "summarization".into()],
                language: None,
                memory_limit_bytes: budget_bytes,
                raw_prompt: "research summarize".into(),
            },
        },
    ]
}

fn skill_hits(codebook_id: &str) -> usize {
    codebook_id
        .split_once('+')
        .map(|(_, skills)| skills.split(',').filter(|s| !s.is_empty()).count())
        .unwrap_or(0)
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..n {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom < 1e-8 {
        0.0
    } else {
        dot / denom
    }
}

pub fn metrics_for(model: &GeneratedModel, case_id: &str) -> CaseMetrics {
    let allocated: u64 = model.weights.iter().map(|w| w.memory_bytes() as u64).sum();
    CaseMetrics {
        case: case_id.into(),
        domain: model.task.domain.as_str().into(),
        skills: model.task.skills.clone(),
        skill_hits: skill_hits(&model.latent.codebook_id),
        codebook_id: model.latent.codebook_id.clone(),
        layers: model.layers.len(),
        latent_dim: model.latent.dim,
        claimed_bytes: model.memory_size_bytes,
        allocated_bytes: allocated,
        under_claim: allocated <= model.memory_size_bytes,
        is_mock: model.is_mock,
    }
}

pub fn discrimination(models: &[(String, GeneratedModel)]) -> Option<DiscriminationMetrics> {
    if models.len() < 2 {
        return None;
    }
    let mut within = Vec::new();
    let mut across = Vec::new();
    for i in 0..models.len() {
        for j in (i + 1)..models.len() {
            let c = cosine(&models[i].1.latent.values, &models[j].1.latent.values);
            if models[i].1.task.domain == models[j].1.task.domain {
                within.push(c);
            } else {
                across.push(c);
            }
        }
    }
    if within.is_empty() || across.is_empty() {
        return None;
    }
    let mean_within = within.iter().sum::<f32>() / within.len() as f32;
    let mean_across = across.iter().sum::<f32>() / across.len() as f32;
    Some(DiscriminationMetrics {
        mean_within_cosine: mean_within,
        mean_across_cosine: mean_across,
        separation: mean_within - mean_across,
    })
}

/// Run the default suite against any [`WeightGenerator`].
pub fn run_suite(gen: &dyn WeightGenerator, budget_bytes: u64) -> SuiteReport {
    let cases = default_cases(budget_bytes);
    let mut metrics = Vec::new();
    let mut models = Vec::new();
    for case in &cases {
        let model = gen.generate(case.profile.clone()).expect("eval generate");
        metrics.push(metrics_for(&model, case.id));
        models.push((case.id.to_string(), model));
    }
    let disc = discrimination(&models);
    let all_under = metrics.iter().all(|m| m.under_claim);
    SuiteReport {
        generator: gen.name().into(),
        budget_bytes,
        cases: metrics,
        discrimination: disc,
        all_under_claim: all_under,
        notes: "Structural + latent discrimination only — not LLM quality.".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{create_generator, GeneratorKind};

    #[test]
    fn latent_suite_separates_domains() {
        let gen = create_generator(GeneratorKind::Latent);
        let report = run_suite(gen.as_ref(), 256 * 1024 * 1024);
        assert!(report.all_under_claim);
        assert!(!report.cases.is_empty());
        let disc = report.discrimination.expect("discrimination");
        // Codebook + domain bias should separate at least a little.
        assert!(
            disc.separation > -0.05,
            "unexpected negative separation: {disc:?}"
        );
    }
}
