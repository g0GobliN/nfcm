//! Neural weight generation abstractions for NFCM.
//!
//! - Phase 1: [`MockWeightGenerator`] — metadata-only placeholder
//! - Phase 2: [`LatentWeightGenerator`] — latent → tiny real tensors
//!   (untrained stub, or trained toy checkpoint via `NFCM_HYPERNET_CHECKPOINT`)
//!
//! Select via `NFCM_WEIGHT_GENERATOR=mock|latent` or [`create_generator`].

mod artifact;
mod checkpoint;
mod compiler;
mod latent;
mod mock;
mod types;

pub use artifact::{
    artifact_filename, artifact_path, load_brain_artifact, load_generated_model,
    save_generated_model, ArtifactError,
};
pub use checkpoint::{resolve_checkpoint_path, CheckpointError, HyperCheckpoint};
pub use compiler::TaskCompiler;
pub use latent::LatentWeightGenerator;
pub use mock::MockWeightGenerator;
pub use types::{
    generate_with_progress, GeneratedModel, GenerationProgress, LatentCode, LayerSpec,
    OptimizationProfile, TaskCategory, TaskProfile, WeightGenerator, WeightGeneratorError,
};

use serde::{Deserialize, Serialize};

/// Which weight generator the runtime starts with.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum GeneratorKind {
    #[default]
    Mock,
    Latent,
}

impl GeneratorKind {
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "mock" => Some(Self::Mock),
            "latent" | "latent-proto" | "proto" | "hypernet" => Some(Self::Latent),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Mock => "mock",
            Self::Latent => "latent",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratorInfo {
    pub kind: GeneratorKind,
    pub name: String,
    pub is_mock: bool,
    pub notes: String,
}

pub fn create_generator(kind: GeneratorKind) -> Box<dyn WeightGenerator> {
    match kind {
        GeneratorKind::Mock => Box::new(MockWeightGenerator::new()),
        GeneratorKind::Latent => Box::new(LatentWeightGenerator::from_env_checkpoint()),
    }
}

pub fn generator_kind_from_env() -> GeneratorKind {
    std::env::var("NFCM_WEIGHT_GENERATOR")
        .ok()
        .and_then(|s| GeneratorKind::parse(&s))
        .unwrap_or_default()
}

pub fn generator_info(kind: GeneratorKind, gen: &dyn WeightGenerator) -> GeneratorInfo {
    let notes = match kind {
        GeneratorKind::Mock => {
            "Metadata-only mock — no real tensors. Swap with NFCM_WEIGHT_GENERATOR=latent.".into()
        }
        GeneratorKind::Latent => {
            if gen.name().contains("hypernet") {
                "Trained toy hypernetwork checkpoint decode — not an LLM.".into()
            } else {
                "Phase 2 latent-proto — deterministic untrained hypernetwork stub with tiny tensors."
                    .into()
            }
        }
    };
    GeneratorInfo {
        kind,
        name: gen.name().to_string(),
        is_mock: matches!(kind, GeneratorKind::Mock),
        notes,
    }
}
