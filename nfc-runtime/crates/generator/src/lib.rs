//! Neural weight generation abstractions for NFCM.
//!
//! Phase 1 ships a **mock** generator that produces deterministic placeholder
//! weights. The `WeightGenerator` trait is the extension point for a future
//! hypernetwork / latent foundation compressor.

mod compiler;
mod mock;
mod types;

pub use compiler::TaskCompiler;
pub use mock::MockWeightGenerator;
pub use types::{
    GeneratedModel, GenerationProgress, LatentCode, LayerSpec, OptimizationProfile, TaskCategory,
    TaskProfile, WeightGenerator, WeightGeneratorError,
};
