//! Core types for the neural generator interface.

use nfc_tensor::{DType, Tensor, TensorShape};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

#[derive(Debug, Error)]
pub enum WeightGeneratorError {
    #[error("unsupported task profile: {0}")]
    UnsupportedTask(String),
    #[error("memory limit too low: {limit_bytes} bytes")]
    MemoryLimitTooLow { limit_bytes: u64 },
    #[error("generation failed: {0}")]
    Failed(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskCategory {
    Coding,
    Math,
    Writing,
    Research,
    Medical,
    Custom,
}

impl TaskCategory {
    pub fn as_str(self) -> &'static str {
        match self {
            TaskCategory::Coding => "coding",
            TaskCategory::Math => "math",
            TaskCategory::Writing => "writing",
            TaskCategory::Research => "research",
            TaskCategory::Medical => "medical",
            TaskCategory::Custom => "custom",
        }
    }
}

/// Compiled task description that drives weight generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskProfile {
    pub domain: TaskCategory,
    pub skills: Vec<String>,
    pub language: Option<String>,
    pub memory_limit_bytes: u64,
    pub raw_prompt: String,
}

impl TaskProfile {
    pub fn memory_limit_mb(&self) -> u64 {
        self.memory_limit_bytes / (1024 * 1024)
    }
}

/// Compact latent handle — placeholder for future compressed knowledge codes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatentCode {
    pub dim: usize,
    pub values: Vec<f32>,
    pub codebook_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationProfile {
    pub quantize: bool,
    pub target_memory_bytes: u64,
    pub activation_sparsity: f32,
    pub notes: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerSpec {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: DType,
}

/// Output of a weight generator.
///
/// `is_mock: true` means metadata-only placeholder.
/// `is_mock: false` may still be an **untrained** research prototype — check
/// `optimization_profile.notes` before treating output as a production model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedModel {
    pub id: Uuid,
    pub name: String,
    pub task: TaskProfile,
    pub layers: Vec<LayerSpec>,
    pub weights: Vec<Tensor>,
    pub latent: LatentCode,
    pub memory_size_bytes: u64,
    pub optimization_profile: OptimizationProfile,
    pub is_mock: bool,
}

impl GeneratedModel {
    pub fn memory_size_mb(&self) -> u64 {
        self.memory_size_bytes / (1024 * 1024)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GenerationProgress {
    AnalyzingTask,
    SelectingComponents,
    GeneratingModel,
    OptimizingMemory,
    Complete,
}

impl GenerationProgress {
    pub fn label(&self) -> &'static str {
        match self {
            GenerationProgress::AnalyzingTask => "Analyzing task",
            GenerationProgress::SelectingComponents => "Selecting components",
            GenerationProgress::GeneratingModel => "Generating model",
            GenerationProgress::OptimizingMemory => "Optimizing memory",
            GenerationProgress::Complete => "Complete",
        }
    }
}

/// Extension point for real hypernetwork weight generation.
pub trait WeightGenerator: Send + Sync {
    fn name(&self) -> &str;

    fn generate(&self, task: TaskProfile) -> Result<GeneratedModel, WeightGeneratorError>;
}

/// Shared progress wrapper for any [`WeightGenerator`].
pub fn generate_with_progress<G, F>(
    generator: &G,
    task: TaskProfile,
    mut on_progress: F,
) -> Result<GeneratedModel, WeightGeneratorError>
where
    G: WeightGenerator + ?Sized,
    F: FnMut(GenerationProgress),
{
    on_progress(GenerationProgress::AnalyzingTask);
    on_progress(GenerationProgress::SelectingComponents);
    on_progress(GenerationProgress::GeneratingModel);
    let model = generator.generate(task)?;
    on_progress(GenerationProgress::OptimizingMemory);
    on_progress(GenerationProgress::Complete);
    Ok(model)
}

/// Helper to build a zero-filled layer tensor from a shape.
#[allow(dead_code)]
pub fn zero_layer(name: &str, dims: &[usize], dtype: DType) -> (LayerSpec, Tensor) {
    let shape = TensorShape::new(dims.to_vec());
    let tensor = Tensor::zeros(shape, dtype);
    (
        LayerSpec {
            name: name.to_string(),
            shape: dims.to_vec(),
            dtype,
        },
        tensor,
    )
}
