//! NFCM local AI runtime engine.
//!
//! Orchestrates hardware detection, model registry, mock weight generation,
//! memory budgeting, and a placeholder inference path.

mod engine;
mod inference;
mod memory;
mod scheduler;

pub use engine::{RuntimeConfig, RuntimeEngine, RuntimeError, RuntimeSnapshot, RuntimeStatus};
pub use inference::{InferenceRequest, InferenceResponse};
pub use memory::{MemoryBudget, MemoryManager, MemorySnapshot};
pub use scheduler::{JobId, Scheduler, SchedulerJob, SchedulerState};

pub use nfc_generator::{
    GeneratedModel, GenerationProgress, MockWeightGenerator, TaskCategory, TaskCompiler,
    TaskProfile, WeightGenerator,
};
pub use nfc_hardware::{HardwareDetector, HardwareProfile};
pub use nfc_storage::{
    Architecture, CacheManager, Model, ModelRegistry, ModelStatus, TaskType,
};
