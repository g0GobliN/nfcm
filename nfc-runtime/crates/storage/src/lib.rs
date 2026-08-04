//! Model registry and filesystem/SQLite storage for NFCM.

mod cache;
mod model;
mod registry;

pub use cache::{CacheEntry, CacheManager};
pub use model::{Architecture, Model, ModelStatus, TaskType};
pub use registry::{ModelRegistry, RegistryError};

pub const SCHEMA_VERSION: u32 = 1;
