//! Model metadata — a model is more than a weight file.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskType {
    Coding,
    Math,
    Writing,
    Research,
    Medical,
    Custom,
}

impl TaskType {
    pub fn as_str(self) -> &'static str {
        match self {
            TaskType::Coding => "coding",
            TaskType::Math => "math",
            TaskType::Writing => "writing",
            TaskType::Research => "research",
            TaskType::Medical => "medical",
            TaskType::Custom => "custom",
        }
    }
}

impl std::fmt::Display for TaskType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Architecture {
    /// Placeholder for future hypernetwork-generated subnetworks.
    NfcmSubnetwork,
    /// Compatibility stubs for external runtimes (not wired in Phase 1).
    Onnx,
    Gguf,
    Candle,
    Mock,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelStatus {
    Registered,
    Generating,
    Ready,
    Loaded,
    Unloaded,
    Failed,
}

/// First-class model descriptor used across registry, generator, and runtime.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model {
    pub id: Uuid,
    pub name: String,
    /// Approximate on-disk / in-memory size in bytes.
    pub size_bytes: u64,
    pub architecture: Architecture,
    pub task_type: TaskType,
    /// Soft memory budget required to load this model (bytes).
    pub memory_requirement_bytes: u64,
    pub status: ModelStatus,
    pub skills: Vec<String>,
    pub description: String,
    pub path: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl Model {
    pub fn new(
        name: impl Into<String>,
        task_type: TaskType,
        architecture: Architecture,
        memory_requirement_bytes: u64,
    ) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            name: name.into(),
            size_bytes: memory_requirement_bytes,
            architecture,
            task_type,
            memory_requirement_bytes,
            status: ModelStatus::Registered,
            skills: Vec::new(),
            description: String::new(),
            path: None,
            created_at: now,
            updated_at: now,
        }
    }

    pub fn with_skills(mut self, skills: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.skills = skills.into_iter().map(Into::into).collect();
        self
    }

    pub fn memory_requirement_mb(&self) -> u64 {
        self.memory_requirement_bytes / (1024 * 1024)
    }
}
