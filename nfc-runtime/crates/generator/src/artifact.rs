//! On-disk brain artifacts (full [`GeneratedModel`] including tensors).

use crate::types::GeneratedModel;
use std::path::{Path, PathBuf};
use thiserror::Error;
use uuid::Uuid;

#[derive(Debug, Error)]
pub enum ArtifactError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("invalid artifact: {0}")]
    Invalid(String),
}

/// Preferred filename for a persisted NFCM brain.
pub fn artifact_filename(id: Uuid) -> String {
    format!("{id}.nfcm.json")
}

pub fn artifact_path(models_dir: impl AsRef<Path>, id: Uuid) -> PathBuf {
    models_dir.as_ref().join(artifact_filename(id))
}

/// Write the full generated model (weights included).
pub fn save_generated_model(
    models_dir: impl AsRef<Path>,
    generated: &GeneratedModel,
) -> Result<PathBuf, ArtifactError> {
    let path = artifact_path(models_dir, generated.id);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let bytes = serde_json::to_vec_pretty(generated)?;
    std::fs::write(&path, bytes)?;
    Ok(path)
}

/// Load a full generated model from disk.
pub fn load_generated_model(path: impl AsRef<Path>) -> Result<GeneratedModel, ArtifactError> {
    let bytes = std::fs::read(path.as_ref())?;
    let model: GeneratedModel = serde_json::from_slice(&bytes)?;
    if model.weights.is_empty() && !model.is_mock {
        return Err(ArtifactError::Invalid(
            "artifact has is_mock=false but empty weights".into(),
        ));
    }
    Ok(model)
}

/// Try path as full artifact; also accept legacy metadata-only `.nfcm-mock.json`.
pub fn load_brain_artifact(path: impl AsRef<Path>) -> Result<GeneratedModel, ArtifactError> {
    let path = path.as_ref();
    match load_generated_model(path) {
        Ok(m) => Ok(m),
        Err(e) => {
            // Legacy metadata JSON without weights — surface clearly.
            if path
                .extension()
                .and_then(|s| s.to_str())
                .is_some_and(|ext| ext == "json")
            {
                Err(ArtifactError::Invalid(format!(
                    "could not load brain artifact at {}: {e} \
                     (legacy metadata-only files need recompile)",
                    path.display()
                )))
            } else {
                Err(e)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LatentWeightGenerator, TaskCategory, TaskProfile, WeightGenerator};
    use tempfile::tempdir;

    #[test]
    fn roundtrip_latent_brain() {
        let gen = LatentWeightGenerator::new();
        let task = TaskProfile {
            domain: TaskCategory::Coding,
            skills: vec!["rust".into()],
            language: Some("rust".into()),
            memory_limit_bytes: 128 * 1024 * 1024,
            raw_prompt: "rust".into(),
        };
        let model = gen.generate(task).unwrap();
        let dir = tempdir().unwrap();
        let path = save_generated_model(dir.path(), &model).unwrap();
        let loaded = load_generated_model(&path).unwrap();
        assert_eq!(loaded.id, model.id);
        assert_eq!(loaded.weights.len(), model.weights.len());
        assert_eq!(loaded.weights[0].data, model.weights[0].data);
        assert_eq!(loaded.latent.values, model.latent.values);
    }
}
