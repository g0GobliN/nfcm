//! SQLite-backed model registry.

use crate::model::{Architecture, Model, ModelStatus, TaskType};
use chrono::{DateTime, Utc};
use rusqlite::{params, Connection};
use std::path::{Path, PathBuf};
use thiserror::Error;
use tracing::info;
use uuid::Uuid;

#[derive(Debug, Error)]
pub enum RegistryError {
    #[error("sqlite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("model not found: {0}")]
    NotFound(Uuid),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("serialization error: {0}")]
    Serde(#[from] serde_json::Error),
}

pub struct ModelRegistry {
    db_path: PathBuf,
    models_dir: PathBuf,
}

impl ModelRegistry {
    pub fn open(data_dir: impl AsRef<Path>) -> Result<Self, RegistryError> {
        let data_dir = data_dir.as_ref();
        std::fs::create_dir_all(data_dir)?;
        let models_dir = data_dir.join("models");
        std::fs::create_dir_all(&models_dir)?;
        let db_path = data_dir.join("registry.sqlite");
        let registry = Self {
            db_path,
            models_dir,
        };
        registry.init_schema()?;
        Ok(registry)
    }

    pub fn models_dir(&self) -> &Path {
        &self.models_dir
    }

    fn conn(&self) -> Result<Connection, RegistryError> {
        Ok(Connection::open(&self.db_path)?)
    }

    fn init_schema(&self) -> Result<(), RegistryError> {
        let conn = self.conn()?;
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS models (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                architecture TEXT NOT NULL,
                task_type TEXT NOT NULL,
                memory_requirement_bytes INTEGER NOT NULL,
                status TEXT NOT NULL,
                skills_json TEXT NOT NULL,
                description TEXT NOT NULL,
                path TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            "#,
        )?;
        Ok(())
    }

    pub fn upsert(&self, model: &Model) -> Result<(), RegistryError> {
        let conn = self.conn()?;
        conn.execute(
            r#"
            INSERT INTO models (
                id, name, size_bytes, architecture, task_type,
                memory_requirement_bytes, status, skills_json, description,
                path, created_at, updated_at
            ) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12)
            ON CONFLICT(id) DO UPDATE SET
                name=excluded.name,
                size_bytes=excluded.size_bytes,
                architecture=excluded.architecture,
                task_type=excluded.task_type,
                memory_requirement_bytes=excluded.memory_requirement_bytes,
                status=excluded.status,
                skills_json=excluded.skills_json,
                description=excluded.description,
                path=excluded.path,
                updated_at=excluded.updated_at
            "#,
            params![
                model.id.to_string(),
                model.name,
                model.size_bytes as i64,
                serde_json::to_string(&model.architecture)?,
                serde_json::to_string(&model.task_type)?,
                model.memory_requirement_bytes as i64,
                serde_json::to_string(&model.status)?,
                serde_json::to_string(&model.skills)?,
                model.description,
                model.path,
                model.created_at.to_rfc3339(),
                model.updated_at.to_rfc3339(),
            ],
        )?;
        info!(id = %model.id, name = %model.name, "model upserted");
        Ok(())
    }

    pub fn get(&self, id: Uuid) -> Result<Model, RegistryError> {
        let conn = self.conn()?;
        let mut stmt = conn.prepare(
            "SELECT id, name, size_bytes, architecture, task_type, memory_requirement_bytes,
                    status, skills_json, description, path, created_at, updated_at
             FROM models WHERE id = ?1",
        )?;
        let model = stmt
            .query_row(params![id.to_string()], row_to_model)
            .map_err(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => RegistryError::NotFound(id),
                other => RegistryError::Sqlite(other),
            })?;
        Ok(model)
    }

    pub fn list(&self) -> Result<Vec<Model>, RegistryError> {
        let conn = self.conn()?;
        let mut stmt = conn.prepare(
            "SELECT id, name, size_bytes, architecture, task_type, memory_requirement_bytes,
                    status, skills_json, description, path, created_at, updated_at
             FROM models ORDER BY updated_at DESC",
        )?;
        let rows = stmt.query_map([], row_to_model)?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    pub fn delete(&self, id: Uuid) -> Result<(), RegistryError> {
        let model = self.get(id)?;
        if let Some(path) = &model.path {
            let p = PathBuf::from(path);
            if p.exists() {
                let _ = std::fs::remove_file(&p);
            }
        }
        let conn = self.conn()?;
        let n = conn.execute("DELETE FROM models WHERE id = ?1", params![id.to_string()])?;
        if n == 0 {
            return Err(RegistryError::NotFound(id));
        }
        Ok(())
    }

    pub fn update_status(&self, id: Uuid, status: ModelStatus) -> Result<Model, RegistryError> {
        let mut model = self.get(id)?;
        model.status = status;
        model.updated_at = Utc::now();
        self.upsert(&model)?;
        Ok(model)
    }
}

fn row_to_model(row: &rusqlite::Row<'_>) -> rusqlite::Result<Model> {
    let id: String = row.get(0)?;
    let architecture: String = row.get(3)?;
    let task_type: String = row.get(4)?;
    let status: String = row.get(6)?;
    let skills_json: String = row.get(7)?;
    let created_at: String = row.get(10)?;
    let updated_at: String = row.get(11)?;

    Ok(Model {
        id: Uuid::parse_str(&id).map_err(|e| {
            rusqlite::Error::FromSqlConversionFailure(0, rusqlite::types::Type::Text, Box::new(e))
        })?,
        name: row.get(1)?,
        size_bytes: row.get::<_, i64>(2)? as u64,
        architecture: serde_json::from_str(&architecture).unwrap_or(Architecture::Mock),
        task_type: serde_json::from_str(&task_type).unwrap_or(TaskType::Custom),
        memory_requirement_bytes: row.get::<_, i64>(5)? as u64,
        status: serde_json::from_str(&status).unwrap_or(ModelStatus::Registered),
        skills: serde_json::from_str(&skills_json).unwrap_or_default(),
        description: row.get(8)?,
        path: row.get(9)?,
        created_at: DateTime::parse_from_rfc3339(&created_at)
            .map(|d| d.with_timezone(&Utc))
            .unwrap_or_else(|_| Utc::now()),
        updated_at: DateTime::parse_from_rfc3339(&updated_at)
            .map(|d| d.with_timezone(&Utc))
            .unwrap_or_else(|_| Utc::now()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn upsert_list_delete() {
        let dir = tempdir().unwrap();
        let reg = ModelRegistry::open(dir.path()).unwrap();
        let model = Model::new(
            "Python Coder",
            TaskType::Coding,
            Architecture::Mock,
            64 * 1024 * 1024,
        )
        .with_skills(["python", "debugging"]);
        let id = model.id;
        reg.upsert(&model).unwrap();
        assert_eq!(reg.list().unwrap().len(), 1);
        assert_eq!(reg.get(id).unwrap().name, "Python Coder");
        reg.delete(id).unwrap();
        assert!(reg.list().unwrap().is_empty());
    }
}
