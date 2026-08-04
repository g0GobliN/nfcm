//! Filesystem-backed cache for generated artifacts.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use thiserror::Error;
use tracing::debug;
use uuid::Uuid;

#[derive(Debug, Error)]
pub enum CacheError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("cache entry not found: {0}")]
    NotFound(String),
    #[error("serialization error: {0}")]
    Serde(#[from] serde_json::Error),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    pub key: String,
    pub model_id: Option<Uuid>,
    pub path: PathBuf,
    pub size_bytes: u64,
    pub last_accessed: DateTime<Utc>,
}

pub struct CacheManager {
    root: PathBuf,
    max_bytes: u64,
}

impl CacheManager {
    pub fn open(root: impl AsRef<Path>, max_bytes: u64) -> Result<Self, CacheError> {
        let root = root.as_ref().to_path_buf();
        fs::create_dir_all(&root)?;
        Ok(Self { root, max_bytes })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn max_bytes(&self) -> u64 {
        self.max_bytes
    }

    pub fn put(
        &self,
        key: &str,
        bytes: &[u8],
        model_id: Option<Uuid>,
    ) -> Result<CacheEntry, CacheError> {
        let path = self.root.join(sanitize_key(key));
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&path, bytes)?;
        let entry = CacheEntry {
            key: key.to_string(),
            model_id,
            path,
            size_bytes: bytes.len() as u64,
            last_accessed: Utc::now(),
        };
        let meta_path = self.meta_path(key);
        fs::write(meta_path, serde_json::to_vec_pretty(&entry)?)?;
        self.evict_if_needed()?;
        Ok(entry)
    }

    pub fn get(&self, key: &str) -> Result<Vec<u8>, CacheError> {
        let path = self.root.join(sanitize_key(key));
        let bytes = fs::read(&path).map_err(|_| CacheError::NotFound(key.to_string()))?;
        if let Ok(mut entry) = self.load_meta(key) {
            entry.last_accessed = Utc::now();
            let _ = fs::write(self.meta_path(key), serde_json::to_vec_pretty(&entry)?);
        }
        Ok(bytes)
    }

    pub fn remove(&self, key: &str) -> Result<(), CacheError> {
        let path = self.root.join(sanitize_key(key));
        let meta = self.meta_path(key);
        let _ = fs::remove_file(path);
        let _ = fs::remove_file(meta);
        Ok(())
    }

    pub fn usage_bytes(&self) -> u64 {
        walk_size(&self.root)
    }

    fn meta_path(&self, key: &str) -> PathBuf {
        self.root.join(format!("{}.meta.json", sanitize_key(key)))
    }

    fn load_meta(&self, key: &str) -> Result<CacheEntry, CacheError> {
        let raw = fs::read(self.meta_path(key))?;
        Ok(serde_json::from_slice(&raw)?)
    }

    fn evict_if_needed(&self) -> Result<(), CacheError> {
        let mut usage = self.usage_bytes();
        if usage <= self.max_bytes {
            return Ok(());
        }
        debug!(
            usage,
            max = self.max_bytes,
            "cache over budget; eviction needed"
        );
        // Phase 1: delete oldest meta-tracked files until under budget.
        let mut metas: Vec<CacheEntry> = fs::read_dir(&self.root)?
            .flatten()
            .filter_map(|e| {
                let p = e.path();
                if p.extension().and_then(|x| x.to_str()) == Some("json")
                    && p.file_name()?.to_str()?.ends_with(".meta.json")
                {
                    fs::read(&p)
                        .ok()
                        .and_then(|b| serde_json::from_slice(&b).ok())
                } else {
                    None
                }
            })
            .collect();
        metas.sort_by_key(|m| m.last_accessed);
        for entry in metas {
            if usage <= self.max_bytes {
                break;
            }
            usage = usage.saturating_sub(entry.size_bytes);
            let _ = self.remove(&entry.key);
        }
        Ok(())
    }
}

fn sanitize_key(key: &str) -> String {
    key.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

fn walk_size(path: &Path) -> u64 {
    let mut total = 0u64;
    let Ok(entries) = fs::read_dir(path) else {
        return 0;
    };
    for entry in entries.flatten() {
        let p = entry.path();
        if p.is_dir() {
            total += walk_size(&p);
        } else if let Ok(meta) = entry.metadata() {
            total += meta.len();
        }
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn put_get_roundtrip() {
        let dir = tempdir().unwrap();
        let cache = CacheManager::open(dir.path(), 10_000_000).unwrap();
        cache.put("hello", b"world", None).unwrap();
        assert_eq!(cache.get("hello").unwrap(), b"world");
    }
}
