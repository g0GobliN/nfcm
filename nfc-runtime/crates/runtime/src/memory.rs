//! Dynamic memory tracking and soft eviction for the NFCM runtime.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;
use uuid::Uuid;

#[derive(Debug, Error)]
pub enum MemoryError {
    #[error("insufficient memory: need {need_bytes} bytes, available {available_bytes} bytes")]
    Insufficient {
        need_bytes: u64,
        available_bytes: u64,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryBudget {
    pub max_ram_bytes: u64,
    pub generator_reserve_bytes: u64,
    pub cache_reserve_bytes: u64,
}

impl Default for MemoryBudget {
    fn default() -> Self {
        Self {
            max_ram_bytes: 1024 * 1024 * 1024, // 1 GiB default soft limit
            generator_reserve_bytes: 300 * 1024 * 1024,
            cache_reserve_bytes: 100 * 1024 * 1024,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComponentKind {
    Generator,
    ActiveModel,
    Cache,
    Other,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryAllocation {
    pub id: Uuid,
    pub label: String,
    pub kind: ComponentKind,
    pub bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySnapshot {
    pub max_ram_bytes: u64,
    pub used_bytes: u64,
    pub generator_bytes: u64,
    pub active_model_bytes: u64,
    pub cache_bytes: u64,
    pub other_bytes: u64,
    pub allocations: Vec<MemoryAllocation>,
}

impl MemorySnapshot {
    pub fn used_mb(&self) -> u64 {
        self.used_bytes / (1024 * 1024)
    }

    pub fn max_mb(&self) -> u64 {
        self.max_ram_bytes / (1024 * 1024)
    }

    pub fn utilization(&self) -> f64 {
        if self.max_ram_bytes == 0 {
            0.0
        } else {
            self.used_bytes as f64 / self.max_ram_bytes as f64
        }
    }
}

pub struct MemoryManager {
    budget: MemoryBudget,
    allocations: HashMap<Uuid, MemoryAllocation>,
}

impl MemoryManager {
    pub fn new(budget: MemoryBudget) -> Self {
        let mut mgr = Self {
            budget,
            allocations: HashMap::new(),
        };
        // Reserve generator + cache soft slots so the dashboard matches the design example.
        let _ = mgr.allocate("generator-reserve", ComponentKind::Generator, mgr.budget.generator_reserve_bytes);
        let _ = mgr.allocate("cache-reserve", ComponentKind::Cache, mgr.budget.cache_reserve_bytes);
        mgr
    }

    pub fn budget(&self) -> &MemoryBudget {
        &self.budget
    }

    pub fn set_max_ram(&mut self, max_ram_bytes: u64) {
        self.budget.max_ram_bytes = max_ram_bytes;
    }

    pub fn used_bytes(&self) -> u64 {
        self.allocations.values().map(|a| a.bytes).sum()
    }

    pub fn available_bytes(&self) -> u64 {
        self.budget.max_ram_bytes.saturating_sub(self.used_bytes())
    }

    pub fn allocate(
        &mut self,
        label: impl Into<String>,
        kind: ComponentKind,
        bytes: u64,
    ) -> Result<Uuid, MemoryError> {
        if bytes > self.available_bytes() {
            return Err(MemoryError::Insufficient {
                need_bytes: bytes,
                available_bytes: self.available_bytes(),
            });
        }
        let id = Uuid::new_v4();
        self.allocations.insert(
            id,
            MemoryAllocation {
                id,
                label: label.into(),
                kind,
                bytes,
            },
        );
        Ok(id)
    }

    pub fn release(&mut self, id: Uuid) -> bool {
        self.allocations.remove(&id).is_some()
    }

    pub fn release_by_kind(&mut self, kind: ComponentKind) -> u64 {
        let ids: Vec<Uuid> = self
            .allocations
            .iter()
            .filter(|(_, a)| a.kind == kind)
            .map(|(id, _)| *id)
            .collect();
        let mut freed = 0u64;
        for id in ids {
            if let Some(a) = self.allocations.remove(&id) {
                freed += a.bytes;
            }
        }
        freed
    }

    /// Drop unused active-model allocations when over budget (LRU-less Phase 1: drop all but newest).
    pub fn optimize(&mut self) -> u64 {
        if self.used_bytes() <= self.budget.max_ram_bytes {
            return 0;
        }
        let mut models: Vec<MemoryAllocation> = self
            .allocations
            .values()
            .filter(|a| a.kind == ComponentKind::ActiveModel)
            .cloned()
            .collect();
        models.sort_by_key(|a| a.bytes);
        let mut freed = 0u64;
        while self.used_bytes() > self.budget.max_ram_bytes {
            let Some(victim) = models.pop() else {
                break;
            };
            if self.allocations.remove(&victim.id).is_some() {
                freed += victim.bytes;
            }
        }
        freed
    }

    pub fn snapshot(&self) -> MemorySnapshot {
        let mut generator_bytes = 0;
        let mut active_model_bytes = 0;
        let mut cache_bytes = 0;
        let mut other_bytes = 0;
        for a in self.allocations.values() {
            match a.kind {
                ComponentKind::Generator => generator_bytes += a.bytes,
                ComponentKind::ActiveModel => active_model_bytes += a.bytes,
                ComponentKind::Cache => cache_bytes += a.bytes,
                ComponentKind::Other => other_bytes += a.bytes,
            }
        }
        MemorySnapshot {
            max_ram_bytes: self.budget.max_ram_bytes,
            used_bytes: self.used_bytes(),
            generator_bytes,
            active_model_bytes,
            cache_bytes,
            other_bytes,
            allocations: self.allocations.values().cloned().collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocate_and_release() {
        let mut mm = MemoryManager::new(MemoryBudget {
            max_ram_bytes: 1024 * 1024 * 1024,
            generator_reserve_bytes: 100 * 1024 * 1024,
            cache_reserve_bytes: 50 * 1024 * 1024,
        });
        let id = mm
            .allocate("model", ComponentKind::ActiveModel, 200 * 1024 * 1024)
            .unwrap();
        assert!(mm.used_bytes() >= 350 * 1024 * 1024);
        assert!(mm.release(id));
    }

    #[test]
    fn rejects_over_budget() {
        let mut mm = MemoryManager::new(MemoryBudget {
            max_ram_bytes: 100,
            generator_reserve_bytes: 0,
            cache_reserve_bytes: 0,
        });
        // Clear default reserves by using tiny budget with zero reserves
        let err = mm.allocate("big", ComponentKind::ActiveModel, 200).unwrap_err();
        assert!(matches!(err, MemoryError::Insufficient { .. }));
    }
}
