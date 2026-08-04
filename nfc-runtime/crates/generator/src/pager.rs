//! LRU skill pager — keep only hot codebook residuals resident under a RAM budget.
//!
//! The full [`SkillCodebook`] is the on-disk / cold bank. [`SkillPager`] pages vectors
//! into a hot working set and evicts least-recently-used skills when over budget.
//! Current banks are tiny; this seam is for larger trained codebooks later.

use crate::codebook::SkillCodebook;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

/// Bytes for one f32 residual of `dim` length.
pub fn skill_bytes(dim: usize) -> u64 {
    (dim * std::mem::size_of::<f32>()) as u64
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PagerStats {
    pub codebook_id: String,
    pub bank_skills: usize,
    pub hot_skills: usize,
    pub resident_bytes: u64,
    pub max_resident_bytes: u64,
    pub page_ins: u64,
    pub evictions: u64,
}

pub struct SkillPager {
    bank: SkillCodebook,
    hot: HashMap<String, Vec<f32>>,
    lru: VecDeque<String>,
    max_resident_bytes: u64,
    page_ins: u64,
    evictions: u64,
}

impl SkillPager {
    pub fn new(bank: SkillCodebook, max_resident_bytes: u64) -> Self {
        let min_one = skill_bytes(bank.dim).max(1);
        Self {
            max_resident_bytes: max_resident_bytes.max(min_one),
            bank,
            hot: HashMap::new(),
            lru: VecDeque::new(),
            page_ins: 0,
            evictions: 0,
        }
    }

    /// Default budget: `NFCM_CODEBOOK_RAM_BYTES` or enough for 4 skills.
    pub fn from_bank(bank: SkillCodebook) -> Self {
        let default = skill_bytes(bank.dim).saturating_mul(4);
        let max = std::env::var("NFCM_CODEBOOK_RAM_BYTES")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(default);
        Self::new(bank, max)
    }

    pub fn bank(&self) -> &SkillCodebook {
        &self.bank
    }

    pub fn max_resident_bytes(&self) -> u64 {
        self.max_resident_bytes
    }

    pub fn resident_bytes(&self) -> u64 {
        self.hot.len() as u64 * skill_bytes(self.bank.dim)
    }

    pub fn hot_skills(&self) -> Vec<String> {
        self.lru.iter().cloned().collect()
    }

    pub fn stats(&self) -> PagerStats {
        PagerStats {
            codebook_id: self.bank.id.clone(),
            bank_skills: self.bank.len(),
            hot_skills: self.hot.len(),
            resident_bytes: self.resident_bytes(),
            max_resident_bytes: self.max_resident_bytes,
            page_ins: self.page_ins,
            evictions: self.evictions,
        }
    }

    fn touch(&mut self, key: &str) {
        self.lru.retain(|k| k != key);
        self.lru.push_back(key.to_string());
    }

    fn page_in(&mut self, key: &str) -> bool {
        if self.hot.contains_key(key) {
            self.touch(key);
            return true;
        }
        let Some(entry) = self.bank.entries.get(key).cloned() else {
            return false;
        };
        // Evict until we can fit one more skill.
        let need = skill_bytes(self.bank.dim);
        while self.resident_bytes() + need > self.max_resident_bytes {
            if !self.evict_one(&[]) {
                break;
            }
        }
        if self.resident_bytes() + need > self.max_resident_bytes && !self.hot.is_empty() {
            // Still over: force evict until empty or fits
            while self.resident_bytes() + need > self.max_resident_bytes {
                if !self.evict_one(&[]) {
                    break;
                }
            }
        }
        self.hot.insert(key.to_string(), entry);
        self.lru.push_back(key.to_string());
        self.page_ins += 1;
        true
    }

    /// Evict one LRU skill not in `protect`. Returns false if nothing to evict.
    fn evict_one(&mut self, protect: &[String]) -> bool {
        let victim = self
            .lru
            .iter()
            .find(|k| !protect.iter().any(|p| p == *k))
            .cloned()
            .or_else(|| self.lru.front().cloned());
        let Some(key) = victim else {
            return false;
        };
        self.lru.retain(|k| k != &key);
        if self.hot.remove(&key).is_some() {
            self.evictions += 1;
            true
        } else {
            false
        }
    }

    /// Drop all hot skills except `keep` (used by memory optimize).
    pub fn evict_cold(&mut self, keep: &[String]) -> u64 {
        let before = self.resident_bytes();
        let drop_keys: Vec<String> = self
            .hot
            .keys()
            .filter(|k| !keep.iter().any(|p| p == *k))
            .cloned()
            .collect();
        for key in drop_keys {
            self.hot.remove(&key);
            self.lru.retain(|k| k != &key);
            self.evictions += 1;
        }
        before.saturating_sub(self.resident_bytes())
    }

    /// Evict until under budget; keep currently requested skills if possible.
    pub fn optimize(&mut self) -> u64 {
        let before = self.resident_bytes();
        while self.resident_bytes() > self.max_resident_bytes {
            if !self.evict_one(&[]) {
                break;
            }
        }
        before.saturating_sub(self.resident_bytes())
    }

    /// Page in requested skills, blend residuals (same contract as [`SkillCodebook::activate`]).
    pub fn activate(&mut self, skills: &[String]) -> (Vec<String>, Vec<f32>) {
        let mut hit = Vec::new();
        let mut requested_keys = Vec::new();
        for skill in skills {
            let key = skill.trim().to_ascii_lowercase();
            if key.is_empty() {
                continue;
            }
            if self.page_in(&key) {
                hit.push(key.clone());
                requested_keys.push(key);
            }
        }
        // Prefer keeping this request's skills hot if we still overflow.
        while self.resident_bytes() > self.max_resident_bytes {
            if !self.evict_one(&requested_keys) {
                break;
            }
        }

        let mut residual = vec![0.0f32; self.bank.dim];
        for key in &hit {
            if let Some(entry) = self.hot.get(key) {
                for (i, v) in residual.iter_mut().enumerate() {
                    *v += entry.get(i).copied().unwrap_or(0.0);
                }
            }
        }
        let scale = 0.22f32;
        for v in &mut residual {
            *v *= scale;
        }
        (hit, residual)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pages_and_evicts_under_budget() {
        let bank = SkillCodebook::builtin();
        // Budget for exactly 2 skills.
        let mut pager = SkillPager::new(bank.clone(), skill_bytes(bank.dim) * 2);
        let (hit, _) = pager.activate(&["python".into(), "debugging".into()]);
        assert_eq!(hit.len(), 2);
        assert_eq!(pager.hot.len(), 2);

        let (hit2, _) = pager.activate(&["rust".into(), "testing".into()]);
        assert_eq!(hit2.len(), 2);
        assert!(pager.hot.len() <= 2);
        assert!(pager.stats().evictions >= 1);
        assert!(pager.hot.contains_key("rust") || pager.hot.contains_key("testing"));
    }

    #[test]
    fn evict_cold_frees_bytes() {
        let bank = SkillCodebook::builtin();
        let mut pager = SkillPager::new(bank.clone(), skill_bytes(bank.dim) * 8);
        pager.activate(&["python".into(), "rust".into(), "math".into()]);
        let freed = pager.evict_cold(&["python".into()]);
        assert!(freed > 0);
        assert_eq!(pager.hot.len(), 1);
        assert!(pager.hot.contains_key("python"));
    }
}
