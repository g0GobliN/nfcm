//! Local hardware detection for NFCM runtime scheduling and memory budgets.

use serde::{Deserialize, Serialize};
use sysinfo::System;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum HardwareError {
    #[error("failed to read system information")]
    SysInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuInfo {
    pub brand: String,
    pub cores: usize,
    pub frequency_mhz: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RamInfo {
    /// Total physical memory in bytes.
    pub total_bytes: u64,
    /// Available memory in bytes.
    pub available_bytes: u64,
    /// Used memory in bytes.
    pub used_bytes: u64,
}

impl RamInfo {
    pub fn total_mb(&self) -> u64 {
        self.total_bytes / (1024 * 1024)
    }

    pub fn available_mb(&self) -> u64 {
        self.available_bytes / (1024 * 1024)
    }

    pub fn used_mb(&self) -> u64 {
        self.used_bytes / (1024 * 1024)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    pub name: String,
    pub vendor: String,
    /// Estimated VRAM in bytes when known; 0 if unknown / CPU-only.
    pub vram_bytes: u64,
    pub available: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareProfile {
    pub cpu: CpuInfo,
    pub ram: RamInfo,
    pub gpus: Vec<GpuInfo>,
    pub hostname: String,
}

/// Detects CPU, RAM, and best-effort GPU presence.
///
/// GPU detection is intentionally conservative in Phase 1: we report a
/// placeholder entry when no discrete GPU metadata is available rather than
/// claiming CUDA/ROCm capabilities we have not verified.
pub struct HardwareDetector;

impl HardwareDetector {
    pub fn detect() -> Result<HardwareProfile, HardwareError> {
        let mut sys = System::new_all();
        sys.refresh_all();

        let cpu = Self::detect_cpu(&sys);
        let ram = Self::detect_ram(&sys);
        let gpus = Self::detect_gpus();
        let hostname = System::host_name().unwrap_or_else(|| "localhost".into());

        Ok(HardwareProfile {
            cpu,
            ram,
            gpus,
            hostname,
        })
    }

    fn detect_cpu(sys: &System) -> CpuInfo {
        let cpus = sys.cpus();
        let brand = cpus
            .first()
            .map(|c| c.brand().trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "Unknown CPU".into());
        let frequency_mhz = cpus.first().map(|c| c.frequency()).unwrap_or(0);
        CpuInfo {
            brand,
            cores: sys.physical_core_count().unwrap_or(cpus.len()).max(1),
            frequency_mhz,
        }
    }

    fn detect_ram(sys: &System) -> RamInfo {
        RamInfo {
            total_bytes: sys.total_memory(),
            available_bytes: sys.available_memory(),
            used_bytes: sys.used_memory(),
        }
    }

    fn detect_gpus() -> Vec<GpuInfo> {
        // Phase 1: probe common Linux sysfs path; fall back to CPU-only notice.
        if let Ok(entries) = std::fs::read_dir("/sys/class/drm") {
            let mut gpus = Vec::new();
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if name.starts_with("card") && !name.contains('-') {
                    let vendor_path = entry.path().join("device/vendor");
                    let vendor = std::fs::read_to_string(vendor_path)
                        .unwrap_or_else(|_| "unknown".into())
                        .trim()
                        .to_string();
                    gpus.push(GpuInfo {
                        name: format!("DRM {name}"),
                        vendor: match vendor.as_str() {
                            "0x10de" => "NVIDIA".into(),
                            "0x1002" => "AMD".into(),
                            "0x8086" => "Intel".into(),
                            other => other.to_string(),
                        },
                        vram_bytes: 0,
                        available: true,
                    });
                }
            }
            if !gpus.is_empty() {
                return gpus;
            }
        }

        vec![GpuInfo {
            name: "CPU fallback".into(),
            vendor: "none".into(),
            vram_bytes: 0,
            available: false,
        }]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_returns_cpu_and_ram() {
        let profile = HardwareDetector::detect().expect("hardware detect");
        assert!(profile.cpu.cores >= 1);
        assert!(profile.ram.total_bytes > 0);
        assert!(!profile.gpus.is_empty());
    }
}
