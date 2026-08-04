//! RuntimeEngine — load, unload, infer, optimize.

use crate::inference::{
    backend_kind_from_env, create_backend, BackendInfo, BackendKind, InferenceBackend,
    InferenceRequest, InferenceResponse, ModelContext,
};
use crate::memory::{ComponentKind, MemoryBudget, MemoryManager, MemorySnapshot};
use crate::scheduler::{Scheduler, SchedulerJob};
use nfc_generator::{
    GeneratedModel, GenerationProgress, MockWeightGenerator, TaskCategory, TaskCompiler,
    TaskProfile, WeightGenerator,
};
use nfc_hardware::{HardwareDetector, HardwareProfile};
use nfc_storage::{Architecture, CacheManager, Model, ModelRegistry, ModelStatus, TaskType};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use thiserror::Error;
use tracing::{info, warn};
use uuid::Uuid;

#[derive(Debug, Error)]
pub enum RuntimeError {
    #[error("hardware error: {0}")]
    Hardware(String),
    #[error("storage error: {0}")]
    Storage(String),
    #[error("generator error: {0}")]
    Generator(String),
    #[error("memory error: {0}")]
    Memory(String),
    #[error("inference error: {0}")]
    Inference(String),
    #[error("no model loaded")]
    NoModelLoaded,
    #[error("model not found: {0}")]
    ModelNotFound(Uuid),
    #[error("invalid request: {0}")]
    Invalid(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeStatus {
    Stopped,
    Starting,
    Running,
    Compiling,
    Error,
}

#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    pub data_dir: PathBuf,
    pub memory_budget: MemoryBudget,
    pub backend_kind: BackendKind,
}

impl RuntimeConfig {
    pub fn new(data_dir: impl Into<PathBuf>) -> Self {
        Self {
            data_dir: data_dir.into(),
            memory_budget: MemoryBudget::default(),
            backend_kind: backend_kind_from_env(),
        }
    }

    pub fn with_backend(mut self, kind: BackendKind) -> Self {
        self.backend_kind = kind;
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeSnapshot {
    pub status: RuntimeStatus,
    pub hardware: HardwareProfile,
    pub memory: MemorySnapshot,
    pub active_model: Option<Model>,
    pub models: Vec<Model>,
    pub logs: Vec<String>,
    pub console_lines: Vec<String>,
    pub inference_backend: BackendInfo,
}

struct Inner {
    status: RuntimeStatus,
    registry: ModelRegistry,
    cache: CacheManager,
    memory: MemoryManager,
    scheduler: Scheduler,
    hardware: HardwareProfile,
    active_model: Option<Model>,
    active_generated: Option<GeneratedModel>,
    active_alloc: Option<Uuid>,
    logs: Vec<String>,
    console_lines: Vec<String>,
    compiler: TaskCompiler,
    generator: MockWeightGenerator,
    backend: Box<dyn InferenceBackend>,
    backend_kind: BackendKind,
}

/// Central orchestrator for the local NFCM runtime.
pub struct RuntimeEngine {
    inner: Arc<Mutex<Inner>>,
}

impl RuntimeEngine {
    pub fn start(config: RuntimeConfig) -> Result<Self, RuntimeError> {
        std::fs::create_dir_all(&config.data_dir)
            .map_err(|e| RuntimeError::Storage(format!("create data dir: {e}")))?;

        let registry = ModelRegistry::open(config.data_dir.join("registry"))
            .map_err(|e| RuntimeError::Storage(e.to_string()))?;
        let cache = CacheManager::open(
            config.data_dir.join("cache"),
            config.memory_budget.cache_reserve_bytes,
        )
        .map_err(|e| RuntimeError::Storage(e.to_string()))?;

        let hardware =
            HardwareDetector::detect().map_err(|e| RuntimeError::Hardware(e.to_string()))?;

        let mut memory = MemoryManager::new(config.memory_budget);
        let suggested = hardware.ram.available_bytes / 2;
        if suggested > 0 && suggested < memory.budget().max_ram_bytes {
            memory.set_max_ram(suggested.max(256 * 1024 * 1024));
        }

        let backend = create_backend(config.backend_kind);
        let mut inner = Inner {
            status: RuntimeStatus::Running,
            registry,
            cache,
            memory,
            scheduler: Scheduler::new(),
            hardware,
            active_model: None,
            active_generated: None,
            active_alloc: None,
            logs: Vec::new(),
            console_lines: Vec::new(),
            compiler: TaskCompiler::default(),
            generator: MockWeightGenerator::new(),
            backend,
            backend_kind: config.backend_kind,
        };
        inner.log(format!(
            "NFCM runtime started (backend={} — Phase 2 inference seam)",
            config.backend_kind.as_str()
        ));
        inner.console("> runtime online");
        inner.console(format!(
            "Inference backend: {} ({})",
            inner.backend.name(),
            if inner.backend.is_mock() {
                "mock"
            } else {
                "pluggable"
            }
        ));
        inner.console("Ready.");

        Ok(Self {
            inner: Arc::new(Mutex::new(inner)),
        })
    }

    pub fn snapshot(&self) -> Result<RuntimeSnapshot, RuntimeError> {
        let inner = self.inner.lock();
        let models = inner
            .registry
            .list()
            .map_err(|e| RuntimeError::Storage(e.to_string()))?;
        Ok(RuntimeSnapshot {
            status: inner.status,
            hardware: inner.hardware.clone(),
            memory: inner.memory.snapshot(),
            active_model: inner.active_model.clone(),
            models,
            logs: inner.logs.clone(),
            console_lines: inner.console_lines.clone(),
            inference_backend: inner.backend.info(),
        })
    }

    pub fn backend_kind(&self) -> BackendKind {
        self.inner.lock().backend_kind
    }

    pub fn list_models(&self) -> Result<Vec<Model>, RuntimeError> {
        self.inner
            .lock()
            .registry
            .list()
            .map_err(|e| RuntimeError::Storage(e.to_string()))
    }

    pub fn delete_model(&self, id: Uuid) -> Result<(), RuntimeError> {
        let mut inner = self.inner.lock();
        if inner.active_model.as_ref().map(|m| m.id) == Some(id) {
            Self::unload_locked(&mut inner)?;
        }
        inner
            .registry
            .delete(id)
            .map_err(|e| RuntimeError::Storage(e.to_string()))?;
        inner.log(format!("deleted model {id}"));
        Ok(())
    }

    pub fn import_model_stub(
        &self,
        name: String,
        task_type: TaskType,
        memory_mb: u64,
    ) -> Result<Model, RuntimeError> {
        let mut inner = self.inner.lock();
        let mut model = Model::new(name, task_type, Architecture::Mock, memory_mb * 1024 * 1024);
        model.status = ModelStatus::Ready;
        model.description = "Imported stub (no weights) — registry entry only".into();
        inner
            .registry
            .upsert(&model)
            .map_err(|e| RuntimeError::Storage(e.to_string()))?;
        inner.log(format!("imported stub model {}", model.name));
        Ok(model)
    }

    pub fn compile_prompt(&self, prompt: &str) -> TaskProfile {
        self.inner.lock().compiler.compile(prompt)
    }

    pub fn compile_category(
        &self,
        category: TaskCategory,
        language: Option<String>,
        memory_limit_mb: Option<u64>,
    ) -> TaskProfile {
        let bytes = memory_limit_mb.map(|m| m * 1024 * 1024);
        self.inner
            .lock()
            .compiler
            .compile_category(category, language, bytes)
    }

    pub fn compile_brain<F>(
        &self,
        profile: TaskProfile,
        load: bool,
        mut on_progress: F,
    ) -> Result<Model, RuntimeError>
    where
        F: FnMut(GenerationProgress),
    {
        let mut inner = self.inner.lock();
        inner.status = RuntimeStatus::Compiling;
        inner.console(format!("> compile {}", profile.raw_prompt));
        inner.console("Generating neural configuration... (mock)");

        let generated = inner
            .generator
            .generate_with_progress(profile.clone(), |p| {
                on_progress(p.clone());
            })
            .map_err(|e| RuntimeError::Generator(e.to_string()))?;

        let path = inner
            .registry
            .models_dir()
            .join(format!("{}.nfcm-mock.json", generated.id));
        let meta = serde_json::json!({
            "id": generated.id,
            "name": generated.name,
            "is_mock": true,
            "memory_size_bytes": generated.memory_size_bytes,
            "layers": generated.layers,
            "latent": generated.latent,
            "optimization_profile": generated.optimization_profile,
            "task": generated.task,
        });
        std::fs::write(&path, serde_json::to_vec_pretty(&meta).unwrap_or_default())
            .map_err(|e| RuntimeError::Storage(e.to_string()))?;

        let _ = inner.cache.put(
            &format!("model-{}", generated.id),
            &serde_json::to_vec(&meta).unwrap_or_default(),
            Some(generated.id),
        );

        let task_type = category_to_task_type(profile.domain);
        let mut model = Model::new(
            generated.name.clone(),
            task_type,
            Architecture::NfcmSubnetwork,
            generated.memory_size_bytes,
        );
        model.id = generated.id;
        model.skills = profile.skills.clone();
        model.status = ModelStatus::Ready;
        model.path = Some(path.to_string_lossy().to_string());
        model.description = format!(
            "Mock-generated specialist ({}) — replace with hypernetwork later",
            profile.domain.as_str()
        );
        model.size_bytes = generated.memory_size_bytes;

        inner
            .registry
            .upsert(&model)
            .map_err(|e| RuntimeError::Storage(e.to_string()))?;

        inner.console("Loading components...");
        inner.log(format!("compiled mock brain: {}", model.name));

        if load {
            Self::load_generated_locked(&mut inner, model.clone(), generated)?;
        }

        inner.status = RuntimeStatus::Running;
        inner.console("Ready.");
        Ok(model)
    }

    pub fn load_model(&self, id: Uuid) -> Result<Model, RuntimeError> {
        let mut inner = self.inner.lock();
        let model = inner
            .registry
            .get(id)
            .map_err(|_| RuntimeError::ModelNotFound(id))?;
        let generated = GeneratedModel {
            id: model.id,
            name: model.name.clone(),
            task: TaskProfile {
                domain: task_type_to_category(model.task_type),
                skills: model.skills.clone(),
                language: None,
                memory_limit_bytes: model.memory_requirement_bytes,
                raw_prompt: model.name.clone(),
            },
            layers: vec![],
            weights: vec![],
            latent: nfc_generator::LatentCode {
                dim: 0,
                values: vec![],
                codebook_id: "reload".into(),
            },
            memory_size_bytes: model.memory_requirement_bytes,
            optimization_profile: nfc_generator::OptimizationProfile {
                quantize: true,
                target_memory_bytes: model.memory_requirement_bytes,
                activation_sparsity: 0.0,
                notes: "reloaded from registry".into(),
            },
            is_mock: true,
        };
        Self::load_generated_locked(&mut inner, model.clone(), generated)?;
        Ok(model)
    }

    pub fn unload_model(&self) -> Result<(), RuntimeError> {
        let mut inner = self.inner.lock();
        Self::unload_locked(&mut inner)
    }

    pub fn run_inference(
        &self,
        request: InferenceRequest,
    ) -> Result<InferenceResponse, RuntimeError> {
        let inner = self.inner.lock();
        if inner.active_model.is_none() {
            return Err(RuntimeError::NoModelLoaded);
        }
        inner
            .backend
            .infer(&request)
            .map_err(|e| RuntimeError::Inference(e.to_string()))
    }

    pub fn optimize_memory(&self) -> Result<u64, RuntimeError> {
        let mut inner = self.inner.lock();
        let freed = inner.memory.optimize();
        inner.log(format!("optimize_memory freed {freed} bytes"));
        inner.console(format!("Optimized memory (freed {freed} bytes)"));
        Ok(freed)
    }

    pub fn refresh_hardware(&self) -> Result<HardwareProfile, RuntimeError> {
        let profile =
            HardwareDetector::detect().map_err(|e| RuntimeError::Hardware(e.to_string()))?;
        self.inner.lock().hardware = profile.clone();
        Ok(profile)
    }

    pub fn append_console(&self, line: impl Into<String>) {
        self.inner.lock().console(line);
    }

    pub fn enqueue_job(&self, job: SchedulerJob) -> Uuid {
        self.inner.lock().scheduler.enqueue(job)
    }

    fn load_generated_locked(
        inner: &mut Inner,
        mut model: Model,
        generated: GeneratedModel,
    ) -> Result<(), RuntimeError> {
        if inner.active_model.is_some() {
            Self::unload_locked(inner)?;
        }

        let need = generated
            .memory_size_bytes
            .max(model.memory_requirement_bytes);
        match inner
            .memory
            .allocate(model.name.clone(), ComponentKind::ActiveModel, need)
        {
            Ok(alloc_id) => {
                inner.active_alloc = Some(alloc_id);
            }
            Err(e) => {
                warn!(error = %e, "allocation failed; attempting optimize");
                inner.memory.optimize();
                let alloc_id = inner
                    .memory
                    .allocate(model.name.clone(), ComponentKind::ActiveModel, need)
                    .map_err(|e| RuntimeError::Memory(e.to_string()))?;
                inner.active_alloc = Some(alloc_id);
            }
        }

        let ctx = ModelContext {
            model_id: model.id,
            model_name: model.name.clone(),
            architecture: format!("{:?}", model.architecture),
            weights_path: model.path.clone(),
            memory_requirement_bytes: model.memory_requirement_bytes,
        };
        if let Err(e) = inner.backend.attach(&ctx) {
            if let Some(alloc) = inner.active_alloc.take() {
                inner.memory.release(alloc);
            }
            return Err(RuntimeError::Inference(e.to_string()));
        }

        model.status = ModelStatus::Loaded;
        model.updated_at = chrono::Utc::now();
        inner
            .registry
            .upsert(&model)
            .map_err(|e| RuntimeError::Storage(e.to_string()))?;

        inner.active_model = Some(model.clone());
        inner.active_generated = Some(generated);
        inner.log(format!(
            "loaded model {} via backend {}",
            model.name,
            inner.backend.name()
        ));
        inner.console(format!("Loaded: {} [{}]", model.name, inner.backend.name()));
        Ok(())
    }

    fn unload_locked(inner: &mut Inner) -> Result<(), RuntimeError> {
        let _ = inner.backend.detach();
        if let Some(alloc) = inner.active_alloc.take() {
            inner.memory.release(alloc);
        }
        if let Some(mut model) = inner.active_model.take() {
            model.status = ModelStatus::Unloaded;
            model.updated_at = chrono::Utc::now();
            let _ = inner.registry.upsert(&model);
            inner.log(format!("unloaded model {}", model.name));
            inner.console(format!("Unloaded: {}", model.name));
        }
        inner.active_generated = None;
        Ok(())
    }
}

impl Inner {
    fn log(&mut self, msg: impl Into<String>) {
        let line = format!("{}  {}", chrono::Utc::now().format("%H:%M:%S"), msg.into());
        info!("{line}");
        self.logs.push(line);
        if self.logs.len() > 500 {
            self.logs.drain(0..self.logs.len() - 500);
        }
    }

    fn console(&mut self, msg: impl Into<String>) {
        self.console_lines.push(msg.into());
        if self.console_lines.len() > 500 {
            self.console_lines.drain(0..self.console_lines.len() - 500);
        }
    }
}

fn category_to_task_type(c: TaskCategory) -> TaskType {
    match c {
        TaskCategory::Coding => TaskType::Coding,
        TaskCategory::Math => TaskType::Math,
        TaskCategory::Writing => TaskType::Writing,
        TaskCategory::Research => TaskType::Research,
        TaskCategory::Medical => TaskType::Medical,
        TaskCategory::Custom => TaskType::Custom,
    }
}

fn task_type_to_category(t: TaskType) -> TaskCategory {
    match t {
        TaskType::Coding => TaskCategory::Coding,
        TaskType::Math => TaskCategory::Math,
        TaskType::Writing => TaskCategory::Writing,
        TaskType::Research => TaskCategory::Research,
        TaskType::Medical => TaskCategory::Medical,
        TaskType::Custom => TaskCategory::Custom,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn compile_load_infer_unload() {
        let dir = tempdir().unwrap();
        let engine =
            RuntimeEngine::start(RuntimeConfig::new(dir.path()).with_backend(BackendKind::Mock))
                .unwrap();
        let profile = engine.compile_prompt("I need a Python coding assistant");
        assert_eq!(profile.domain, TaskCategory::Coding);

        let model = engine.compile_brain(profile, true, |_| {}).unwrap();
        assert!(model.name.to_lowercase().contains("python") || model.name.contains("Coding"));

        let snap = engine.snapshot().unwrap();
        assert!(snap.active_model.is_some());
        assert_eq!(snap.status, RuntimeStatus::Running);
        assert!(snap.inference_backend.is_mock);
        assert!(snap.inference_backend.ready);

        let resp = engine
            .run_inference(InferenceRequest {
                prompt: "hello".into(),
                max_tokens: 64,
            })
            .unwrap();
        assert!(resp.is_mock);
        assert_eq!(resp.backend, "mock-v1");

        engine.unload_model().unwrap();
        assert!(engine.snapshot().unwrap().active_model.is_none());
        assert!(!engine.snapshot().unwrap().inference_backend.ready);
    }

    #[test]
    fn candle_without_feature_fails_attach_on_load() {
        let dir = tempdir().unwrap();
        let engine =
            RuntimeEngine::start(RuntimeConfig::new(dir.path()).with_backend(BackendKind::Candle))
                .unwrap();
        let profile = engine.compile_prompt("coding rust");
        let model = engine.compile_brain(profile, false, |_| {}).unwrap();
        #[cfg(not(feature = "candle"))]
        {
            let err = engine.load_model(model.id).unwrap_err();
            assert!(matches!(err, RuntimeError::Inference(_)));
        }
        #[cfg(feature = "candle")]
        {
            let _ = engine.load_model(model.id).unwrap();
        }
    }
}
