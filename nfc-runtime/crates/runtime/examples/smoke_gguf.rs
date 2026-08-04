//! Real GGUF smoke: import TinyLlama, infer one prompt.
//!
//! ```bash
//! ./scripts/smoke-gguf.sh
//! ```

use nfc_runtime::{BackendKind, GeneratorKind, InferenceRequest, RuntimeConfig, RuntimeEngine};
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter("nfc_runtime=info")
        .init();

    let model = std::env::var("NFCM_GGUF_MODEL")
        .map(PathBuf::from)
        .expect("NFCM_GGUF_MODEL");
    if !model.is_file() {
        return Err(format!("missing model {}", model.display()).into());
    }

    let dir = tempfile::tempdir()?;
    let engine = RuntimeEngine::start(
        RuntimeConfig::new(dir.path())
            .with_backend(BackendKind::Gguf)
            .with_generator(GeneratorKind::Mock),
    )?;

    let imported = engine.import_gguf_model(
        model.to_string_lossy().to_string(),
        Some("TinyLlama Chat".into()),
        512,
    )?;
    println!("Imported: {} ({})", imported.name, imported.id);

    engine.load_model(imported.id)?;
    let snap = engine.snapshot()?;
    println!(
        "Backend: {} mock={} ready={}",
        snap.inference_backend.name, snap.inference_backend.is_mock, snap.inference_backend.ready
    );

    let resp = engine.run_inference(InferenceRequest {
        prompt: "Say hello in one short sentence.".into(),
        max_tokens: 48,
    })?;
    println!("\n[{}] {}", resp.backend, resp.text);
    assert!(!resp.is_mock);
    assert!(!resp.text.trim().is_empty());
    Ok(())
}
