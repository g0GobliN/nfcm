//! Headless smoke demo for the NFCM runtime (no UI).

use nfc_runtime::{InferenceRequest, RuntimeConfig, RuntimeEngine};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter("nfc_runtime=info")
        .init();

    let dir = tempfile::tempdir()?;
    let engine = RuntimeEngine::start(RuntimeConfig::new(dir.path()))?;

    let profile = engine.compile_prompt("I need a Python coding assistant");
    println!("TaskProfile: {profile:?}");

    let model = engine.compile_brain(profile, true, |p| {
        println!("  progress: {}", p.label());
    })?;
    println!("Loaded: {} ({})", model.name, model.id);

    let snap = engine.snapshot()?;
    println!(
        "Memory: {} / {} MB",
        snap.memory.used_mb(),
        snap.memory.max_mb()
    );

    let resp = engine.run_inference(InferenceRequest {
        prompt: "explain ownership".into(),
        max_tokens: 64,
    })?;
    println!("\n{}", resp.text);

    Ok(())
}
