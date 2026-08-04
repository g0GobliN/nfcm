//! Compare mock vs latent generators at a fixed RAM budget.
//!
//! ```bash
//! cargo run -p nfc-runtime --example eval_generators
//! ```
//!
//! Prints JSON lines suitable for pasting into experiment notes.
//! Does **not** claim model quality — only structural / footprint metrics.

use nfc_generator::{create_generator, GeneratorKind, TaskCompiler, WeightGenerator};
use serde_json::json;

fn main() {
    let compiler = TaskCompiler::new(256 * 1024 * 1024);
    let prompts = [
        "I need a Python coding assistant",
        "help me prove this math problem",
        "write a clear product blog post",
        "summarize medical research papers",
    ];

    println!("# NFCM generator eval (structural only — not quality scores)");
    println!("budget_mb=256");

    for kind in [GeneratorKind::Mock, GeneratorKind::Latent] {
        let gen = create_generator(kind);
        for prompt in prompts {
            let profile = compiler.compile(prompt);
            let model = WeightGenerator::generate(gen.as_ref(), profile.clone()).expect("generate");
            let alloc: usize = model.weights.iter().map(|w| w.memory_bytes()).sum();
            let row = json!({
                "generator": WeightGenerator::name(gen.as_ref()),
                "kind": kind.as_str(),
                "prompt": prompt,
                "domain": profile.domain.as_str(),
                "skills": profile.skills,
                "is_mock": model.is_mock,
                "layers": model.layers.len(),
                "latent_dim": model.latent.dim,
                "claimed_mb": model.memory_size_bytes / (1024 * 1024),
                "allocated_bytes": alloc,
                "notes": model.optimization_profile.notes,
            });
            println!("{}", row);
        }
    }
}
