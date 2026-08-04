//! Fixed-RAM eval suite — structural footprint + latent discrimination + probe energy.
//!
//! ```bash
//! cargo run -p nfc-runtime --example eval_generators
//! NFCM_EVAL_BUDGET_MB=128 cargo run -p nfc-runtime --example eval_generators
//! ```
//!
//! Does **not** claim LLM quality — only research plumbing metrics.

use nfc_generator::{create_generator, run_suite, GeneratorKind, WeightGenerator};
use nfc_runtime::{InferenceBackend, LatentProbeBackend, ModelContext};
use serde_json::json;

fn main() {
    let budget_mb: u64 = std::env::var("NFCM_EVAL_BUDGET_MB")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(256);
    let budget = budget_mb * 1024 * 1024;

    println!("# NFCM fixed-RAM eval suite");
    println!("budget_mb={budget_mb}");
    println!("notes=structural+discrimination+probe_energy — not LLM quality");
    println!();

    for kind in [GeneratorKind::Mock, GeneratorKind::Latent] {
        let gen = create_generator(kind);
        let report = run_suite(gen.as_ref(), budget);
        println!("## generator={}", report.generator);
        if let Some(d) = &report.discrimination {
            println!(
                "discrimination within={:.4} across={:.4} separation={:.4}",
                d.mean_within_cosine, d.mean_across_cosine, d.separation
            );
        } else {
            println!("discrimination=n/a");
        }
        println!("all_under_claim={}", report.all_under_claim);

        for case in &report.cases {
            let mut probe_energy = None;
            if !case.is_mock {
                // Rebuild model for probe attach (suite already generated; regenerate once).
                if let Some(eval_case) = nfc_generator::default_cases(budget)
                    .into_iter()
                    .find(|c| c.id == case.case)
                {
                    if let Ok(model) = WeightGenerator::generate(gen.as_ref(), eval_case.profile) {
                        let mut probe = LatentProbeBackend::new();
                        let ctx = ModelContext {
                            model_id: model.id,
                            model_name: model.name.clone(),
                            architecture: "NfcmSubnetwork".into(),
                            weights_path: None,
                            memory_requirement_bytes: model.memory_size_bytes,
                            skills: model.task.skills.clone(),
                            latent_values: model.latent.values.clone(),
                            weights: model.weights.clone(),
                        };
                        if probe.attach(&ctx).is_ok() {
                            let matched = model.task.raw_prompt.clone();
                            let mismatched = if model.task.domain.as_str() == "coding" {
                                "prove this calculus identity"
                            } else {
                                "debug this python stacktrace"
                            };
                            if let (Ok(e_match), Ok(e_mis)) = (
                                probe.activation_energy(&matched),
                                probe.activation_energy(mismatched),
                            ) {
                                probe_energy = Some(json!({
                                    "matched_prompt": e_match,
                                    "mismatched_prompt": e_mis,
                                    "delta": e_match - e_mis,
                                }));
                            }
                            let _ = probe.detach();
                        }
                    }
                }
            }

            let row = json!({
                "case": case.case,
                "generator": report.generator,
                "kind": kind.as_str(),
                "domain": case.domain,
                "skills": case.skills,
                "skill_hits": case.skill_hits,
                "codebook_id": case.codebook_id,
                "layers": case.layers,
                "latent_dim": case.latent_dim,
                "claimed_mb": case.claimed_bytes / (1024 * 1024),
                "allocated_bytes": case.allocated_bytes,
                "under_claim": case.under_claim,
                "probe_energy": probe_energy,
            });
            println!("{row}");
        }
        println!();
    }
}
