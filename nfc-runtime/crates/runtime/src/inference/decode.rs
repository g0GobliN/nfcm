//! Tiny text decode from latent activations.
//!
//! Maps the final activation vector onto a fixed 128-word vocabulary and
//! greedily emits tokens. Uses real tensors from the forward pass — still
//! **not** a trained LLM.

/// Fixed vocab (index = logit slot). Length must stay 128.
pub const VOCAB: [&str; 128] = [
    "the", "a", "to", "of", "and", "in", "for", "is", "on", "with", "this", "that", "you", "it",
    "as", "be", "are", "from", "or", "at", "by", "an", "we", "can", "will", "use", "code",
    "function", "error", "fix", "check", "try", "return", "value", "type", "data", "model",
    "local", "memory", "skill", "task", "step", "first", "then", "next", "also", "here", "need",
    "help", "please", "sure", "python", "rust", "debug", "test", "math", "proof", "write",
    "research", "summary", "medical", "note", "input", "output", "layer", "weight", "latent",
    "compile", "brain", "specialist", "budget", "ram", "device", "safe", "clear", "simple",
    "example", "pattern", "logic", "plan", "answer", "question", "why", "how", "what", "when",
    "where", "because", "so", "if", "else", "true", "false", "null", "list", "map", "set", "loop",
    "call", "run", "build", "load", "save", "open", "read", "parse", "print", "log", "ok", "done",
    "start", "end", "high", "low", "more", "less", "good", "maybe", "token", "decode", "probe",
    "nfcm", "tarc", "hello", "world", "thanks", "path", "file",
];

fn softmax_argmax(logits: &[f32]) -> usize {
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    best_i
}

fn l2_normalize(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
    for x in v.iter_mut() {
        *x /= n;
    }
}

/// Project activations → logits over [`VOCAB`], greedy decode up to `max_tokens`.
pub fn decode_tokens(
    acts: &[f32],
    latent: &[f32],
    skills: &[String],
    prompt: &str,
    max_tokens: usize,
) -> String {
    let n = VOCAB.len().min(acts.len().max(1));
    let mut state: Vec<f32> = acts.iter().copied().take(n).collect();
    if state.len() < n {
        state.resize(n, 0.0);
    }
    for (i, s) in state.iter_mut().enumerate() {
        let lv = latent
            .get(i % latent.len().max(1))
            .copied()
            .unwrap_or(0.0);
        *s += 0.15 * lv;
    }
    for (si, skill) in skills.iter().enumerate() {
        let h = simple_hash(skill.as_bytes()) as usize;
        let idx = h % n;
        state[idx] += 0.25 + 0.05 * si as f32;
    }
    for (ti, tok) in prompt.split_whitespace().take(12).enumerate() {
        let h = simple_hash(tok.as_bytes()) as usize % n;
        state[h] += 0.12 / (1.0 + ti as f32);
    }

    let max_tokens = max_tokens.clamp(4, 64);
    let mut out = Vec::with_capacity(max_tokens);
    let mut prev = None::<usize>;
    for _ in 0..max_tokens {
        let mut logits = state.clone();
        if let Some(p) = prev {
            logits[p] -= 1.5;
        }
        let i = softmax_argmax(&logits);
        let word = VOCAB[i];
        out.push(word);
        prev = Some(i);
        if out.len() >= 8 && matches!(word, "done" | "ok" | "end" | "thanks") {
            break;
        }
        for (j, s) in state.iter_mut().enumerate() {
            *s *= 0.82;
            if j == i {
                *s += 0.55;
            }
        }
        l2_normalize(&mut state);
    }

    let joined = out.join(" ");
    let mut chars = joined.chars();
    match chars.next() {
        None => String::new(),
        Some(c) => c.to_uppercase().collect::<String>() + chars.as_str(),
    }
}

fn simple_hash(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in bytes {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_emits_words() {
        let acts: Vec<f32> = (0..128).map(|i| (i as f32 * 0.01).sin()).collect();
        let text = decode_tokens(&acts, &[0.2, -0.1], &["python".into()], "fix bug", 16);
        assert!(!text.is_empty());
        assert!(text.split_whitespace().count() >= 4);
    }

    #[test]
    fn vocab_len_is_128() {
        assert_eq!(VOCAB.len(), 128);
    }
}
