#!/usr/bin/env python3
"""NFCM TARC — train a tiny decode head on task phrases (fixed 128-word vocab).

Learns E[128, D] so that for a context activation a:
  logits = E @ a   →   next / bag words from short task phrases.

Writes checkpoints/decode-v2.json (preferred) consumed by Rust latent-decode
(NFCM_DECODE_HEAD=/path or auto-resolve).

Usage:
  python3 train_decode.py

Honest scope: toy linear head on a tiny in-vocab phrase bank — not an LLM.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DIM = 128
STEPS = 6000
LR = 0.04
SEED = 42

VOCAB = [
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
]
assert len(VOCAB) == 128

CORPUS_PATH = Path(__file__).resolve().parent.parent / "corpus" / "docs.json"


def load_phrases() -> dict[str, list[str]]:
    """In-vocab phrases from shared corpus docs (drop OOV tokens)."""
    raw = json.loads(CORPUS_PATH.read_text())
    vocab = set(VOCAB)
    out: dict[str, list[str]] = {}
    for doc in raw:
        domain = doc["domain"]
        kept = [w for w in doc["text"].split() if w in vocab]
        if len(kept) < 3:
            continue
        out.setdefault(domain, []).append(" ".join(kept))
    if not out:
        raise SystemExit(f"no in-vocab phrases in {CORPUS_PATH}")
    return out


PHRASES = load_phrases()


def word_idx(w: str) -> int:
    return VOCAB.index(w)


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


def tokenize(phrase: str) -> list[int]:
    ids = []
    for w in phrase.split():
        if w not in VOCAB:
            raise SystemExit(f"OOV word in phrase bank: {w!r}")
        ids.append(word_idx(w))
    return ids


def ctx_activation(ctx: list[int], rng: np.random.Generator) -> np.ndarray:
    """Bag of context word one-hots + noise → L2 unit vector."""
    a = rng.normal(0, 0.05, DIM).astype(np.float32)
    for i, tok in enumerate(ctx):
        a[tok] += 1.0 / (1.0 + 0.15 * i)
    a /= np.linalg.norm(a) + 1e-6
    return a


def build_examples() -> list[tuple[list[int], int]]:
    """(context_ids, next_word_id) pairs from phrases."""
    examples: list[tuple[list[int], int]] = []
    for phrases in PHRASES.values():
        for phrase in phrases:
            ids = tokenize(phrase)
            for t in range(1, len(ids)):
                # Use up to last 4 tokens as context
                ctx = ids[max(0, t - 4) : t]
                examples.append((ctx, ids[t]))
    return examples


def main() -> None:
    rng = np.random.default_rng(SEED)
    E = rng.normal(0, 0.05, (128, DIM)).astype(np.float32)
    examples = build_examples()
    losses: list[float] = []
    hits = 0
    seen = 0

    for step in range(STEPS):
        ctx, target = examples[step % len(examples)]
        a = ctx_activation(ctx, rng)
        logits = E @ a
        probs = softmax(logits)
        y = np.zeros(128, dtype=np.float32)
        y[target] = 1.0
        loss = float(-np.log(probs[target] + 1e-8))
        losses.append(loss)
        g_E = np.outer(probs - y, a).astype(np.float32)
        E -= LR * g_E

        pred = int(np.argmax(probs))
        hits += int(pred == target)
        seen += 1

        if step % 1000 == 0 or step == STEPS - 1:
            acc = hits / max(seen, 1)
            print(
                f"step={step:4d} loss={loss:.4f} "
                f"recent_top1={acc:.3f} pred={VOCAB[pred]} tgt={VOCAB[target]}"
            )
            hits = 0
            seen = 0

    # Eval: teacher-forced next-word top-1 on full bank
    correct = 0
    total = 0
    by_domain: dict[str, dict] = {}
    for domain, phrases in PHRASES.items():
        d_ok = 0
        d_n = 0
        samples = []
        for phrase in phrases:
            ids = tokenize(phrase)
            for t in range(1, len(ids)):
                ctx = ids[max(0, t - 4) : t]
                a = ctx_activation(ctx, rng)
                # deterministic: zero noise for eval
                a = np.zeros(DIM, dtype=np.float32)
                for i, tok in enumerate(ctx):
                    a[tok] += 1.0 / (1.0 + 0.15 * i)
                a /= np.linalg.norm(a) + 1e-6
                pred = int(np.argmax(E @ a))
                ok = pred == ids[t]
                correct += int(ok)
                total += 1
                d_ok += int(ok)
                d_n += 1
                if len(samples) < 3:
                    samples.append(
                        {
                            "ctx": " ".join(VOCAB[i] for i in ctx),
                            "pred": VOCAB[pred],
                            "tgt": VOCAB[ids[t]],
                            "ok": ok,
                        }
                    )
        by_domain[domain] = {
            "top1": d_ok / max(d_n, 1),
            "n": d_n,
            "samples": samples,
        }

    top1 = correct / max(total, 1)
    out_dir = Path(__file__).resolve().parent / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "decode-v2.json"
    payload = {
        "version": 1,
        "dim": DIM,
        "vocab": VOCAB,
        "e": E.reshape(-1).tolist(),
        "notes": (
            f"NFCM TARC phrase-trained decode head; seed={SEED}; steps={STEPS}; "
            f"top1={top1:.3f}; toy task text — not an LLM"
        ),
        "train": {
            "top1": top1,
            "n_examples": len(examples),
            "by_domain": {k: {"top1": v["top1"], "n": v["n"]} for k, v in by_domain.items()},
            "final_loss": losses[-1],
        },
    }
    path.write_text(json.dumps(payload))
    print(f"wrote {path} top1={top1:.3f} n={total}")
    for d, s in by_domain.items():
        print(f"  {d}: top1={s['top1']:.3f} n={s['n']} eg={s['samples'][0]}")
    if top1 < 0.35:
        raise SystemExit("decode head too weak on phrase bank")


if __name__ == "__main__":
    main()
