#!/usr/bin/env python3
"""NFCM TARC — task-loss skill codebook trainer (scaled v2).

Owns three losses (numpy SGD, no PyTorch):
  L_domain  — sum of skill vectors predicts domain one-hot
  L_pair    — co-occurring skills reconstruct each other
  L_margin  — same-domain cosine high, cross-domain below margin

Corpus: expanded skill bank + short co-occurrence phrases (toy text).

Writes checkpoints/skill-task-v2.json (SkillCodebook JSON).

Usage:
  python3 train_task_loss.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DIM = 128
STEPS = 8000
LR = 0.035
SEED = 42
MARGIN = 0.12
W_DOMAIN = 1.0
W_PAIR = 0.6
W_MARGIN = 1.0

# Expanded skill bank (still toy — not a foundation corpus)
BUNDLES: dict[str, list[str]] = {
    "coding": [
        "python",
        "rust",
        "javascript",
        "typescript",
        "debugging",
        "testing",
        "refactoring",
        "algorithms",
        "api",
        "cli",
        "async",
        "parsing",
        "logging",
        "memory",
        "concurrency",
        "types",
    ],
    "math": [
        "math",
        "calculus",
        "statistics",
        "linear-algebra",
        "probability",
        "optimization",
        "proofs",
        "numerics",
    ],
    "writing": [
        "writing",
        "summarization",
        "editing",
        "outline",
        "clarity",
        "docs",
    ],
    "research": [
        "research",
        "summarization",
        "literature",
        "hypothesis",
        "experiment",
        "citation",
        "analysis",
    ],
    "medical": [
        "medical",
        "triage",
        "notes",
        "safety",
        "symptoms",
        "dosage",
    ],
}

DOMAINS = list(BUNDLES.keys())
CORPUS_PATH = Path(__file__).resolve().parent.parent / "corpus" / "docs.json"


def load_corpus_cooccur() -> list[list[str]]:
    """Skill co-occurrence lists from the shared corpus docs."""
    raw = json.loads(CORPUS_PATH.read_text())
    out: list[list[str]] = []
    for doc in raw:
        skills = [s for s in doc.get("skills", []) if isinstance(s, str)]
        if len(skills) >= 2:
            out.append(skills)
    if not out:
        raise SystemExit(f"corpus empty or missing skills: {CORPUS_PATH}")
    return out


COOCCUR = load_corpus_cooccur()


def all_skills() -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for skills in BUNDLES.values():
        for s in skills:
            if s not in seen:
                seen.add(s)
                out.append(s)
    return out


def l2n(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v if n < 1e-8 else v / n


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


def main() -> None:
    rng = np.random.default_rng(SEED)
    skills = all_skills()
    idx = {s: i for i, s in enumerate(skills)}
    skill_domain = {s: d for d, ss in BUNDLES.items() for s in ss}

    vecs = rng.normal(0, 0.4, (len(skills), DIM)).astype(np.float32)
    for i in range(len(skills)):
        vecs[i] = l2n(vecs[i])

    W = rng.normal(0, 0.05, (len(DOMAINS), DIM)).astype(np.float32)
    hist: list[float] = []

    for step in range(STEPS):
        d = DOMAINS[step % len(DOMAINS)]
        bundle = BUNDLES[d]
        k = int(rng.integers(1, min(4, len(bundle)) + 1))
        chosen = list(rng.choice(bundle, size=k, replace=False))
        ids = [idx[s] for s in chosen]
        z = np.sum(vecs[ids], axis=0).astype(np.float32)

        logits = W @ z
        probs = softmax(logits)
        y = DOMAINS.index(d)
        loss_domain = float(-np.log(probs[y] + 1e-8))
        dlogits = probs.copy()
        dlogits[y] -= 1.0
        g_W = np.outer(dlogits, z).astype(np.float32)
        g_z_domain = (W.T @ dlogits).astype(np.float32)

        loss_pair = 0.0
        g_vecs_pair = {i: np.zeros(DIM, dtype=np.float32) for i in ids}
        # Prefer corpus co-occurrence when available
        if step % 2 == 0 and COOCCUR:
            phrase = COOCCUR[step % len(COOCCUR)]
            phrase = [s for s in phrase if s in idx]
            if len(phrase) >= 2:
                a, b = idx[phrase[0]], idx[phrase[1]]
                err = vecs[a] - vecs[b]
                loss_pair = float(np.mean(err * err))
                g = (2.0 / DIM) * err
                g_vecs_pair = {a: g, b: -g}
                ids = list({a, b} | set(ids))
        elif len(ids) >= 2:
            a, b = ids[0], ids[1]
            err = vecs[a] - vecs[b]
            loss_pair = float(np.mean(err * err))
            g = (2.0 / DIM) * err
            g_vecs_pair[a] += g
            g_vecs_pair[b] -= g

        a_s = chosen[0]
        ia = idx[a_s]
        pos_cands = [s for s in bundle if s != a_s] or [a_s]
        b_s = str(rng.choice(pos_cands))
        ib = idx[b_s]
        neg_cands = [s for s in skills if skill_domain[s] != d]
        n_s = str(rng.choice(neg_cands))
        inn = idx[n_s]
        va, vb, vn = vecs[ia], vecs[ib], vecs[inn]
        c_pos = float(np.dot(va, vb))
        c_neg = float(np.dot(va, vn))
        loss_pos = (1.0 - c_pos) ** 2
        loss_neg = max(0.0, c_neg - MARGIN) ** 2
        loss_margin = loss_pos + loss_neg
        g_a = -2.0 * (1.0 - c_pos) * vb
        g_b = -2.0 * (1.0 - c_pos) * va
        g_n = np.zeros(DIM, dtype=np.float32)
        if c_neg > MARGIN:
            g_a = g_a + 2.0 * (c_neg - MARGIN) * vn
            g_n = 2.0 * (c_neg - MARGIN) * va

        loss = W_DOMAIN * loss_domain + W_PAIR * loss_pair + W_MARGIN * loss_margin
        hist.append(loss)

        touched = set(ids) | {ia, ib, inn}
        for i in touched:
            g = np.zeros(DIM, dtype=np.float32)
            if i in ids:
                g = g + W_DOMAIN * g_z_domain / max(len(ids), 1)
            if i in g_vecs_pair:
                g = g + W_PAIR * g_vecs_pair[i]
            if i == ia:
                g = g + W_MARGIN * g_a
            if i == ib:
                g = g + W_MARGIN * g_b
            if i == inn:
                g = g + W_MARGIN * g_n
            vecs[i] = l2n(vecs[i] - LR * g)
        W -= LR * W_DOMAIN * g_W

        if step % 1000 == 0 or step == STEPS - 1:
            print(
                f"step={step:4d} loss={loss:.4f} "
                f"domain={loss_domain:.3f} pair={loss_pair:.3f} margin={loss_margin:.3f} "
                f"acc_hint={probs[y]:.2f}"
            )

    correct = 0
    total = 0
    for d, bundle in BUNDLES.items():
        for _ in range(30):
            k = int(rng.integers(1, min(3, len(bundle)) + 1))
            chosen = list(rng.choice(bundle, size=k, replace=False))
            z = np.sum(vecs[[idx[s] for s in chosen]], axis=0)
            pred = int(np.argmax(W @ z))
            correct += int(pred == DOMAINS.index(d))
            total += 1
    domain_acc = correct / total

    within, across = [], []
    for i, s1 in enumerate(skills):
        for j, s2 in enumerate(skills):
            if i >= j:
                continue
            c = float(np.dot(vecs[i], vecs[j]))
            if skill_domain[s1] == skill_domain[s2]:
                within.append(c)
            else:
                across.append(c)

    entries = {s: [round(float(x), 6) for x in vecs[idx[s]]] for s in skills}
    out = {
        "id": "skill-task-v2",
        "dim": DIM,
        "version": 1,
        "notes": (
            "NFCM TARC task-loss codebook v2 (dim=128, expanded skills + phrase co-occur). "
            "Research toy — not a foundation model."
        ),
        "entries": entries,
        "train": {
            "method": "TARC-task-loss-v2",
            "steps": STEPS,
            "lr": LR,
            "seed": SEED,
            "n_skills": len(skills),
            "n_cooccur": len(COOCCUR),
            "corpus": str(CORPUS_PATH.name),
            "domain_acc_probe": domain_acc,
            "mean_within_cosine": float(np.mean(within)),
            "mean_across_cosine": float(np.mean(across)),
            "final_loss": hist[-1],
            "mean_loss_last_200": float(np.mean(hist[-200:])),
        },
    }
    out_dir = Path(__file__).resolve().parent / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "skill-task-v2.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(
        f"wrote {path} skills={len(skills)} domain_acc={domain_acc:.3f} "
        f"within={np.mean(within):.3f} across={np.mean(across):.3f}"
    )
    if domain_acc < 0.55:
        raise SystemExit("domain probe too weak — retune")


if __name__ == "__main__":
    main()
