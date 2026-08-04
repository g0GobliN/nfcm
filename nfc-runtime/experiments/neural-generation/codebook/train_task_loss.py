#!/usr/bin/env python3
"""NFCM TARC — task-loss skill codebook trainer.

Owns three losses (numpy SGD, no PyTorch):
  L_domain  — sum of skill vectors predicts domain one-hot
  L_pair    — co-occurring skills reconstruct each other (linear)
  L_margin  — same-domain cosine high, cross-domain below margin

Writes checkpoints/skill-task-v1.json (SkillCodebook JSON).

Usage:
  python3 train_task_loss.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DIM = 64
STEPS = 4000
LR = 0.04
SEED = 42
MARGIN = 0.15
W_DOMAIN = 1.0
W_PAIR = 0.5
W_MARGIN = 1.0

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
    ],
    "math": ["math", "calculus", "statistics"],
    "writing": ["writing", "summarization"],
    "research": ["research", "summarization"],
    "medical": ["medical"],
}

DOMAINS = list(BUNDLES.keys())


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

    # Domain classifier W: [n_domains, DIM]
    W = rng.normal(0, 0.05, (len(DOMAINS), DIM)).astype(np.float32)

    hist: list[float] = []

    for step in range(STEPS):
        d = DOMAINS[step % len(DOMAINS)]
        bundle = BUNDLES[d]
        # Sample 1–3 skills from domain
        k = int(rng.integers(1, min(3, len(bundle)) + 1))
        chosen = list(rng.choice(bundle, size=k, replace=False))
        ids = [idx[s] for s in chosen]
        z = np.sum(vecs[ids], axis=0).astype(np.float32)

        # --- domain loss ---
        logits = W @ z
        probs = softmax(logits)
        y = DOMAINS.index(d)
        loss_domain = float(-np.log(probs[y] + 1e-8))
        # dL/dlogits
        dlogits = probs.copy()
        dlogits[y] -= 1.0
        g_W = np.outer(dlogits, z).astype(np.float32)
        g_z_domain = (W.T @ dlogits).astype(np.float32)

        # --- pair reconstruction (if >=2 skills) ---
        loss_pair = 0.0
        g_vecs_pair = {i: np.zeros(DIM, dtype=np.float32) for i in ids}
        if len(ids) >= 2:
            a, b = ids[0], ids[1]
            # predict b from a via shared W_pair = I (use cosine to target direction)
            pred = vecs[a]
            tgt = vecs[b]
            err = pred - tgt
            loss_pair = float(np.mean(err * err))
            g = (2.0 / DIM) * err
            g_vecs_pair[a] += g
            g_vecs_pair[b] -= g

        # --- margin contrastive ---
        a_s = chosen[0]
        ia = idx[a_s]
        # positive
        pos_cands = [s for s in bundle if s != a_s] or [a_s]
        b_s = rng.choice(pos_cands)
        ib = idx[b_s]
        neg_cands = [s for s in skills if skill_domain[s] != d]
        n_s = rng.choice(neg_cands)
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

        # Apply grads to skill vectors that participated
        # domain: distribute g_z_domain across chosen skills
        for i in ids:
            vecs[i] = l2n(vecs[i] - LR * (W_DOMAIN * g_z_domain / len(ids) + W_PAIR * g_vecs_pair[i]))
        vecs[ia] = l2n(vecs[ia] - LR * W_MARGIN * g_a)
        vecs[ib] = l2n(vecs[ib] - LR * W_MARGIN * g_b)
        vecs[inn] = l2n(vecs[inn] - LR * W_MARGIN * g_n)
        W -= LR * W_DOMAIN * g_W

        if step % 500 == 0 or step == STEPS - 1:
            print(
                f"step={step:4d} loss={loss:.4f} "
                f"domain={loss_domain:.3f} pair={loss_pair:.3f} margin={loss_margin:.3f} "
                f"acc_hint={probs[y]:.2f}"
            )

    # Domain accuracy probe
    correct = 0
    total = 0
    for d, bundle in BUNDLES.items():
        for _ in range(20):
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
        "id": "skill-task-v1",
        "dim": DIM,
        "version": 1,
        "notes": (
            "NFCM TARC task-loss codebook (domain + pair + margin). "
            "Research toy — not a foundation model."
        ),
        "entries": entries,
        "train": {
            "method": "TARC-task-loss",
            "steps": STEPS,
            "lr": LR,
            "seed": SEED,
            "domain_acc_probe": domain_acc,
            "mean_within_cosine": float(np.mean(within)),
            "mean_across_cosine": float(np.mean(across)),
            "final_loss": hist[-1],
            "mean_loss_last_200": float(np.mean(hist[-200:])),
        },
    }
    out_dir = Path(__file__).resolve().parent / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "skill-task-v1.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(
        f"wrote {path} domain_acc={domain_acc:.3f} "
        f"within={np.mean(within):.3f} across={np.mean(across):.3f}"
    )
    if domain_acc < 0.55:
        raise SystemExit("domain probe too weak — retune")


if __name__ == "__main__":
    main()
