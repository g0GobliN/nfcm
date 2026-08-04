#!/usr/bin/env python3
"""NFCM TARC — task-metric hypernetwork trainer (scaled v2).

Goes beyond teacher-mimic. Losses:
  L_recon   — decode task latent z → weight patch ≈ task-conditioned teacher
  L_sep     — Frobenius distance between coding vs math specialist weights
  L_budget  — keep |alpha| modest (RAM proxy / soft compression)

Writes checkpoints/task-v2.json (HyperCheckpoint JSON for Rust loader).

Usage:
  python3 train_task.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

LATENT_DIM = 128
H = 128
ROWS, COLS = 64, 64
STEPS = 4000
LR = 0.025
SEED = 13
W_RECON = 1.0
W_SEP = 0.15
W_BUDGET = 0.02


def domain_latent(rng: np.random.Generator, domain: str) -> np.ndarray:
    z = rng.uniform(-1, 1, LATENT_DIM).astype(np.float32)
    slots = {"coding": 0, "math": 1, "writing": 2, "research": 3, "medical": 4}
    z[slots[domain]] += 0.55
    z += rng.normal(0, 0.05, LATENT_DIM).astype(np.float32)
    return z


def teacher(z: np.ndarray, rows: int, cols: int) -> np.ndarray:
    d = z.shape[0]
    w = np.zeros((rows, cols), dtype=np.float32)
    scale = 0.12 + 0.08 * float(np.mean(np.abs(z[:5])))
    for r in range(rows):
        for c in range(cols):
            w[r, c] = scale * z[r % d] * z[c % d]
    return w


def student(z: np.ndarray, p: np.ndarray, q: np.ndarray, alpha: float) -> np.ndarray:
    u = p @ z
    v = q @ z
    w = np.zeros((ROWS, COLS), dtype=np.float32)
    for r in range(ROWS):
        for c in range(COLS):
            w[r, c] = alpha * u[r % H] * v[c % H]
    return w


def fro_dist(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    return float(np.sqrt(np.mean(d * d)))


def main() -> None:
    rng = np.random.default_rng(SEED)
    p = np.eye(H, LATENT_DIM, dtype=np.float32)
    q = np.eye(H, LATENT_DIM, dtype=np.float32)
    p += rng.normal(0, 0.01, p.shape).astype(np.float32)
    q += rng.normal(0, 0.01, q.shape).astype(np.float32)
    alpha = np.float32(0.12)

    losses: list[float] = []
    seps: list[float] = []
    domains = ["coding", "math", "writing", "research", "medical"]

    for step in range(STEPS):
        z = domain_latent(rng, domains[step % len(domains)])
        tgt = teacher(z, ROWS, COLS)
        pred = student(z, p, q, float(alpha))
        err = pred - tgt
        loss_recon = float(np.mean(err * err))

        z_c = domain_latent(rng, "coding")
        z_m = domain_latent(rng, "math")
        w_c = student(z_c, p, q, float(alpha))
        w_m = student(z_m, p, q, float(alpha))
        sep = fro_dist(w_c, w_m)
        loss_sep = -sep
        loss_budget = float(alpha * alpha)
        loss = W_RECON * loss_recon + W_SEP * loss_sep + W_BUDGET * loss_budget
        losses.append(loss)
        seps.append(sep)

        u = p @ z
        v = q @ z
        d_alpha = 0.0
        d_u = np.zeros(H, dtype=np.float32)
        d_v = np.zeros(H, dtype=np.float32)
        scale = 2.0 / (ROWS * COLS)
        for r in range(ROWS):
            for c in range(COLS):
                e = err[r, c]
                ur = u[r % H]
                vc = v[c % H]
                d_alpha += scale * e * ur * vc
                d_u[r % H] += scale * e * float(alpha) * vc
                d_v[c % H] += scale * e * float(alpha) * ur
        g_p = np.outer(d_u, z).astype(np.float32) * W_RECON
        g_q = np.outer(d_v, z).astype(np.float32) * W_RECON
        d_alpha *= W_RECON

        eps = 1e-3
        alpha_p = float(alpha) + eps
        sep_p = fro_dist(student(z_c, p, q, alpha_p), student(z_m, p, q, alpha_p))
        d_alpha += W_SEP * (-(sep_p - sep) / eps)
        d_alpha += W_BUDGET * 2.0 * float(alpha)

        p -= LR * g_p
        q -= LR * g_q
        alpha = np.float32(alpha - LR * d_alpha)

        if step % 500 == 0 or step == STEPS - 1:
            print(
                f"step={step:4d} loss={loss:.5f} recon={loss_recon:.5f} "
                f"sep={sep:.4f} alpha={float(alpha):.4f}"
            )

    final_sep = float(np.mean(seps[-100:]))
    z_c = domain_latent(rng, "coding")
    recon_c = float(np.mean((student(z_c, p, q, float(alpha)) - teacher(z_c, ROWS, COLS)) ** 2))

    out_dir = Path(__file__).resolve().parent / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "task-v2.json"
    payload = {
        "version": 1,
        "latent_dim": LATENT_DIM,
        "alpha": float(alpha),
        "p_rows": H,
        "p_cols": LATENT_DIM,
        "p": p.reshape(-1).tolist(),
        "q_rows": H,
        "q_cols": LATENT_DIM,
        "q": q.reshape(-1).tolist(),
        "notes": (
            f"NFCM TARC task-metric hypernet v2; dim={LATENT_DIM}; patch={ROWS}x{COLS}; "
            f"seed={SEED}; steps={STEPS}; recon_c={recon_c:.3e}; mean_sep={final_sep:.4f}; "
            "research toy — not an LLM"
        ),
    }
    out.write_text(json.dumps(payload))
    print(f"wrote {out} recon_c={recon_c:.3e} sep={final_sep:.4f}")
    if losses[-1] > losses[0] and recon_c > 1e-2:
        raise SystemExit("training did not improve enough")


if __name__ == "__main__":
    main()
