#!/usr/bin/env python3
"""Train a toy hypernetwork remix (P, Q, alpha) to match the Rust outer-product teacher.

Teacher (noise-free): W[r,c] = 0.15 * z[r % D] * z[c % D]
Student:              W[r,c] = alpha * (P @ z)[r % Hp] * (Q @ z)[c % Hq]

Usage:
  python3 train_toy.py
  # writes checkpoints/toy-v1.json

No PyTorch required (numpy only).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

LATENT_DIM = 64
H = 64  # projection rows (matches latent dim; keeps checkpoint small)
ROWS, COLS = 32, 32  # train on small patches
STEPS = 800
LR = 0.05
SEED = 7


def teacher(z: np.ndarray, rows: int, cols: int) -> np.ndarray:
    d = z.shape[0]
    w = np.zeros((rows, cols), dtype=np.float32)
    for r in range(rows):
        for c in range(cols):
            w[r, c] = 0.15 * z[r % d] * z[c % d]
    return w


def student(z: np.ndarray, p: np.ndarray, q: np.ndarray, alpha: float, rows: int, cols: int) -> np.ndarray:
    u = p @ z
    v = q @ z
    w = np.zeros((rows, cols), dtype=np.float32)
    for r in range(rows):
        for c in range(cols):
            w[r, c] = alpha * u[r % u.shape[0]] * v[c % v.shape[0]]
    return w


def main() -> None:
    rng = np.random.default_rng(SEED)
    # Init near identity remix
    p = np.zeros((H, LATENT_DIM), dtype=np.float32)
    q = np.zeros((H, LATENT_DIM), dtype=np.float32)
    for i in range(LATENT_DIM):
        p[i, i] = 1.0
        q[i, i] = 1.0
    p += rng.normal(0, 0.01, p.shape).astype(np.float32)
    q += rng.normal(0, 0.01, q.shape).astype(np.float32)
    alpha = np.float32(0.12)

    losses = []
    for step in range(STEPS):
        z = rng.uniform(-1, 1, LATENT_DIM).astype(np.float32)
        tgt = teacher(z, ROWS, COLS)
        pred = student(z, p, q, float(alpha), ROWS, COLS)
        err = pred - tgt
        loss = float(np.mean(err * err))
        losses.append(loss)

        # Gradients via finite differences on alpha + analytic-ish for P/Q (outer product)
        # dL/dalpha
        u = p @ z
        v = q @ z
        d_alpha = 0.0
        d_u = np.zeros_like(u)
        d_v = np.zeros_like(v)
        scale = 2.0 / (ROWS * COLS)
        for r in range(ROWS):
            for c in range(COLS):
                e = err[r, c]
                ur = u[r % H]
                vc = v[c % H]
                d_alpha += scale * e * ur * vc
                d_u[r % H] += scale * e * float(alpha) * vc
                d_v[c % H] += scale * e * float(alpha) * ur

        # dL/dP = d_u outer z ; dL/dQ = d_v outer z
        g_p = np.outer(d_u, z).astype(np.float32)
        g_q = np.outer(d_v, z).astype(np.float32)

        p -= LR * g_p
        q -= LR * g_q
        alpha = np.float32(alpha - LR * d_alpha)

        if step % 100 == 0 or step == STEPS - 1:
            print(f"step {step:4d}  loss={loss:.6e}  alpha={float(alpha):.4f}")

    out_dir = Path(__file__).resolve().parent / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "toy-v1.json"
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
            f"numpy SGD toy hypernet; seed={SEED}; steps={STEPS}; "
            f"final_loss≈{losses[-1]:.3e}; mimics noise-free outer-product teacher"
        ),
    }
    out.write_text(json.dumps(payload))
    print(f"wrote {out} ({out.stat().st_size} bytes)")
    print(f"loss start={losses[0]:.3e} end={losses[-1]:.3e}")
    if not (losses[-1] < losses[0] or losses[-1] < 1e-4):
        raise SystemExit("training did not improve enough")


if __name__ == "__main__":
    main()
