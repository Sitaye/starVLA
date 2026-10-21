# Copyright 2026 StarVLA Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Forensic statistics over a precomputed OE geometry cache (gate B postmortem).

Read-only, numpy-only, CPU-only. Answers, per suite:
  - is the per-state scale (c_s = mean diag of raw G) heavy-tailed?  -> trace
    arm's per-sample weight w_s = c_s / trace_mean;
  - is the normalized geometry (G/c_s) anisotropic beyond its mean diagonal?
    -> lambda_max(G_hat) / mean(diag) is the true gradient amplification
    factor that grad-clip sees (the batch loss ratio understates it);
  - finite rate, symmetry error, effective ranks, gripper-axis diagnostics
    (the FD step-function miss predicts ~0 grip axis in G-family matrices),
    and scale-vs-position drift within episodes.

Usage: python -m starVLA.analysis.oevla.cache_diag --root /path/to/oe_g_cache
"""

import argparse
import json
from pathlib import Path

import numpy as np

PCTL = (0.1, 1, 5, 25, 50, 75, 95, 99, 99.9)
EIG_SAMPLE = 2000
GRIP_COLS = np.arange(6, 56, 7)  # t*7+6 for t in 0..7


def _pctl(name, x):
    x = x[np.isfinite(x)]
    q = np.percentile(x, PCTL)
    return "  " + name + ": " + " ".join(f"p{p:g}={v:.4g}" for p, v in zip(PCTL, q, strict=True))


def suite_stats(directory: Path, rng: np.random.Generator):
    G = np.load(directory / "G.npy", mmap_mode="r")
    try:
        lengths = {int(k): int(v) for k, v in json.loads((directory / "lengths.json").read_text()).items()}
        meta = json.loads((directory / "meta.json").read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"{directory}: unreadable lengths.json/meta.json ({e})") from e
    chunk_h, action_dim = int(meta["chunk_h"]), int(meta["action_dim"])
    assert G.shape[1] == chunk_h * action_dim, f"{directory}: G d-dim {G.shape[1]} != chunk_h*action_dim"
    trace_mean = float(np.load(directory / "trace_mean.npy"))

    N = G.shape[0]
    rows_per_ep = {tid: max(n - chunk_h + 1, 0) for tid, n in lengths.items()}
    offsets, start = {}, 0
    for tid in sorted(rows_per_ep):
        offsets[tid] = start
        start += rows_per_ep[tid]
    n_expected = start
    assert N == n_expected, f"{directory}: G rows {N} != offsets total {n_expected}"

    # sweep in blocks to keep memory bounded ([block, d, d] f16 -> f32 math)
    block = 4096
    cs = np.empty(N, dtype=np.float64)
    grip_diag = np.empty(N, dtype=np.float64)
    nonfinite = np.zeros(N, dtype=bool)
    sym_err = 0.0
    pos_frac = np.empty(N, dtype=np.float64)
    for lo in range(0, N, block):
        blk = np.asarray(G[lo : lo + block], dtype=np.float64)
        finite_rows = np.isfinite(blk).all(axis=(1, 2))
        nonfinite[lo : lo + len(blk)] = ~finite_rows
        blk = np.nan_to_num(blk, nan=0.0)
        cs[lo : lo + len(blk)] = np.diagonal(blk, axis1=1, axis2=2).mean(axis=1)
        grip_diag[lo : lo + len(blk)] = blk[:, GRIP_COLS, GRIP_COLS].mean(axis=1)
        err = np.abs(blk - np.swapaxes(blk, 1, 2)).max() / max(np.abs(blk).max(), 1e-12)
        sym_err = max(sym_err, float(err))
        # within-episode relative position of each row
        for tid, off in offsets.items():
            n = rows_per_ep[tid]
            a, b = max(off, lo), min(off + n, lo + len(blk))
            if b > a:
                pos_frac[a:b] = (np.arange(a, b) - off) / max(n - 1, 1)

    valid = ~nonfinite
    w = cs[valid] / trace_mean
    n_valid = int(valid.sum())

    # eigen statistics on a random sample of valid rows (upcast f16 -> f32/f64)
    sample_idx = rng.choice(np.flatnonzero(valid), size=min(EIG_SAMPLE, n_valid), replace=False)
    lam_max = np.empty(len(sample_idx))
    eff_rank = np.empty(len(sample_idx))
    for i, r in enumerate(sample_idx):
        row = np.asarray(G[r], dtype=np.float64)
        h = 0.5 * (row + row.T)
        ev = np.linalg.eigvalsh(h)[::-1]
        lam_max[i] = ev[0] / max(cs[r], 1e-12)
        cum = np.cumsum(np.clip(ev, 0, None))
        eff_rank[i] = 1.0 + np.searchsorted(cum / cum[-1], 0.9)

    print(f"\n== {directory.name} ==")
    print(f"  rows={N} valid={n_valid} ({n_valid / N:.4f}) nonfinite={int(nonfinite.sum())} max_sym_err={sym_err:.3g}")
    print(_pctl("c_s (raw mean-diag)", cs[valid]))
    print(_pctl("w_s = c_s/trace_mean", w))
    for th in (2, 5, 10, 30):
        print(f"  frac(w_s > {th}) = {(w > th).mean():.5f}")
    print(f"  trace_mean(c̄) = {trace_mean:.4g}")
    print(_pctl("lam_max(G_hat)/mean-diag  [clip-seen amplification]", lam_max))
    print(_pctl("effective_rank(G_hat) @90%", eff_rank))
    print(_pctl("grip-axis mean diag (raw; FD-step check)", grip_diag[valid]))
    # scale vs position drift (Spearman via ranks)
    from numpy import argsort

    rank_cs = argsort(argsort(cs[valid])).astype(np.float64)
    rank_pos = argsort(argsort(pos_frac[valid])).astype(np.float64)
    rho = np.corrcoef(rank_cs, rank_pos)[0, 1]
    print(f"  spearman(c_s, within-episode position) = {rho:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    dirs = sorted(p for p in Path(args.root).iterdir() if p.is_dir())
    for directory in dirs:
        suite_stats(directory, rng)


if __name__ == "__main__":
    main()
