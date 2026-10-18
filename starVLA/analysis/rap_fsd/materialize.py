# Copyright 2026 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""RAP-FSD Stage A component 4: materialize fold-OOF z targets and mode visualizations.

Outputs (all under --output-dir):
  z.npy          [N, k] float32, one row per valid universe row (same deterministic
                 enumeration as run_stage_a.assemble_universe: sorted shards, stride
                 subsample, NaN-anchor removal). Fold rows use THEIR OWN fold model
                 (mu-hat_F(C) + CCA) so z is out-of-fold; holdout rows use the average
                 of the two fold models (protocol section 8 fold -> holdout aggregation).
                 NOTE (audit H5): the Stage B TRAINING target should be the single
                 full-data projection -- pass --also-full-data to additionally write
                 z_full.npy; gates always used the fold-OOF variant.
  z_rows.npz     per-row metadata: episode_id (str array), suite, label, step, qbar.
  modes/mode_XX.png  per-mode 8x8 loading maps per view (quiver arrows over magnitude).
  README.md      usage + cluster launch commands (8-GPU export, CPU statistics).

The statistics (CCA basis, mu-hat_F) are refit here from the same inputs as
run_stage_a.py, so z.npy is guaranteed consistent with the Stage A results.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import stats
from .run_stage_a import (
    FOLD_LABELS,
    Universe,
    assemble_universe,
    build_pools,
    fit_fold_model,
)
from .split import load_manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--residual-dir", required=True)
    parser.add_argument("--flow-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--holdout-size", type=int, default=2500)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--k", type=int, default=16, help="top-k modes to materialize")
    parser.add_argument("--cca-lambda", type=float, default=stats.CCA_RIDGE_LAMBDA)
    parser.add_argument("--probe-lam", type=float, default=1e-2)
    parser.add_argument("--also-full-data", action="store_true", help="also write z_full.npy (audit H5 target)")
    parser.add_argument("--seed", type=int, default=stats.SEED_DEFAULT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _materialize_universe(args: argparse.Namespace) -> tuple[Universe, dict, dict]:
    manifest = load_manifest(args.manifest)
    from .run_stage_a import discover_shards

    shards = discover_shards(args.residual_dir)
    uni = assemble_universe(manifest, shards, args)
    # fit_fold_model needs the k grid: use the standard grid up to --k so the fold
    # OOF curve stays comparable to run_stage_a's
    args.k_candidates = tuple(k for k in stats.K_CANDIDATES if k <= args.k) or (args.k,)
    args.rrr_lambda = 0.1  # RRR control is not materialized; default matches run_stage_a
    pools = build_pools(uni)
    fold_models = {fold: fit_fold_model(pools[fold], fold, args) for fold in FOLD_LABELS}
    return uni, pools, fold_models


def _oof_z(uni: Universe, pools: dict, fold_models: dict, k: int) -> np.ndarray:
    """Fold-OOF z: each row projected with its own fold's model; holdout = fold average."""
    z = np.zeros((uni.r_a.shape[0], k), dtype=np.float64)
    label_arr = np.asarray(uni.label)
    for fold in FOLD_LABELS:
        m = fold_models[fold]
        sel = label_arr == fold
        f_tilde = uni.f_raw[sel] - m.mu_f.predict(uni.c[sel])
        z[sel] = m.cca.project(f_tilde, k)
    sel_h = label_arr == "holdout"
    if sel_h.any():
        # Holdout rows: average of the two fold models' projections (out-of-model).
        preds = []
        for fold in FOLD_LABELS:
            m = fold_models[fold]
            f_tilde_h = uni.f_raw[sel_h] - m.mu_f.predict(uni.c[sel_h])
            preds.append(m.cca.project(f_tilde_h, k))
        z[sel_h] = np.mean(preds, axis=0)
    return z


def _render_modes(cca, k: int, grid_hw: tuple[int, int], out_dir: Path, flow_shape: tuple) -> None:
    """Per-mode 8x8 loading maps per view: quiver arrows over magnitude heatmap."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_views = flow_shape[1]
    ht, wt = grid_hw
    modes_dir = out_dir / "modes"
    modes_dir.mkdir(parents=True, exist_ok=True)
    for mode in range(k):
        vec = cca.a_flow[:, mode]  # standardized-flow space, [V*Ht*Wt*2]
        load = vec.reshape(n_views, ht, wt, 2)
        fig, axes = plt.subplots(1, n_views, figsize=(4 * n_views, 4.4), squeeze=False)
        for view in range(n_views):
            ax = axes[0, view]
            u, v = load[view, :, :, 0], load[view, :, :, 1]
            mag = np.sqrt(u**2 + v**2)
            ax.imshow(mag, cmap="viridis", origin="upper")
            yy, xx = np.mgrid[0:ht, 0:wt]
            ax.quiver(xx, yy, u, v, color="white", scale=None, scale_units="xy", alpha=0.7)
            ax.set_title(f"mode {mode:02d} view {view}")
            ax.set_xticks([])
            ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(modes_dir / f"mode_{mode:02d}.png", dpi=120)
        plt.close(fig)


def _write_readme(out_dir: Path, k: int, n_rows: int, args: argparse.Namespace, flow_shape: tuple) -> None:
    text = f"""# RAP-FSD Stage A materialized artifacts

Generated by `python -m starVLA.analysis.rap_fsd.materialize` (protocol v1.2, seed {args.seed}).

## Contents
- `z.npy` float32 [N={n_rows}, k={k}]: fold-OOF z targets, row i corresponds to the
  i-th row of the deterministic universe enumeration (sorted shards, stride
  {args.stride}, NaN anchors removed); see `z_rows.npz` for per-row metadata.
- `z_rows.npz`: episode_id / suite / label / step / qbar per row.
{(
    "- `z_full.npy`: single full-data projection target (audit H5) for Stage B training."
    if args.also_full_data
    else "- (z_full.npy not written; pass --also-full-data for the audit H5 training target.)"
)}
- `modes/mode_XX.png`: top-{k} CCA mode loading maps, {flow_shape[1]} view(s), {flow_shape[2]}x{flow_shape[3]} grid
  (quiver arrows over magnitude heatmap, standardized-flow space).

## Pipeline usage
```bash
# 0. episode split (CPU, deterministic, byte-identical across runs)
python -m starVLA.analysis.rap_fsd.split --data-mix libero_all --output stage_a/manifest.json

# 1-2. prediction + context export, sharded over 8 GPUs (one process per GPU)
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i python -m starVLA.analysis.rap_fsd.export_pi0 \\
    --ckpt-path <ckpt> --manifest stage_a/manifest.json \\
    --data-args examples/simBenchmarks/LIBERO/train_files/starvla_cotrain_libero.yaml \\
    --output-dir stage_a/shards/gpu$i --batch-size 64 --stride 1 \\
    --shard-index $i --num-shards 8 --export-evis holdout &
done; wait
# merge the 8 shard dirs into one: stage_a/shards_all/ (one npz per episode)

# 3. statistics + gates (CPU only)
python -m starVLA.analysis.rap_fsd.run_stage_a \\
  --manifest stage_a/manifest.json --residual-dir stage_a/shards_all \\
  --flow-root <flow_cache_root> --output-dir stage_a/results

# 4. materialize z targets + mode visualizations (CPU only)
python -m starVLA.analysis.rap_fsd.materialize \\
  --manifest stage_a/manifest.json --residual-dir stage_a/shards_all \\
  --flow-root <flow_cache_root> --output-dir stage_a/materialized --k {k}
```
"""
    try:
        (out_dir / "README.md").write_text(text)
    except OSError as exc:
        raise RuntimeError(f"failed to write README: {exc}") from exc


def run_materialize(args: argparse.Namespace) -> None:
    uni, pools, fold_models = _materialize_universe(args)
    z = _oof_z(uni, pools, fold_models, args.k)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "z.npy", z.astype(np.float32))
    np.savez(
        out_dir / "z_rows.npz",
        episode_id=np.asarray(uni.episode_id),
        suite=np.asarray(uni.suite),
        label=np.asarray(uni.label),
        step=np.asarray(uni.step, dtype=np.int64),
        qbar=uni.qbar.astype(np.float32),
    )
    if args.also_full_data:
        from .run_stage_a import _full_data_projection

        full = _full_data_projection(pools, args)
        f_all = uni.f_raw - full["mu_f"].predict(uni.c)
        np.save(out_dir / "z_full.npy", full["cca"].project(f_all, args.k).astype(np.float32))
    flow_shape = _infer_flow_shape(args)
    _render_modes(fold_models[FOLD_LABELS[0]].cca, args.k, (flow_shape[2], flow_shape[3]), out_dir, flow_shape)
    _write_readme(out_dir, args.k, int(z.shape[0]), args, flow_shape)
    print(f"[materialize] wrote z.npy {z.shape}, z_rows.npz, modes/, README.md under {out_dir}")


def _infer_flow_shape(args: argparse.Namespace) -> tuple:
    """(N, V, Ht, Wt, 2) from the first dataset's flow.npy header (mmap, no read)."""
    flow_dir = Path(args.flow_root)
    for dataset_dir in sorted(flow_dir.iterdir()):
        flow_path = dataset_dir / "flow.npy"
        if flow_path.exists():
            flow = np.load(flow_path, mmap_mode="r")
            return tuple(int(v) for v in flow.shape)
    raise RuntimeError(f"no flow.npy under {flow_dir}")


def dry_run(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.manifest)
    print(f"[dry-run] manifest protocol: {manifest.get('protocol')}; k={args.k}; stride={args.stride}")
    print("[dry-run] OK (full materialization needs the flow cache + shards on the cluster)")


def main() -> None:
    args = parse_args()
    if args.dry_run:
        dry_run(args)
        return
    run_materialize(args)


if __name__ == "__main__":
    main()
