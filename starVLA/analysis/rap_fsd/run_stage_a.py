# Copyright 2026 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""RAP-FSD Stage A component 3 CLI orchestration.

Reads the split manifest (component 1), the pi0 prediction shards (component 2) and the
RAFT flow cache, runs the frozen pre-registered protocol v1.2 statistics (component 3
core in stats.py) and writes results.json + report.md with the go/no-go decision.

Protocol v1.2 (docs/research/ac_audit_rapfsd_math_20261018.md section 8, frozen):
  K=2 folds + holdout 2500 (sensitivity 2000/3000); delta R2 gate >= 1%; Freedman-Lane
  residual permutation >= 1000 reps, max-over-k; episode-level block bootstrap 95% CI;
  predictability probe (linear + small MLP) >= 50% of Var(z_{1:k*}); median principal
  angle < 45 deg; seed=42; k candidates {4, 8, 16, 32}; stride s=8.

Shard contract (written by export_pi0.py, consumed here):
  one npz per episode named ``<episode_id with "/" -> "__">.npz`` with arrays
  A [T,56] (ground-truth normalized delta actions), pred_A0 [T,56], C [T,D_C] float32,
  steps [T] int64, and a 0-d string ``meta_json`` holding episode_id / suite / task_id /
  c_layout {"lang": [off, n], "evis_pooled": [off, n], "state": [off, n]}.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import stats
from .split import label_lookup, load_manifest

FOLD_LABELS = ("fold_0", "fold_1")
# action layout of libero_franka policies: 8x7 delta qpos normalized (pi0 style)
ACTION_DIM = 56
STRIDE_DEFAULT = 8
KEY_STRIDE = 100_000  # episode key convention: trajectory_id*100_000 + step


@dataclass
class Universe:
    """All valid (stride-subsampled, NaN-filtered) rows across every episode shard."""

    episode_id: list[str] = field(default_factory=list)
    suite: list[str] = field(default_factory=list)
    task: list[str] = field(default_factory=list)
    step: list[int] = field(default_factory=list)
    label: list[str] = field(default_factory=list)
    f_raw: np.ndarray | None = None  # [n, d_F] float64 flattened flow (raw)
    a_raw: np.ndarray | None = None  # [n, 56] ground-truth normalized actions (pre-anchor)
    r_a: np.ndarray | None = None  # [n, 56] anchored residual R_A = A - pred_A0 (v1.2.2)
    c: np.ndarray | None = None  # [n, D_C] context features (C_ext = C union pred_A0)
    qbar: np.ndarray | None = None  # [n] flow cache mean q (NaN when cache absent)
    c_layout: dict | None = None
    nan_excluded: int = 0
    stride_excluded: int = 0
    excluded_action_stats: dict | None = None


@dataclass
class FoldModel:
    """Everything fitted on one fold pool (protocol section 8: fold-internal CV)."""

    fold: str
    mu_a: stats.ConditionalMean
    mu_f: stats.ConditionalMean
    a_tilde: np.ndarray
    f_tilde: np.ndarray
    cca: stats.CCAProjection
    oof_curve: dict[int, float]
    probe_coef: dict[int, np.ndarray]  # linear probe per k, fit on fold rows
    rrr: dict


@dataclass
class RowPool:
    """A labelled subset of universe rows (fold_0 / fold_1 / holdout)."""

    label: str
    f: np.ndarray | None = None  # raw flattened flow
    r_a: np.ndarray | None = None
    a_raw: np.ndarray | None = None
    c: np.ndarray | None = None
    episode_id: np.ndarray | None = None
    suite: np.ndarray | None = None
    task: np.ndarray | None = None
    step: np.ndarray | None = None
    probe_x: np.ndarray | None = None  # pooled E_vis block of C
    qbar: np.ndarray | None = None

    @property
    def n_rows(self) -> int:
        return int(self.r_a.shape[0])


def discover_shards(residual_dir: str | Path) -> dict[str, Path]:
    """Map episode_id -> shard npz path by scanning ``__``-safe names (``__`` = ``/``)."""
    residual_dir = Path(residual_dir)
    shards = {}
    for path in sorted(residual_dir.glob("*.npz")):
        episode_id = path.stem.replace("__", "/")
        if episode_id in shards:
            raise RuntimeError(f"duplicate shard for episode {episode_id}: {path}")
        shards[episode_id] = path
    if not shards:
        raise RuntimeError(f"no shard npz found under {residual_dir}")
    return shards


def _safe_json_loads(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"malformed shard meta_json: {exc}") from exc


def _flow_join(
    flow_dir: Path, dataset_name: str, keys_query: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, ...]]:
    """Vectorised searchsorted join of query keys (traj*100000+step) against keys.npy.

    Returns (F_rows [n_found, d_F] float64, qbar_rows [n_found], found_mask [n_query],
    per-row flow shape). Missing keys come back as NaN rows (invalid anchors). The mmap
    handles flow.npy [N,V,8,8,2] fp16 of possibly tens of GB without a full load.
    """
    flow_path = flow_dir / dataset_name / "flow.npy"
    keys_path = flow_dir / dataset_name / "keys.npy"
    q_path = flow_dir / dataset_name / "q.npy"
    for path in (flow_path, keys_path, q_path):
        if not path.exists():
            raise RuntimeError(f"flow cache file missing: {path}")
    keys = np.load(keys_path)
    flow = np.load(flow_path, mmap_mode="r")
    q = np.load(q_path, mmap_mode="r")
    n_cache = keys.shape[0]
    idx = np.searchsorted(keys, keys_query)
    idx_clipped = np.minimum(idx, max(n_cache - 1, 0))
    found = (idx < n_cache) & (keys[idx_clipped] == keys_query)
    n_found = int(found.sum())
    flow_sel = np.asarray(flow[idx_clipped[found]], dtype=np.float64).reshape(n_found, -1)
    q_sel = np.asarray(q[idx_clipped[found]], dtype=np.float64).reshape(n_found, -1)
    qbar_sel = q_sel.mean(axis=1) if q_sel.size else np.empty(0)
    return flow_sel, qbar_sel, found, tuple(int(v) for v in flow.shape[1:])


def apply_anchor(
    a_all: np.ndarray, pred_all: np.ndarray, c_all: np.ndarray, layout: dict | None
) -> tuple[np.ndarray, np.ndarray, dict | None]:
    """Literal design form (adjudication 2026-10-19 §9, user-approved): R_A = A - pred_A0
    and C_ext = C union pred_A0 (the frozen design requires the reference-policy prediction
    inside the conditioning context for BOTH mu-hat_A and mu-hat_F)."""
    if a_all.shape != pred_all.shape:
        raise RuntimeError(f"pred_A0 shape {pred_all.shape} != A shape {a_all.shape} - shards are inconsistent")
    r_a = a_all - pred_all
    c_ext = np.concatenate([c_all, pred_all.astype(np.float32)], axis=1)
    if layout is not None:
        layout = {**layout, "pred_a0": [int(c_ext.shape[1] - pred_all.shape[1]), int(pred_all.shape[1])]}
    return r_a, c_ext, layout


def assemble_universe(manifest: dict, shards: dict[str, Path], args: argparse.Namespace) -> Universe:
    """Enumerate rows: stride subsample per episode, join the flow cache, filter NaN anchors."""
    uni = Universe()
    labels = label_lookup(manifest, args.holdout_size)
    rows_a, rows_pred, rows_c = [], [], []
    per_dataset: dict[str, list[tuple[int, np.ndarray]]] = {}
    for shard_path in shards.values():
        try:
            with np.load(shard_path, allow_pickle=False) as z:
                steps = np.asarray(z["steps"], dtype=np.int64)
                a_arr = np.asarray(z["A"], dtype=np.float64)
                pred_arr = np.asarray(z["pred_A0"], dtype=np.float64)
                c_arr = np.asarray(z["C"], dtype=np.float32)
                meta_json = str(z["meta_json"])
        except (KeyError, OSError, ValueError) as exc:
            raise RuntimeError(f"failed to read shard {shard_path}: {exc}") from exc
        episode_id = shard_path.stem.replace("__", "/")
        if episode_id not in labels:
            raise RuntimeError(f"shard episode not in manifest: {episode_id}")
        meta = _safe_json_loads(meta_json)
        n_full = steps.shape[0]
        # per-episode phase: mix the trajectory id into the seed so different episodes get
        # different offsets from stats.subsample_rows (each call draws from its own rng)
        ep_seed = args.seed + int(episode_id.rsplit("/", 1)[1])
        keep = stats.subsample_rows({episode_id: np.arange(n_full)}, stride=args.stride, seed=ep_seed)
        uni.stride_excluded += n_full - keep.shape[0]
        local = keep
        start = len(uni.episode_id)
        rows_a.append(a_arr[local])
        rows_pred.append(pred_arr[local])
        rows_c.append(c_arr[local])
        suite = str(meta.get("suite", manifest["episodes"][episode_id].get("suite", "unknown")))
        task = str(meta.get("task_id", manifest["episodes"][episode_id].get("task", "unknown")))
        uni.episode_id.extend([episode_id] * local.shape[0])
        uni.suite.extend([suite] * local.shape[0])
        uni.task.extend([task] * local.shape[0])
        uni.label.extend([labels[episode_id]] * local.shape[0])
        uni.step.extend(steps[local].tolist())
        if uni.c_layout is None:
            uni.c_layout = meta.get("c_layout")
        dataset_name = episode_id.rsplit("/", 1)[0]
        traj = int(episode_id.rsplit("/", 1)[1])
        per_dataset.setdefault(dataset_name, []).append((start, traj * KEY_STRIDE + steps[local].astype(np.int64)))
    if not rows_a:
        raise RuntimeError("universe is empty: no shard rows survived subsampling")
    uni.a_raw = np.concatenate(rows_a, axis=0)
    uni.r_a, uni.c, uni.c_layout = apply_anchor(
        uni.a_raw, np.concatenate(rows_pred, axis=0), np.concatenate(rows_c, axis=0), uni.c_layout
    )
    n_rows = uni.r_a.shape[0]
    if args.flow_root is None:
        raise RuntimeError("--flow-root is required: the gate statistics need the flow targets")
    flow_dir = Path(args.flow_root)
    row_dim = _row_dim_from_datasets(flow_dir, list(per_dataset))
    f_parts = np.full((n_rows, row_dim), np.nan)
    q_parts = np.full(n_rows, np.nan)
    shapes = set()
    for dataset_name, entries in per_dataset.items():
        keys_query = np.concatenate([qk for _, qk in entries])
        f_rows, q_rows, found, shape = _flow_join(flow_dir, dataset_name, keys_query)
        shapes.add(shape)
        pos = 0
        cursor = 0  # _flow_join returns found rows packed in query order
        for start, qk in entries:
            m = qk.shape[0]
            take = found[pos : pos + m]
            hits = np.nonzero(take)[0]
            if hits.size:
                f_parts[start + hits] = f_rows[cursor : cursor + hits.size]
                q_parts[start + hits] = q_rows[cursor : cursor + hits.size]
            cursor += int(hits.size)
            pos += m
    if len(shapes) > 1:
        raise RuntimeError(f"inconsistent flow shapes across datasets: {shapes}")
    valid = stats.nan_anchor_mask(f_parts)  # protocol section 8: NaN anchor exclusion
    uni.nan_excluded = int((~valid).sum())
    uni.excluded_action_stats = _action_stats(uni.a_raw[~valid]) if uni.nan_excluded else None
    uni.f_raw = f_parts
    uni.qbar = q_parts
    _mask_universe(uni, valid)
    return uni


def _action_stats(a: np.ndarray) -> dict:
    return {
        "n_rows": int(a.shape[0]),
        "mean_abs": float(np.abs(a).mean()),
        "mean_norm": float(np.linalg.norm(a, axis=1).mean()),
        "std_norm": float(np.linalg.norm(a, axis=1).std()),
    }


def _row_dim_from_datasets(flow_root: str, dataset_names: list[str]) -> int:
    if not dataset_names:
        raise RuntimeError("no datasets to join against the flow cache")
    flow = np.load(Path(flow_root) / dataset_names[0] / "flow.npy", mmap_mode="r")
    return int(np.prod([int(v) for v in flow.shape[1:]]))


def _mask_universe(uni: Universe, valid: np.ndarray) -> None:
    uni.f_raw = uni.f_raw[valid]
    uni.qbar = uni.qbar[valid]
    uni.a_raw = uni.a_raw[valid]
    uni.r_a = uni.r_a[valid]
    uni.c = uni.c[valid]
    keep = valid.tolist()
    uni.episode_id = [v for v, k in zip(uni.episode_id, keep, strict=True) if k]
    uni.suite = [v for v, k in zip(uni.suite, keep, strict=True) if k]
    uni.task = [v for v, k in zip(uni.task, keep, strict=True) if k]
    uni.label = [v for v, k in zip(uni.label, keep, strict=True) if k]
    uni.step = [v for v, k in zip(uni.step, keep, strict=True) if k]


def probe_features_from_c(c: np.ndarray, c_layout: dict | None) -> np.ndarray:
    """Frozen probe features: per-view pooled E_vis block of C (protocol section 8 G3)."""
    if c_layout and "evis_pooled" in c_layout:
        off, n_cols = c_layout["evis_pooled"]
        return c[:, off : off + n_cols]
    raise ValueError(f"c_layout is missing the evis_pooled block: {c_layout}")


def build_pools(uni: Universe) -> dict[str, RowPool]:
    """Slice the universe into fold/holdout pools.

    Protocol section 8: mu-hat_A(C) and mu-hat_F(C) are fitted on EACH FOLD POOL
    independently (fold-internal lambda CV) so holdout residuals are out-of-fold; the
    holdout pool is residualised by BOTH fold models at evaluation time.
    """
    episode_id = np.asarray(uni.episode_id)
    suite = np.asarray(uni.suite)
    task = np.asarray(uni.task)
    step = np.asarray(uni.step, dtype=np.int64)
    label_arr = np.asarray(uni.label)
    pools = {}
    for label in (*FOLD_LABELS, "holdout"):
        mask = label_arr == label
        c = np.asarray(uni.c, dtype=np.float64)[mask]
        pool = RowPool(
            label=label,
            r_a=np.asarray(uni.r_a, dtype=np.float64)[mask],
            a_raw=np.asarray(uni.a_raw, dtype=np.float64)[mask],
            c=c,
            episode_id=episode_id[mask],
            suite=suite[mask],
            task=task[mask],
            step=step[mask],
            probe_x=probe_features_from_c(c, uni.c_layout).astype(np.float32),
            qbar=np.asarray(uni.qbar, dtype=np.float64)[mask],
        )
        if uni.f_raw is not None:
            pool.f = np.asarray(uni.f_raw, dtype=np.float64)[mask]
        pools[label] = pool
    return pools


def fit_fold_model(pool: RowPool, fold: str, args: argparse.Namespace) -> FoldModel:
    """Fit the full per-fold pipeline (protocol section 8 fold-internal procedure)."""
    mu_a = stats.fit_conditional_mean(pool.c, pool.r_a, seed=args.seed)
    mu_f = stats.fit_conditional_mean(pool.c, pool.f, seed=args.seed)
    a_tilde = stats.residualize(pool.r_a, pool.c, mu_a)
    f_tilde = stats.residualize(pool.f, pool.c, mu_f)
    cca = stats.fit_cca(f_tilde, a_tilde, ridge_lambda=args.cca_lambda)
    max_f = min(cca.a_flow.shape[1], int(max(args.k_candidates)))
    z_fold = cca.project(f_tilde, max_f)
    oof_curve = stats.delta_r2_oof_curve(
        z_fold, a_tilde, k_candidates=args.k_candidates, lam=args.probe_lam, seed=args.seed
    )
    probe_coef = {}
    for k in oof_curve:
        probe_coef[k] = stats.fit_linear_probe(z_fold[:, :k], a_tilde, lam=args.probe_lam)
    rrr = stats.fit_rrr(f_tilde, a_tilde, lam=args.rrr_lambda)
    return FoldModel(
        fold=fold,
        mu_a=mu_a,
        mu_f=mu_f,
        a_tilde=a_tilde,
        f_tilde=f_tilde,
        cca=cca,
        oof_curve=oof_curve,
        probe_coef=probe_coef,
        rrr=rrr,
    )


def select_k_star(fold_models: dict[str, FoldModel], n_holdout_rows: int) -> tuple[int, float, dict]:
    """k* = min k with mean fold-OOF delta R2(k) >= max - SE(MDE at argmax k).

    v1.3 amendment (adjudication 2026-10-19 §5): v1.2 froze the k candidate set but left the
    curve-to-k* rule as an unregistered argmax, which selects unstable modes on a plateau.
    """
    curves = [m.oof_curve for m in fold_models.values()]
    common = sorted(set.intersection(*(set(c) for c in curves)))
    mean_curve = {k: float(np.mean([c[k] for c in curves])) for k in common}
    if not mean_curve:
        raise RuntimeError("no common k across folds")
    k_max = max(mean_curve, key=mean_curve.get)
    thr = float(mean_curve[k_max] - stats.mde_standard_error(n_holdout_rows, int(k_max)))
    plateau = {k: v for k, v in mean_curve.items() if v >= thr}
    k_star = min(plateau)
    info = {
        "rule": "v1.3 plateau: min k within one SE(MDE) of the curve maximum",
        "k_max": int(k_max),
        "threshold": thr,
        "plateau_ks": sorted(plateau),
    }
    return int(k_star), float(mean_curve[k_star]), info


def holdout_fold_arrays(fold_models: dict[str, FoldModel], holdout: RowPool) -> list[tuple[np.ndarray, np.ndarray]]:
    """Out-of-fold holdout residuals per fold: (A_tilde_f, F_tilde_f)."""
    arrays = []
    for fold in FOLD_LABELS:
        m = fold_models[fold]
        a_tilde = holdout.r_a - m.mu_a.predict(holdout.c)
        f_tilde = holdout.f - m.mu_f.predict(holdout.c)
        arrays.append((a_tilde, f_tilde))
    return arrays


def holdout_delta_r2_curve(
    fold_models: dict[str, FoldModel],
    holdout: RowPool,
) -> tuple[dict[int, float], np.ndarray]:
    """Pooled holdout delta R2(k): each fold's probe scores its own out-of-fold residuals;
    the two fold predictions AND the residual targets are averaged before the statistic
    (protocol section 8 fold -> holdout aggregation). Returns (curve, a_tilde_avg)."""
    arrays = holdout_fold_arrays(fold_models, holdout)
    a_tilde_avg = np.mean([a_t for a_t, _ in arrays], axis=0)
    curve = {}
    for k in fold_models[FOLD_LABELS[0]].probe_coef:
        preds = [
            fold_models[fold].cca.project(f_t, k) @ fold_models[fold].probe_coef[k]
            for (_, f_t), fold in zip(arrays, FOLD_LABELS, strict=True)
        ]
        curve[int(k)] = stats.r2_explained(a_tilde_avg, np.mean(preds, axis=0))
    return curve, a_tilde_avg


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--manifest", required=True, help="split manifest JSON from split.py")
    parser.add_argument("--residual-dir", required=True, help="directory of export_pi0.py episode npz shards")
    parser.add_argument("--flow-root", default=None, help="RAFT flow cache root (per-dataset subdirs)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--holdout-size", type=int, default=None,
        help="primary holdout tier to read gates at; default = manifest's primary_holdout_size "
        "(protocol v1.2 amendment v1.2.1 rescales tiers to the actual dataset, e.g. 300/400/500)",
    )
    parser.add_argument("--stride", type=int, default=STRIDE_DEFAULT)
    parser.add_argument("--k-candidates", default="4,8,16,32")
    parser.add_argument("--n-perms", type=int, default=stats.N_PERMS_MIN)
    parser.add_argument("--n-bootstrap", type=int, default=stats.N_BOOTSTRAP_DEFAULT)
    parser.add_argument("--cca-lambda", type=float, default=stats.CCA_RIDGE_LAMBDA)
    parser.add_argument("--cca-lambdas-sensitivity", default="0.01,0.05,0.1,0.5,1.0")
    parser.add_argument("--probe-lam", type=float, default=1e-2)
    parser.add_argument("--rrr-lambda", type=float, default=0.1)
    parser.add_argument("--qbar-threshold", type=float, default=0.5)
    parser.add_argument("--mlp-epochs", type=int, default=40)
    parser.add_argument("--mlp-train-rows", type=int, default=8000)
    parser.add_argument(
        "--probe-pool",
        type=int,
        default=4,
        help="spatial avg-pool factor for the G3 token-level E_vis probe (MLP uses 2x this)",
    )
    parser.add_argument("--seed", type=int, default=stats.SEED_DEFAULT)
    parser.add_argument("--dry-run", action="store_true", help="self-check inputs and exit")
    return parser.parse_args(argv)


def _parse_ints(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(",") if part.strip())


def _parse_floats(text: str) -> tuple[float, ...]:
    return tuple(float(part) for part in text.split(",") if part.strip())


def dry_run(args: argparse.Namespace) -> None:
    """Self-check without heavy compute: manifest, shards, flow cache presence."""
    manifest = load_manifest(args.manifest)
    shards = discover_shards(args.residual_dir)
    labels = label_lookup(manifest, args.holdout_size)
    missing = [ep for ep in shards if ep not in labels]
    print(f"[dry-run] manifest episodes: {len(manifest['episodes'])}")
    print(f"[dry-run] shards: {len(shards)}; shards missing from manifest: {len(missing)}")
    counts = {lab: sum(1 for v in labels.values() if v == lab) for lab in (*FOLD_LABELS, "holdout")}
    print(f"[dry-run] label counts (holdout={args.holdout_size}): {counts}")
    if args.flow_root is not None:
        datasets = sorted({ep.rsplit("/", 1)[0] for ep in shards})
        for dataset in datasets:
            for name in ("flow.npy", "keys.npy", "q.npy"):
                path = Path(args.flow_root) / dataset / name
                if not path.exists():
                    raise RuntimeError(f"flow cache file missing: {path}")
        print(f"[dry-run] flow cache datasets present: {datasets}")
    print("[dry-run] OK")


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"not JSON serializable: {type(obj)}")


def main() -> None:
    args = parse_args()
    args.k_candidates = _parse_ints(args.k_candidates)
    args.cca_lambdas_sensitivity = _parse_floats(args.cca_lambdas_sensitivity)
    if args.dry_run:
        dry_run(args)
        return
    results = run_stage_a(args)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.json"
    report_path = out_dir / "report.md"
    try:
        results_path.write_text(json.dumps(results, indent=2, sort_keys=True, default=_json_default) + "\n")
        report_path.write_text(render_report(results))
    except (OSError, TypeError) as exc:
        raise RuntimeError(f"failed to write Stage A outputs: {exc}") from exc
    print(f"[rap_fsd] wrote {results_path} and {report_path}")
    print(f"[rap_fsd] verdict: {results['verdict']['decision']}")


def run_stage_a(args: argparse.Namespace) -> dict:
    """Execute the full frozen protocol; returns the results.json payload."""
    manifest = load_manifest(args.manifest)
    if args.holdout_size is None:
        args.holdout_size = int(manifest["primary_holdout_size"])
    if args.holdout_size not in manifest["holdout_sizes"]:
        raise SystemExit(
            f"--holdout-size {args.holdout_size} not recorded in manifest "
            f"(tiers: {manifest['holdout_sizes']}, primary: {manifest['primary_holdout_size']})"
        )
    shards = discover_shards(args.residual_dir)
    uni = assemble_universe(manifest, shards, args)
    pools = build_pools(uni)
    for pool in pools.values():
        if pool.a_raw is None:
            raise RuntimeError("universe is missing a_raw: anchor (R_A = A - pred_A0) was not applied")
    fold_models = {fold: fit_fold_model(pools[fold], fold, args) for fold in FOLD_LABELS}
    k_star, fold_oof_star, k_star_rule = select_k_star(fold_models, pools["holdout"].n_rows)
    holdout = pools["holdout"]
    curve_h, a_tilde_h = holdout_delta_r2_curve(fold_models, holdout)
    delta_r2_star = curve_h[int(k_star)]
    avg_a_hat = _holdout_avg_predictions(fold_models, holdout, k_star)
    boot = stats.block_bootstrap_delta_r2(
        a_tilde_h, avg_a_hat, holdout.episode_id, n_reps=args.n_bootstrap, seed=args.seed
    )
    null = stats.freedman_lane_null(
        _holdout_f_tilde_avg(fold_models, holdout),
        a_tilde_h,
        holdout.c,
        holdout.episode_id,
        k_candidates=args.k_candidates,
        n_perms=args.n_perms,
        cca_ridge_lambda=args.cca_lambda,
        probe_lam=args.probe_lam,
        seed=args.seed,
    )
    probes = _predictability_probes(pools, uni, k_star, args)
    positive_control = _positive_control(pools, uni, fold_models, k_star, args)
    angle_stats = _fold_principal_angles(fold_models, k_star)
    gates = stats.evaluate_gates(
        delta_r2_star=delta_r2_star,
        delta_r2_star_ci_low=boot["ci_low"],
        delta_r2_k_star=k_star,
        null_max_over_k=null["max_over_k"],
        probe_r2_best=max(probes["linear"]["r2"], probes["mlp"]["r2"]),
        median_principal_angle_deg=angle_stats["median_deg"],
    )
    results = {
        "protocol": "rap_fsd/stage_a/v1.2.2",
        "config": _config_snapshot(args),
        "k_star_rule": k_star_rule,
        "data": _data_report(manifest, uni, pools, args),
        "k_star": k_star,
        "fold_oof_curves": {f: {str(k): v for k, v in m.oof_curve.items()} for f, m in fold_models.items()},
        "fold_oof_star": fold_oof_star,
        "holdout_delta_r2_curve": {str(k): v for k, v in curve_h.items()},
        "bootstrap": {k: v for k, v in boot.items() if k != "stats"},
        "mde": {
            "holdout_rows": holdout.n_rows,
            "k_star": k_star,
            "se_delta_r2_k_star": stats.mde_standard_error(holdout.n_rows, k_star),
            "note": (
                "protocol v1.2 section 8 G2 / oracle M4: SE(delta R2) approx sqrt(2k)/n_eff; "
                "reported next to the frozen 1% threshold (MDE-threshold consistency)"
            ),
        },
        "null": {
            "n_perms": int(null["max_over_k"].shape[0]),
            "percentiles": {str(p): float(np.percentile(null["max_over_k"], p)) for p in (50, 90, 95, 99)},
            "per_k_95": {str(k): float(np.percentile(v, 95)) for k, v in null["per_k"].items()},
        },
        "probes": probes,
        "positive_control": positive_control,
        "fold_principal_angle": angle_stats,
        "controls": _controls(fold_models, pools, holdout, k_star, args, a_tilde_h),
        "sensitivity": _sensitivity(manifest, uni, args),
        "cca_lambda_sensitivity": _cca_lambda_sensitivity(fold_models, holdout, k_star, args),
        "per_suite_breakdown": per_suite_breakdown(pools, fold_models, holdout, k_star, args),
        "per_task_breakdown": per_task_breakdown(pools, fold_models, holdout, k_star, args),
        "gate_semantics": _gate_semantics(fold_models, pools, args),
        "gates": gates,
        "verdict": _verdict(gates, k_star),
    }
    return results


def _config_snapshot(args: argparse.Namespace) -> dict:
    """Frozen configuration echo (threshold provenance: protocol v1.2, 2026-10-18)."""
    return {
        "holdout_size_primary": args.holdout_size,
        "stride": args.stride,
        "k_candidates": list(args.k_candidates),
        "n_perms": args.n_perms,
        "n_bootstrap": args.n_bootstrap,
        "cca_lambda": args.cca_lambda,
        "anchor": {
            "form": "literal design: R_A = A - pred_A0; C_ext = C union pred_A0 for mu_A and mu_F",
            "decision": "stage_a_optimistic_adjudication_20261019.md section 9, user-approved 2026-10-19",
        },
        "cca_lambdas_sensitivity": list(args.cca_lambdas_sensitivity),
        "probe_lam": args.probe_lam,
        "rrr_lambda": args.rrr_lambda,
        "qbar_threshold": args.qbar_threshold,
        "mlp_epochs": args.mlp_epochs,
        "mlp_train_rows": args.mlp_train_rows,
        "probe_pool": args.probe_pool,
        "seed": args.seed,
        "threshold_provenance": {
            "delta_r2_min": "protocol v1.2 section 8 G2: 1% of Var(A_tilde)",
            "probe_r2_min": "protocol v1.2 section 8 G3: 50% of Var(z)",
            "median_principal_angle_max_deg": "protocol v1.2 section 8 G4: 45 deg",
            "n_perms_min": "protocol v1.2 section 8 G1: >= 1000 Freedman-Lane reps",
            "null_statistic": "max-over-k (audit section 2.2 M4 multivariate selection)",
        },
    }


def _holdout_avg_predictions(fold_models: dict, holdout: RowPool, k_star: int) -> np.ndarray:
    preds = []
    for fold in FOLD_LABELS:
        m = fold_models[fold]
        f_tilde = holdout.f - m.mu_f.predict(holdout.c)
        z_k = m.cca.project(f_tilde, k_star)
        preds.append(z_k @ m.probe_coef[k_star])
    return np.mean(preds, axis=0)


def _holdout_f_tilde_avg(fold_models: dict, holdout: RowPool) -> np.ndarray:
    f_tildes = [holdout.f - fold_models[fold].mu_f.predict(holdout.c) for fold in FOLD_LABELS]
    return np.mean(f_tildes, axis=0)


def _load_holdout_evis(pools: dict, args: argparse.Namespace) -> dict | None:
    """Load token-level E_vis for holdout rows from the residual shards (protocol section 8
    G3: the probe reads the current VLA tokens E_theta(X), not the pooled block that is
    part of C). Returns pooled linear/MLP feature matrices aligned to a kept-row mask.

    Shard evis layout: [n_exported, V, H, W, D] fp16 with evis_rows = exported steps;
    pool rows whose step was not exported are dropped (count reported). Shards are read
    once each (the evis payload is large; per-row np.load would re-open the archive for
    every transition row)."""
    holdout = pools["holdout"]
    shard_paths = {p.stem: p for p in sorted(Path(args.residual_dir).glob("*.npz"))}
    rows_by_episode: dict[str, list[int]] = {}
    for row, episode_id in enumerate(holdout.episode_id.tolist()):
        rows_by_episode.setdefault(str(episode_id), []).append(row)
    linear_rows, mlp_rows, keep_idx, missing = [], [], [], 0
    for episode_id, rows in sorted(rows_by_episode.items()):
        shard = shard_paths.get(episode_id.replace("/", "__"))
        if shard is None:
            missing += len(rows)
            continue
        try:
            with np.load(shard) as z:
                if "evis" not in z or "evis_rows" not in z:
                    missing += len(rows)
                    continue
                evis_rows = np.asarray(z["evis_rows"], dtype=np.int64)
                evis = np.asarray(z["evis"], dtype=np.float32)
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"failed to read evis from {shard}: {exc}") from exc
        steps = holdout.step[np.asarray(rows, dtype=np.int64)]
        pos = np.searchsorted(evis_rows, steps)
        pos_clipped = np.minimum(pos, evis_rows.shape[0] - 1)
        hit = (pos < evis_rows.shape[0]) & (evis_rows[pos_clipped] == steps)
        missing += int((~hit).sum())
        if hit.any():
            evis_hit = evis[pos_clipped[hit]]
            linear_rows.extend(stats.pool_tokens(evis_hit, args.probe_pool))
            mlp_rows.extend(stats.pool_tokens(evis_hit, 2 * args.probe_pool))
            keep_idx.extend(np.asarray(rows)[hit].tolist())
    if not linear_rows:
        return None
    return {
        "linear": np.stack(linear_rows).astype(np.float32),
        "mlp": np.stack(mlp_rows).astype(np.float32),
        "row_idx": np.asarray(keep_idx, dtype=np.int64),
        "n_missing": missing,
    }


def _predictability_probes(pools: dict, uni: Universe, k_star: int, args: argparse.Namespace) -> dict:
    """G3 probes: target z_{1:k*} from the SINGLE full-data projection (audit H5).

    Protocol section 8 G3 / audit oracle L1: two tiers (linear + small MLP) on frozen
    token-level E_vis. E_vis is exported for holdout episodes only, so probes are fit
    and evaluated INSIDE the holdout pool with row-level OOF / a seeded train-eval split
    (z targets are fixed by the frozen full-data projection, so no gate leakage arises).
    Direction-optimism declaration (audit H5 tension): the full-data projection is fitted
    on ALL rows including the holdout, so the z targets embed a mild in-sample component
    in the projection direction; probe fitting itself is out-of-fold and this is declared
    in the report.
    Fallback when E_vis was not exported: pooled E_vis block of C (weaker by construction:
    the pooled block is inside C, hence largely conditioned away - reported as such)."""
    full = _full_data_projection(pools, args)
    holdout = pools["holdout"]
    z_holdout = full["cca"].project(holdout.f - full["mu_f"].predict(holdout.c), k_star)
    evis = _load_holdout_evis(pools, args)
    if evis is None:
        z_fit = np.concatenate(
            [full["cca"].project(pools[f].f - full["mu_f"].predict(pools[f].c), k_star) for f in FOLD_LABELS],
            axis=0,
        )
        x_fit = np.concatenate([pools[f].probe_x for f in FOLD_LABELS], axis=0)
        idx = np.arange(x_fit.shape[0])
        if x_fit.shape[0] > args.mlp_train_rows:
            rng = np.random.default_rng(args.seed)
            idx = np.sort(rng.choice(x_fit.shape[0], size=args.mlp_train_rows, replace=False))
        linear = stats.linear_probe_r2(x_fit, z_fit, holdout.probe_x, z_holdout, seed=args.seed)
        mlp = stats.mlp_probe_r2(
            x_fit[idx], z_fit[idx], holdout.probe_x, z_holdout, seed=args.seed, epochs=args.mlp_epochs
        )
        return {
            "target": f"z_1:{k_star} from single full-data projection (audit H5)",
            "features": "pooled E_vis block of C (FALLBACK - token-level E_vis not exported; see protocol note)",
            "n_train": int(x_fit.shape[0]),
            "n_eval": int(holdout.probe_x.shape[0]),
            "n_rows_without_evis": 0,
            "linear": {k: v for k, v in linear.items() if k != "cv_curve"},
            "mlp": dict(mlp),
            # audit: G3 probe must read current VLA tokens; pooled block is a documented fallback
            "protocol_note": "export with --export-evis holdout: G3 must read token-level E_vis",
        }
    sel = evis["row_idx"]
    z_sel = z_holdout[sel]
    linear = stats.token_probe_r2(evis["linear"], z_sel, evis["linear"], z_sel, seed=args.seed)
    rng = np.random.default_rng(args.seed + 7)
    perm = rng.permutation(sel.shape[0])
    n_train = int(0.8 * sel.shape[0])
    tr, ev = perm[:n_train], perm[n_train:]
    mlp_tr = min(n_train, args.mlp_train_rows)
    tr_sub = np.sort(rng.choice(tr, size=mlp_tr, replace=False)) if n_train > mlp_tr else tr
    mlp = stats.mlp_probe_r2(
        evis["mlp"][tr_sub], z_sel[tr_sub], evis["mlp"][ev], z_sel[ev], seed=args.seed, epochs=args.mlp_epochs
    )
    return {
        "target": f"z_1:{k_star} from single full-data projection (audit H5)",
        "features": (
            f"token-level E_vis, spatial pool {args.probe_pool}x{args.probe_pool} (linear), "
            f"{2 * args.probe_pool}x{2 * args.probe_pool} (MLP); dual-form ridge + OOF inside holdout"
        ),
        "n_train": int(tr.shape[0]),
        "n_eval": int(ev.shape[0]),
        "n_rows_without_evis": int(evis["n_missing"]),
        "linear": {k: v for k, v in linear.items() if k != "oof_curve"},
        "mlp": dict(mlp),
        "protocol_note": "token-level E_vis per audit (probe reads current VLA tokens, never A_hat_0)",
    }


def _positive_control(
    pools: dict, uni: Universe, fold_models: dict, k_star: int, args: argparse.Namespace
) -> dict:
    """Instrument certification for G3 (adjudication 2026-10-19 §4, informational only).

    Asks whether the SAME probe features can read targets that are known to live in the
    observation chain: (a) raw ground-truth actions A, (b) top-k* PCA of fold-averaged
    OOF F_tilde (in-sample PCA, upper bound). No gate semantics - it separates
    "probe instrument broken" from "z genuinely undecodable".
    """
    holdout = pools["holdout"]
    evis = _load_holdout_evis(pools, args)
    x = evis["linear"] if evis is not None else holdout.probe_x.astype(np.float64)
    rng = np.random.default_rng(args.seed + 7)
    perm = rng.permutation(x.shape[0])
    n_train = int(0.8 * x.shape[0])
    tr, ev = perm[:n_train], perm[n_train:]
    ctrl_a = stats.linear_probe_r2(x[tr], holdout.a_raw[tr], x[ev], holdout.a_raw[ev], seed=args.seed)
    f_t_avg = _holdout_f_tilde_avg(fold_models, holdout)
    f_c = f_t_avg - f_t_avg.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(f_c, full_matrices=False)
    pc_t = u[:, :k_star] * s[:k_star]
    ctrl_b = stats.linear_probe_r2(x[tr], pc_t[tr], x[ev], pc_t[ev], seed=args.seed)
    return {
        "target_a": "raw ground-truth actions A (56 dims, pre-anchor)",
        "target_b": f"top-{k_star} PCA of fold-averaged OOF F_tilde (in-sample PCA, upper bound)",
        "control_a_r2": ctrl_a["r2"],
        "control_b_r2": ctrl_b["r2"],
        "features": "token-level E_vis (linear tier)" if evis is not None else "pooled E_vis fallback",
        "instrument_certified": bool(ctrl_a["r2"] > 0.05),
        "note": "informational only - certifies the probe instrument, no gate semantics",
    }


def _full_data_projection(pools: dict, args: argparse.Namespace) -> dict:
    """Single full-data projection (audit H5): mu-hat_A / mu-hat_F / CCA fitted once on all rows."""
    c_all = np.concatenate([pools[f].c for f in (*FOLD_LABELS, "holdout")], axis=0)
    r_all = np.concatenate([pools[f].r_a for f in (*FOLD_LABELS, "holdout")], axis=0)
    f_all = np.concatenate([pools[f].f for f in (*FOLD_LABELS, "holdout")], axis=0)
    mu_a = stats.fit_conditional_mean(c_all, r_all, seed=args.seed)
    mu_f = stats.fit_conditional_mean(c_all, f_all, seed=args.seed)
    f_tilde = stats.residualize(f_all, c_all, mu_f)
    a_tilde = stats.residualize(r_all, c_all, mu_a)
    cca = stats.fit_cca(f_tilde, a_tilde, ridge_lambda=args.cca_lambda)
    return {"mu_a": mu_a, "mu_f": mu_f, "cca": cca}


def _fold_principal_angles(fold_models: dict, k_star: int) -> dict:
    angles = stats.principal_angles_deg(
        fold_models[FOLD_LABELS[0]].cca.a_flow[:, :k_star],
        fold_models[FOLD_LABELS[1]].cca.a_flow[:, :k_star],
    )
    return {"median_deg": float(np.median(angles)), "all_deg": angles.tolist(), "k_star": int(k_star)}


def _controls(
    fold_models: dict,
    pools: dict,
    holdout: RowPool,
    k_star: int,
    args: argparse.Namespace,
    a_tilde_avg: np.ndarray,
) -> dict:
    """RRR / Random-subspace / reversed-time controls (protocol section 8 controls).

    a_tilde_avg is the main arm's fold-averaged holdout residual target (identical for
    every control arm: reversal/random-basis touch only the flow side, never mu_A/A)."""
    return {
        "rrr": {"delta_r2_k_star": _rrr_holdout_delta_r2(fold_models, holdout, k_star, a_tilde_avg)},
        "random_subspace": {
            "delta_r2_k_star": _random_subspace_delta_r2(fold_models, holdout, k_star, args, a_tilde_avg)
        },
        "reversed_time": {
            "delta_r2_k_star": _reversed_time_delta_r2(
                fold_models, pools, holdout, k_star, args, a_tilde_avg
            )
        },
        "note": "rank ladder / basis fitted per fold; holdout evaluation, fold-averaged predictions",
    }


def _rrr_holdout_delta_r2(
    fold_models: dict, holdout: RowPool, k_star: int, a_tilde_avg: np.ndarray
) -> float:
    """RRR rank-k* predictor fitted per fold, evaluated on holdout rows, fold-averaged."""
    preds = []
    for fold in FOLD_LABELS:
        m = fold_models[fold]
        f_tilde = holdout.f - m.mu_f.predict(holdout.c)
        k_eff = min(k_star, int(m.rrr["s"].shape[0]))
        w_k = m.rrr["u"][:, :k_eff] @ np.diag(m.rrr["s"][:k_eff]) @ m.rrr["vt"][:k_eff]
        preds.append(f_tilde @ w_k)
    return stats.r2_explained(a_tilde_avg, np.mean(preds, axis=0))


def _random_subspace_delta_r2(
    fold_models: dict, holdout: RowPool, k_star: int, args: argparse.Namespace, a_tilde_avg: np.ndarray
) -> float:
    """Random-subspace control: random orthonormal flow basis of the same dim (audit section 3)."""
    preds = []
    for i, fold in enumerate(FOLD_LABELS):
        m = fold_models[fold]
        f_tilde = holdout.f - m.mu_f.predict(holdout.c)
        basis = stats.random_orthogonal_basis(m.cca.mean_f.shape[0], k_star, seed=args.seed + 100 + i)
        z_rand_fold = (m.f_tilde - m.cca.mean_f) / m.cca.std_f @ basis
        coef = stats.fit_linear_probe(z_rand_fold, m.a_tilde, lam=args.probe_lam)
        z_rand_h = (f_tilde - m.cca.mean_f) / m.cca.std_f @ basis
        preds.append(z_rand_h @ coef)
    return stats.r2_explained(a_tilde_avg, np.mean(preds, axis=0))


def _reversed_time_delta_r2(
    fold_models: dict, pools: dict, holdout: RowPool, k_star: int, args: argparse.Namespace, a_tilde_avg: np.ndarray
) -> float:
    """Reversed-time control: the FULL pipeline is refit on flow rows reversed within each
    episode (step t's flow paired with step T-t's action; C / A_tilde keep their original
    order) - mu-hat_F + CCA + linear probe are all refit on the reversed arm, then
    evaluated on the reversed holdout residuals, fold-averaged. A sign flip alone would
    be an identity control (CCA loadings are sign-invariant, the probe absorbs the sign)."""
    preds = []
    f_h_rev = stats.reversed_time_flow(holdout.f, holdout.episode_id)
    for fold in FOLD_LABELS:
        pool = pools[fold]
        m = fold_models[fold]
        f_rev = stats.reversed_time_flow(pool.f, pool.episode_id)
        mu_f_rev = stats.fit_conditional_mean(pool.c, f_rev, seed=args.seed)
        f_tilde_rev = stats.residualize(f_rev, pool.c, mu_f_rev)
        cca_rev = stats.fit_cca(f_tilde_rev, m.a_tilde, ridge_lambda=args.cca_lambda)
        z_rev_fold = cca_rev.project(f_tilde_rev, k_star)
        coef = stats.fit_linear_probe(z_rev_fold, m.a_tilde, lam=args.probe_lam)
        f_tilde_h_rev = f_h_rev - mu_f_rev.predict(holdout.c)
        z_rev_h = cca_rev.project(f_tilde_h_rev, k_star)
        preds.append(z_rev_h @ coef)
    return stats.r2_explained(a_tilde_avg, np.mean(preds, axis=0))


def _sensitivity(manifest: dict, uni: Universe, args: argparse.Namespace) -> dict:
    """Sensitivity tiers: q-bar > 0.5 filter and holdout sizes 2000/2500/3000 (protocol section 8)."""
    return {
        "qbar": _qbar_sensitivity(uni, args),
        "holdout_sizes": _holdout_size_sensitivity(manifest, uni, args),
    }


def _qbar_sensitivity(uni: Universe, args: argparse.Namespace) -> dict:
    """q-bar sensitivity: refit the pipeline on rows whose flow cache q mean exceeds the threshold."""
    qbar = np.asarray(uni.qbar)
    if not np.isfinite(qbar).any():
        return {"status": "unavailable", "reason": "flow cache q.npy not joined"}
    mask = qbar > args.qbar_threshold
    frac = float(mask.mean())
    if int(mask.sum()) < 100:
        return {"status": "skipped", "kept_fraction": frac, "reason": "too few rows above threshold"}
    q_uni = _copy_universe_masked(uni, mask)
    q_pools = build_pools(q_uni)
    q_folds = {f: fit_fold_model(q_pools[f], f, args) for f in FOLD_LABELS}
    k_star_q, _, _ = select_k_star(q_folds, q_pools["holdout"].n_rows)
    curve_q, a_tilde_q = holdout_delta_r2_curve(q_folds, q_pools["holdout"])
    pred_q = _holdout_avg_predictions(q_folds, q_pools["holdout"], k_star_q)
    boot_q = stats.block_bootstrap_delta_r2(
        a_tilde_q, pred_q, q_pools["holdout"].episode_id, n_reps=args.n_bootstrap, seed=args.seed
    )
    return {
        "status": "ok",
        "threshold": args.qbar_threshold,
        "kept_fraction": frac,
        "k_star": k_star_q,
        "delta_r2_k_star": curve_q[int(k_star_q)],
        "bootstrap_ci_low": boot_q["ci_low"],
        "bootstrap_ci_high": boot_q["ci_high"],
    }


def _copy_universe_masked(uni: Universe, mask: np.ndarray) -> Universe:
    return Universe(
        episode_id=[e for e, v in zip(uni.episode_id, mask.tolist(), strict=True) if v],
        suite=[s for s, v in zip(uni.suite, mask.tolist(), strict=True) if v],
        task=[t for t, v in zip(uni.task, mask.tolist(), strict=True) if v],
        step=[s for s, v in zip(uni.step, mask.tolist(), strict=True) if v],
        label=[lab for lab, v in zip(uni.label, mask.tolist(), strict=True) if v],
        f_raw=uni.f_raw[mask] if uni.f_raw is not None else None,
        a_raw=uni.a_raw[mask] if uni.a_raw is not None else None,
        r_a=uni.r_a[mask],
        c=uni.c[mask],
        qbar=uni.qbar[mask],
        c_layout=uni.c_layout,
    )


def _holdout_size_sensitivity(manifest: dict, uni: Universe, args: argparse.Namespace) -> list[dict]:
    """Holdout 2000/2500/3000 tiers. Fold labels are unchanged across sizes (all sizes are
    even, so the sorted-rank parity rule in split.py is size-invariant); only the holdout
    row set changes (nested 2000 subset 2500 subset 3000). G1 permutation null is a
    PRIMARY-holdout gate only - sensitivity tiers report delta R2 + bootstrap CI."""
    # tiers actually present in the manifest: every non-primary tier is a sensitivity
    # read (nested subsets recorded by split.py; e.g. amendment v1.2.1: 300/500 around
    # primary 400). G1 permutation null is a PRIMARY-holdout gate only - sensitivity
    # tiers report delta R2 + bootstrap CI.
    available = set(manifest.get("holdout_sizes", []))
    sizes = sorted(s for s in available if s != args.holdout_size)
    out = []
    for size in sizes:
        labels = label_lookup(manifest, size)
        row_labels = [labels[ep] for ep in uni.episode_id]
        s_uni = _copy_universe_masked(uni, np.ones(uni.r_a.shape[0], dtype=bool))
        s_uni.label = row_labels
        s_pools = build_pools(s_uni)
        s_folds = {f: fit_fold_model(s_pools[f], f, args) for f in FOLD_LABELS}
        k_star_s, _, _ = select_k_star(s_folds, s_pools["holdout"].n_rows)
        curve_s, a_tilde_s = holdout_delta_r2_curve(s_folds, s_pools["holdout"])
        pred_s = _holdout_avg_predictions(s_folds, s_pools["holdout"], k_star_s)
        boot_s = stats.block_bootstrap_delta_r2(
            a_tilde_s, pred_s, s_pools["holdout"].episode_id, n_reps=args.n_bootstrap, seed=args.seed
        )
        out.append(
            {
                "holdout_size": size,
                "k_star": k_star_s,
                "delta_r2_k_star": curve_s[int(k_star_s)],
                "bootstrap_ci_low": boot_s["ci_low"],
                "bootstrap_ci_high": boot_s["ci_high"],
            }
        )
    return out


def _cca_lambda_sensitivity(fold_models: dict, holdout: RowPool, k_star: int, args: argparse.Namespace) -> list[dict]:
    """CCA ridge lambda sensitivity curve (protocol section 8: 0.1*I start + curve)."""
    out = []
    for lam in args.cca_lambdas_sensitivity:
        preds, a_tildes = [], []
        for fold in FOLD_LABELS:
            m = fold_models[fold]
            a_tilde = holdout.r_a - m.mu_a.predict(holdout.c)
            f_tilde = holdout.f - m.mu_f.predict(holdout.c)
            a_tildes.append(a_tilde)
            cca_l = stats.fit_cca(m.f_tilde, m.a_tilde, ridge_lambda=lam)
            z_l = cca_l.project(f_tilde, k_star)
            z_l_fold = cca_l.project(m.f_tilde, k_star)
            coef = stats.fit_linear_probe(z_l_fold, m.a_tilde, lam=args.probe_lam)
            preds.append(z_l @ coef)
        a_avg = np.mean(a_tildes, axis=0)
        out.append({"lambda": lam, "delta_r2_k_star": stats.r2_explained(a_avg, np.mean(preds, axis=0))})
    return out


def _gate_semantics(fold_models: dict, pools: dict, args: argparse.Namespace) -> dict:
    """Raw vs conditional association (oracle M3): the gate speaks for the residual increment."""
    r_all = np.concatenate([pools[f].r_a for f in (*FOLD_LABELS, "holdout")], axis=0)
    f_all = np.concatenate([pools[f].f for f in (*FOLD_LABELS, "holdout")], axis=0)
    cca_raw = stats.fit_cca(f_all - f_all.mean(axis=0), r_all - r_all.mean(axis=0), ridge_lambda=args.cca_lambda)
    return {
        "note": "gates judge the residual-increment hypothesis, not total flow-aux value (oracle M3)",
        "rho1_raw_F_vs_RA": float(cca_raw.rho[0]),
        "rho1_conditional_Ftilde_vs_Atilde": float(np.mean([m.cca.rho[0] for m in fold_models.values()])),
        "var_RA": float((r_all**2).sum() / r_all.shape[0]),
        "var_A_tilde": float(np.mean([m.a_tilde.var(axis=0).sum() for m in fold_models.values()])),
    }


def _data_report(manifest: dict, uni: Universe, pools: dict, args: argparse.Namespace) -> dict:
    suite_counts: dict[str, int] = {}
    for suite, label in zip(uni.suite, uni.label, strict=True):
        key = f"{suite}/{label}"
        suite_counts[key] = suite_counts.get(key, 0) + 1
    nan_total = uni.nan_excluded + int(uni.r_a.shape[0])
    return {
        "n_rows_total": int(uni.r_a.shape[0]),
        "n_episodes": len(set(uni.episode_id)),
        "stride": args.stride,
        "stride_excluded_rows": uni.stride_excluded,
        "nan_anchor_excluded_rows": uni.nan_excluded,
        "nan_anchor_excluded_fraction": (uni.nan_excluded / nan_total) if nan_total else 0.0,
        "rows_per_label": {lab: pools[lab].n_rows for lab in (*FOLD_LABELS, "holdout")},
        "suite_x_label_rows": suite_counts,
        "excluded_action_stats": uni.excluded_action_stats,
        "pred_a0_diagnostics": shard_pred_stats(args.residual_dir),
        "manifest_protocol": manifest.get("protocol"),
        "seed": args.seed,
    }


def shard_pred_stats(residual_dir: str) -> dict:
    """Raw (unconditioned) diagnostics of pred_A0 vs A across shards: bias/RMSE."""
    diffs = []
    for path in sorted(Path(residual_dir).glob("*.npz")):
        try:
            with np.load(path, allow_pickle=False) as z:
                diffs.append(np.asarray(z["A"], dtype=np.float64) - np.asarray(z["pred_A0"], dtype=np.float64))
        except (KeyError, OSError, ValueError) as exc:
            raise RuntimeError(f"failed to read shard {path}: {exc}") from exc
    if not diffs:
        raise RuntimeError(f"no shards under {residual_dir}")
    diff = np.concatenate(diffs, axis=0)
    return {
        "n_rows": int(diff.shape[0]),
        "mean_abs_bias": float(np.abs(diff).mean()),
        "rmse": float(np.sqrt((diff**2).mean())),
    }


def _grouped_breakdown(
    key: str, pools: dict, fold_models: dict, holdout: RowPool, k_star: int, args: argparse.Namespace
) -> list[dict]:
    """Per-group delta R2(k*) for one grouping dimension (suite or task; protocol
    section 8: per-suite/per-task breakdown mandatory - guards against the pooled mean
    hiding long-tail reversals).

    Probes are refit per group on that group's fold rows and evaluated on that group's
    holdout rows, so each row is a within-group out-of-fold statistic."""
    out = []
    group_values = getattr(holdout, key)
    for group in sorted(set(group_values.tolist())):
        sel_h = group_values == group
        preds, a_tildes = [], []
        for fold in FOLD_LABELS:
            m = fold_models[fold]
            pool = pools[fold]
            sel_f = getattr(pool, key) == group
            f_tilde_fold = pool.f[sel_f] - m.mu_f.predict(pool.c[sel_f])
            a_tilde_fold = pool.r_a[sel_f] - m.mu_a.predict(pool.c[sel_f])
            z_train = m.cca.project(f_tilde_fold, k_star)
            coef = stats.fit_linear_probe(z_train, a_tilde_fold, lam=args.probe_lam)
            f_tilde_h = holdout.f[sel_h] - m.mu_f.predict(holdout.c[sel_h])
            preds.append(m.cca.project(f_tilde_h, k_star) @ coef)
            a_tildes.append(holdout.r_a[sel_h] - m.mu_a.predict(holdout.c[sel_h]))
        a_avg = np.mean(a_tildes, axis=0)
        delta_r2 = stats.r2_explained(a_avg, np.mean(preds, axis=0))
        out.append(
            {
                key: group,
                "n_holdout_rows": int(sel_h.sum()),
                "delta_r2_k_star": delta_r2,
                "meets_delta_r2_threshold": bool(delta_r2 >= stats.DELTA_R2_THRESHOLD),
            }
        )
    return out


def per_suite_breakdown(
    pools: dict, fold_models: dict, holdout: RowPool, k_star: int, args: argparse.Namespace
) -> list[dict]:
    """Per-suite delta R2(k*) with the G2 threshold flag (protocol section 8)."""
    return _grouped_breakdown("suite", pools, fold_models, holdout, k_star, args)


def per_task_breakdown(
    pools: dict, fold_models: dict, holdout: RowPool, k_star: int, args: argparse.Namespace
) -> list[dict]:
    """Per-task delta R2(k*) with the G2 threshold flag (protocol section 8)."""
    return _grouped_breakdown("task", pools, fold_models, holdout, k_star, args)


def _verdict(gates: dict, k_star: int) -> dict:
    """go/no-go decision plus failure attribution (audit section 4 reporting)."""
    passed = [
        gates["G1_existence"],
        gates["G2_increment"],
        gates["G3_predictability"],
        gates["G4_fold_consistency"],
    ]
    decision = "GO" if all(passed) else "NO-GO"
    attribution = []
    if not gates["G1_existence"]:
        attribution.append(
            "G1 failed: holdout delta R2(k*) does not exceed the Freedman-Lane max-over-k "
            "null 95th percentile - no evidence of flow information beyond C at this power; "
            "check MDE and consider more permutations/holdout before final no-go."
        )
    if not gates["G2_increment"]:
        attribution.append(
            "G2 failed: residual increment below the frozen 1% threshold or bootstrap CI "
            "includes 0 - effect size too small for Stage B training."
        )
    if not gates["G3_predictability"]:
        attribution.append(
            "G3 failed: frozen embeddings cannot decode z_{1:k*} to >= 50% explained "
            "variance (linear or MLP); z may not be learnable by the policy backbone."
        )
    if not gates["G4_fold_consistency"]:
        attribution.append(
            "G4 failed: fold subspace disagreement (median principal angle >= 45 deg) - "
            "CCA directions unstable; suspect k* too large, ridge too weak, or no signal."
        )
    return {
        "decision": decision,
        "k_star": k_star,
        "attribution": attribution,
        "next_step": "proceed to Stage B (z distillation)" if decision == "GO" else "do not train on z",
    }


def render_report(results: dict) -> str:
    """Markdown report with the go/no-go decision, gates, curves and attributions."""
    lines = [
        "# RAP-FSD Stage A report",
        "",
        "Protocol: pre-registered v1.2 + v1.2.2 anchor fix (R_A = A - pred_A0, C_ext = C union pred_A0)"
        " + v1.3 plateau k* rule; provenance in results.json.",
        "",
    ]
    v = results["verdict"]
    lines.append(f"## Verdict: **{v['decision']}** (k* = {v['k_star']})")
    lines.append("")
    lines.extend(f"- {text}" for text in v["attribution"])
    lines.append("")
    lines.append("## Gates (protocol v1.2, frozen thresholds)")
    lines.append("")
    lines.append("| gate | pass | key statistic |")
    lines.append("| --- | --- | --- |")
    s = results["gates"]["stats"]
    lines.append(
        f"| G1 existence (delta R2 > perm null 95%) | {results['gates']['G1_existence']} | "
        f"obs {s['delta_r2_star']:.4f} vs null95 {s['null_95_max_over_k']:.4f} |"
    )
    g2_ci = s["delta_r2_star_ci_low"]
    lines.append(f"| G2 increment (>= 1%, CI low > 0) | {results['gates']['G2_increment']} | CI low {g2_ci:.4f} |")
    g3_best = s["probe_r2_best"]
    lines.append(f"| G3 predictability (>= 50%) | {results['gates']['G3_predictability']} | best probe {g3_best:.3f} |")
    lines.append(
        f"| G4 fold consistency (< 45 deg) | {results['gates']['G4_fold_consistency']} | "
        f"median {s['median_principal_angle_deg']:.1f} deg |"
    )
    lines.append("")
    lines.append("## MDE (protocol v1.2 G2 / oracle M4)")
    lines.append("")
    mde = results["mde"]
    lines.append(
        f"- SE(delta R2(k*)) = {mde['se_delta_r2_k_star']:.5f} at holdout n={mde['holdout_rows']}, "
        f"k*={mde['k_star']} (compare with the frozen 1% threshold)"
    )
    lines.append("")
    lines.append("## Holdout delta R2 curve (fold-averaged predictions)")
    lines.append("")
    lines.append(", ".join(f"k={k}: {val:.4f}" for k, val in results["holdout_delta_r2_curve"].items()))
    lines.append("")
    lines.append("## Fold-OOF curves")
    lines.append("")
    for fold, curve in results["fold_oof_curves"].items():
        lines.append(f"- {fold}: " + ", ".join(f"k={k}: {val:.4f}" for k, val in curve.items()))
    lines.append("")
    lines.append("## Controls")
    lines.append("")
    for name in ("rrr", "random_subspace", "reversed_time"):
        lines.append(f"- {name}: delta R2(k*) = {results['controls'][name]['delta_r2_k_star']:.4f}")
    lines.append("")
    lines.append("## Probes (G3)")
    lines.append("")
    lines.append(f"- linear: {results['probes']['linear']['r2']:.4f} (lambda {results['probes']['linear']['lam']})")
    lines.append(f"- MLP: {results['probes']['mlp']['r2']:.4f}")
    lines.append(
        "- note: z targets come from the single full-data projection (audit H5), fitted on all rows "
        "including the holdout - G3 carries a mild in-sample optimism in the projection direction"
    )
    lines.append("")
    lines.append("## Positive controls (instrument certification, informational)")
    lines.append("")
    pc = results["positive_control"]
    lines.append(f"- E_vis -> raw actions A: R2 = {pc['control_a_r2']:.4f}")
    lines.append(f"- E_vis -> top-k* PCA of F_tilde: R2 = {pc['control_b_r2']:.4f}")
    lines.append(f"- features: {pc['features']}; instrument_certified = {pc['instrument_certified']}")
    lines.append("")
    lines.append("## Data / NaN anchors")
    lines.append("")
    d = results["data"]
    lines.append(f"- rows {d['n_rows_total']}, episodes {d['n_episodes']}, stride {d['stride']}")
    nan_frac = d["nan_anchor_excluded_fraction"]
    lines.append(
        f"- NaN-anchor excluded: {d['nan_anchor_excluded_rows']} rows ({nan_frac:.4f} of cached rows)"
    )
    lines.append("")
    lines.append("## Per-suite breakdown")
    lines.append("")
    for entry in results.get("per_suite_breakdown", []):
        rows_n = entry["n_holdout_rows"]
        met = "yes" if entry["meets_delta_r2_threshold"] else "no"
        lines.append(
            f"- {entry['suite']}: delta R2(k*) = {entry['delta_r2_k_star']:.4f} ({rows_n} rows, "
            f">= 1% threshold: {met})"
        )
    lines.append("")
    lines.append("## Per-task breakdown")
    lines.append("")
    for entry in results.get("per_task_breakdown", []):
        rows_n = entry["n_holdout_rows"]
        met = "yes" if entry["meets_delta_r2_threshold"] else "no"
        lines.append(
            f"- {entry['task']}: delta R2(k*) = {entry['delta_r2_k_star']:.4f} ({rows_n} rows, "
            f">= 1% threshold: {met})"
        )
    lines.append("")
    lines.append("## Sensitivity tiers")
    lines.append("")
    lines.append(f"- q-bar: {json.dumps(results['sensitivity'].get('qbar', {}), default=_json_default)}")
    for entry in results["sensitivity"].get("holdout_sizes", []):
        lines.append(
            f"- holdout {entry['holdout_size']}: delta R2(k*) = {entry['delta_r2_k_star']:.4f}, "
            f"CI low {entry['bootstrap_ci_low']:.4f}"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
