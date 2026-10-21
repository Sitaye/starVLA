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
"""Offline outcome-geometry precompute for gate B training arms (cluster-only).

For every episode: match the init state once against the benchmark candidates
(replay-fidelity gate, v1.0.4), then replay the demo forward once and record,
at every chunk-complete transition t (t <= T - CHUNK_H), the raw action-space
geometry G_t = J_t^T W J_t where J_t is the finite-difference Jacobian of the
pre-registered analytic dead-reckoning map (gate A's analytic_jacobian, chunk
only). Rows land in a per-suite float16 memmap; unmatched episodes and
incomplete chunks stay NaN (invalid -> samples keep training with OE masked
out, identical across arms). Suite-level artifacts derive at the end:
half_range (per-dim (max-min)/2 in training key order), task_mean (valid-row
mean of the per-state normalized G), trace_mean, and the shuffled-arm
permutation/rank (seeded).

Example:
  python -m starVLA.analysis.oevla.precompute_g \
      --data-root /defaultShare/pubdata/OXE_LEROBOT_DATASET/libero_no_noops_1.0.0_lerobot \
      --out-root /defaultShare/archive/linpenghan/cvpr27/oe_g_cache --workers 40
"""

import argparse
import json
import multiprocessing
import traceback
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from starVLA.analysis import oevla
from starVLA.analysis.oevla import branch, descriptor, estimate, lerobot_source
from starVLA.analysis.oevla.run_gate_a import DEFAULT_SUITES, _setup_libero_env

G_DTYPE = np.float16
MATCH_T0 = branch.CHUNK_H + 1
# Mirror of the training-stats parquet filter in gr00t_lerobot/datasets.py
# (_load_or_compute_statistics) so half_range matches the normalizer source.
_STATS_EXCLUDED_EPISODES = ("episode_033675.parquet",)


def _action_key_order(modality_json):
    """Training action-dim order: Libero4in1 action_keys (x..gripper) sliced
    through the LeRobot modality metadata, so half_range matches the columns
    the StateActionTransform normalizes."""
    meta = modality_json["action"]
    order = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
    spans, dims = [], 0
    for key in order:
        if key not in meta:
            raise ValueError(f"modality.json action.{key} missing")
        span = meta[key]
        spans.append((int(span["start"]), int(span["end"])))
        dims = max(dims, int(span["end"]))
    return spans, dims


def _suite_half_range(suite):
    """(max - min) / 2 per action dim over the suite's raw parquet actions,
    same aggregation as calculate_dataset_statistics (mode abs); the
    passthrough gripper dim is frozen to 1.0."""
    try:
        modality_json = json.loads((Path(suite.dataset_dir) / "meta" / "modality.json").read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"{suite.dataset_dir}: unreadable meta/modality.json ({e})") from e
    actions_all = np.concatenate(
        [
            suite.actions(row["episode_index"])
            for row in suite.episode_rows()
            if f"episode_{row['episode_index']:06d}.parquet" not in _STATS_EXCLUDED_EPISODES
        ],
        axis=0,
    )
    spans, dims = _action_key_order(modality_json)
    half = np.ones(dims, dtype=np.float32)
    for lo, hi in spans[:-1]:  # last span = passthrough gripper
        half[lo:hi] = (actions_all[:, lo:hi].max(axis=0) - actions_all[:, lo:hi].min(axis=0)) / 2.0
    return half


def n_rows(length):
    """Chunks an episode contributes: one per fit position, none when the episode is too short.

    The reader (``oe_cache``) and the writer must agree on this exactly — a negative term
    would shift ``row0`` for every following episode, leaving finite rows in the wrong
    slots, which the loss cannot detect.
    """
    return max(int(length) - branch.CHUNK_H + 1, 0)


def _replay_geometry(env, actions, layout, G_mmap, row0):
    """One full-episode replay pass; writes G rows for t in [0, T - CHUNK_H].

    Returns the number of valid (finite) rows written. Capture happens BEFORE
    stepping action a_t, so row t holds the geometry of the state the training
    sample's base_index lands on."""
    w = descriptor.metric_weights(layout)
    T = len(actions)
    n_valid = 0
    for t in range(T - branch.CHUNK_H + 1):
        _, objects, eef_pos, eef_quat, gripper = descriptor.snapshot_features(env, layout)
        snap = {
            "eef_pos": eef_pos,
            "eef_quat": eef_quat,
            "grasp_name": descriptor.grasp_object(objects, eef_pos, gripper),
            "objects": objects,
        }
        J = estimate.analytic_jacobian(snap, actions[t : t + branch.CHUNK_H], layout)
        G = estimate.outcome_metric(J, w)
        if np.all(np.isfinite(G)):
            G_mmap[row0 + t] = G.astype(G_DTYPE)
            n_valid += 1
        env.step(np.asarray(actions[t], dtype=float))
    return n_valid


def _episode_geometry(payload):
    suite_dir, ep_index, row0, camera, g_path = payload
    try:
        suite, _, _ = branch._suite_ctx(suite_dir, camera)
        row = next(r for r in suite.episode_rows() if r["episode_index"] == ep_index)
        task, T = row["task"], row["length"]
        env, init_states = branch._get_env(suite_dir, task, camera)
        object_names, _ = descriptor.object_bodies(env)
        layout = descriptor.feature_layout(object_names)
        actions = suite.actions(ep_index)
        if actions.shape[0] != T:
            raise ValueError(f"ep{ep_index}: action grid {actions.shape} length != episode length {T}")
        rec = suite.proprio(ep_index)
        match = branch.match_by_replay(
            env, init_states, rec[0], suite.frame_rgb(ep_index), actions, MATCH_T0, rec
        )
        G_mmap = np.load(g_path, mmap_mode="r+")
        env.seed(zlib.crc32(f"{Path(suite_dir).name}#ep{ep_index}".encode("utf-8")))
        env.reset()
        env.set_init_state(np.asarray(init_states[match["init"]], dtype=float).flatten())
        env.sim.forward()
        n_valid = _replay_geometry(env, actions, layout, G_mmap, row0)
        G_mmap.flush()
        return {
            "suite": Path(suite_dir).name,
            "episode_index": ep_index,
            "n_rows": n_rows(T),
            "n_valid": n_valid,
            "replay_excess": max(match["replay_excess"].values()),
        }
    except Exception:
        return {"suite": Path(suite_dir).name, "episode_index": ep_index, "error": traceback.format_exc()}


def _write_stats(data_root, out_root, suites, seed):
    """Per-suite derived artifacts after all rows are written."""
    manifest = {"stage": "precompute", "suites": {}}
    for suite_name in suites:
        directory = Path(out_root) / suite_name
        G = np.load(directory / "G.npy", mmap_mode="r")
        total_rows = G.shape[0]
        c = np.empty(total_rows, dtype=np.float64)
        task_mean = np.zeros(G.shape[1:], dtype=np.float64)
        n_valid = 0
        for i in range(total_rows):
            row = np.asarray(G[i], dtype=np.float64)
            if not np.isfinite(row).all():
                c[i] = np.nan
                continue
            c[i] = float(np.mean(np.diag(row)))
            task_mean += row / c[i]
            n_valid += 1
        trace_mean = float(np.nanmean(c))
        np.lib.format.open_memmap(directory / "task_mean.npy", mode="w+", dtype=np.float16, shape=G.shape[1:])[:] = (
            task_mean / max(n_valid, 1)
        ).astype(np.float16)
        np.save(directory / "trace_mean.npy", np.float32(trace_mean))
        suite = lerobot_source.LeRobotSuite(Path(data_root) / suite_name)
        np.save(directory / "half_range.npy", _suite_half_range(suite))
        rng = np.random.default_rng(seed + zlib.crc32(suite_name.encode("utf-8")))
        valid_idx = np.flatnonzero(np.isfinite(c))
        perm = rng.permutation(valid_idx).astype(np.int64)
        rank = np.full(total_rows, -1, dtype=np.int64)
        rank[valid_idx] = np.arange(len(valid_idx), dtype=np.int64)
        np.save(directory / "perm.npy", perm)
        np.save(directory / "rank.npy", rank)
        manifest["suites"][suite_name] = {
            "rows": int(total_rows),
            "valid_rows": int(n_valid),
            "coverage": n_valid / max(total_rows, 1),
            "trace_mean": trace_mean,
        }
    manifest["seed"] = seed
    manifest["oevla_version"] = getattr(oevla, "__version__", "0.1")
    (Path(out_root) / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({s: m["coverage"] for s, m in manifest["suites"].items()}))
    return manifest


def precompute(data_root, out_root, suites, seed, workers, camera=None):
    """Allocate memmaps, run the episode pool, then write suite artifacts."""
    data_root, out_root = Path(data_root), Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    payloads = []
    for suite_name in suites:
        suite_dir = data_root / suite_name
        suite = lerobot_source.LeRobotSuite(suite_dir)
        rows = suite.episode_rows()
        probe = suite.actions(rows[0]["episode_index"])
        dim = probe.shape[1]
        shape = (sum(n_rows(r["length"]) for r in rows), branch.CHUNK_H * dim, branch.CHUNK_H * dim)
        directory = out_root / suite_name
        directory.mkdir(parents=True, exist_ok=True)
        np.lib.format.open_memmap(directory / "G.npy", mode="w+", dtype=G_DTYPE, shape=shape)[:] = np.nan
        (directory / "lengths.json").write_text(
            json.dumps({str(r["episode_index"]): r["length"] for r in rows})
        )
        (directory / "meta.json").write_text(json.dumps({"chunk_h": branch.CHUNK_H, "action_dim": int(dim)}))
        (out_root / "errors.log").write_text("")  # truncate once per precompute run
        g_path = str(directory / "G.npy")
        row0 = 0
        for r in rows:
            payloads.append((str(suite_dir), r["episode_index"], row0, camera, g_path))
            row0 += n_rows(r["length"])

    n_fail = 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for res in pool.map(_episode_geometry, payloads):
            if "error" in res:
                n_fail += 1
                with (out_root / "errors.log").open("a") as f:
                    f.write(json.dumps({k: res[k] for k in ("suite", "episode_index")}) + "\n")
                    f.write(res["error"] + "\n")
    print(json.dumps({"episodes": len(payloads), "failed_episodes": n_fail}))
    manifest = _write_stats(data_root, out_root, suites, seed)
    manifest["episodes"] = len(payloads)
    manifest["failed_episodes"] = n_fail
    (Path(out_root) / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description="OE-VLA gate B outcome-geometry precompute")
    parser.add_argument("--data-root", type=str, required=True, help="LeRobot root holding the suite dirs")
    parser.add_argument("--out-root", type=str, required=True, help="Output root (one dir per suite)")
    parser.add_argument("--suites", type=str, nargs="*", default=None, help="Default: the 4 no_noops suites")
    parser.add_argument("--seed", type=int, default=oevla.SEED)
    parser.add_argument("--workers", type=int, default=40)
    parser.add_argument("--camera", type=int, default=None)
    parser.add_argument("--libero-home", default=None, help="LIBERO repo root (default: $LIBERO_HOME)")
    args = parser.parse_args()
    _setup_libero_env(args.libero_home)
    suites = args.suites or list(DEFAULT_SUITES)
    precompute(args.data_root, args.out_root, suites, args.seed, args.workers, args.camera)


if __name__ == "__main__":
    main()
