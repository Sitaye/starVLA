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
"""Counterfactual branch generation for OE-VLA gate A (cluster-only execution).

For each selected demo state: restore the recorded simulator state, execute the
expert chunk (H steps) zero-perturbed (repeated baselines = determinism noise
floor) or with one pre-registered probe perturbation, then the fixed expert
continuation (H_post steps); record phi at chunk end and post-horizon, the
success flag and bookkeeping. Branches of a state share the restored state and
env seed. Parallelized over states with multiprocessing (spawn).
"""

import json
import multiprocessing
import traceback
import zlib
from pathlib import Path

import numpy as np

from starVLA.analysis import oevla
from starVLA.analysis.oevla import descriptor, estimate, state_hook

BASELINE_REPS = 3
CHUNK_H = 8
POST_H = 32
EPS_LEVELS = (0.5, 1.0, 2.0)
K_MAIN = 48
K_VAL = 16
K_LIN = 12
TRANSFER_P = 50.0 / 350.0
MAX_SCAN_K = 200 * 3

_ENV_CACHE = {}
_H5_CACHE = {}


def load_task_file(path):
    """Lazy per-worker handle; returns (handle, demos, env_args)."""
    import h5py

    if path not in _H5_CACHE:
        f = h5py.File(path, "r")
        try:
            demos = sorted(f["data"].keys())
            env_args = json.loads(f["data"].attrs["env_args"])
        except KeyError as e:
            raise ValueError(f"{path}: malformed LIBERO hdf5 ({e})") from e
        _H5_CACHE[path] = (f, demos, env_args)
    return _H5_CACHE[path]


def task_episode(path, demo):
    f, demos, _ = load_task_file(path)
    if demo not in demos:
        raise KeyError(f"{demo} not in {path}")
    g = f[f"data/{demo}"]
    return np.asarray(g["states"], dtype=np.float64), np.asarray(g["actions"], dtype=np.float64)


def select_states(demo_dir, suites, per_suite, seed, transfer_p=TRANSFER_P):
    """Pre-registered state selection: per suite, demos round-robin over task
    files, one state per (task, demo, temporal third); a seeded fraction is
    tagged transfer (held out from Jacobian fitting, A6 gate only).
    """
    import h5py

    entries = []
    for suite in suites:
        suite_dir = Path(demo_dir) / suite
        files = sorted(suite_dir.glob("*.hdf5"))
        if not files:
            raise FileNotFoundError(f"no hdf5 under {suite_dir}")
        rng = np.random.default_rng(seed + sum(ord(c) for c in suite))
        picked, k = 0, 0
        seen = set()
        while picked < per_suite:
            if k > MAX_SCAN_K:
                raise RuntimeError(f"{suite}: too many unusable demos")
            path = files[k % len(files)]
            with h5py.File(path, "r") as f:
                demos = sorted(f["data"].keys())
                if not demos:
                    raise ValueError(f"{path} has no demos")
                demo = demos[(k // len(files)) % len(demos)]
                t_total = int(f[f"data/{demo}/states"].shape[0])
            lo = CHUNK_H + 1
            hi = t_total - CHUNK_H - POST_H - 2
            if hi < lo:
                k += 1
                continue
            third = (k // (3 * len(files))) % 3
            t0 = int(lo + (hi - lo) * (2 * third + 1) / 6.0)
            state_id = f"{suite}__{path.stem}__{demo}__{t0}"
            if state_id in seen:
                k += 1
                continue
            seen.add(state_id)
            entries.append(
                {
                    "suite": suite,
                    "task": path.stem,
                    "demo": demo,
                    "t0": t0,
                    "third": third,
                    "transfer": bool(rng.random() < transfer_p),
                }
                )
            picked += 1
            k += 1
    return {"states": entries}


def calibrate_delta_bar(entries, demo_dir):
    """Pre-registered probe scale: per-(step, dim) mean |chunk| over states."""
    chunks = []
    for e in entries:
        _, actions = task_episode(str(Path(demo_dir) / e["suite"] / f"{e['task']}.hdf5"), e["demo"])
        chunks.append(np.abs(actions[e["t0"] : e["t0"] + CHUNK_H]).reshape(-1))
    return np.mean(np.stack(chunks), axis=0)


def _get_env(demo_dir, task, camera):
    from libero.libero.envs import OffScreenRenderEnv

    key = (str(demo_dir), task)
    if key not in _ENV_CACHE:
        _, _, env_args = load_task_file(str(Path(demo_dir) / task))
        env_args = dict(env_args)
        if camera:
            env_args["camera_heights"] = camera
            env_args["camera_widths"] = camera
        else:
            env_args["camera_heights"] = min(int(env_args.get("camera_heights", 256)), 128)
            env_args["camera_widths"] = min(int(env_args.get("camera_widths", 256)), 128)
        _ENV_CACHE[key] = OffScreenRenderEnv(**env_args)
    return _ENV_CACHE[key]


def run_branch(env, state, seed, chunk, post, layout):
    """One rollout: restore -> (perturbed) chunk -> fixed expert continuation.

    Returns phi at chunk end and post-horizon, success via _check_success
    (done recorded separately), and the analytic-baseline snapshot.
    """
    env.seed(int(seed))
    env.reset()
    state_hook.restore(env, state)
    _, objects, eef_pos, eef_quat, gripper = descriptor.snapshot_features(env, layout)
    grasp_name = descriptor.grasp_object(objects, eef_pos, gripper)
    success = False
    phi_chunk = phi_post = None
    n_steps = 0
    done = False
    plan = list(np.asarray(chunk, dtype=float)) + list(np.asarray(post, dtype=float))
    for i, action in enumerate(plan):
        _, _, done, _ = env.step(np.asarray(action, dtype=float))
        n_steps += 1
        if i + 1 == len(chunk) or done:
            phi_chunk, *_ = descriptor.snapshot_features(env, layout)
        if done:
            break
    if phi_chunk is None:
        raise RuntimeError("chunk did not complete (env crashed mid-chunk)")
    success = bool(env._check_success())
    if done:
        phi_post = phi_chunk
    else:
        phi_post, *_ = descriptor.snapshot_features(env, layout)
    valid = bool(np.all(np.isfinite(phi_chunk)) and np.all(np.isfinite(phi_post)))
    snapshot = {
        "eef_pos": eef_pos.tolist(),
        "eef_quat": eef_quat.tolist(),
        "gripper_width": float(gripper),
        "grasp_name": grasp_name,
        "chunk": np.asarray(chunk, dtype=float).reshape(-1).tolist(),
        "objects": {
            name: {"pos": pos_.tolist(), "quat": quat_.tolist()}
            for name, (pos_, quat_) in objects.items()
        },
    }
    return {
        "phi_chunk": phi_chunk.tolist(),
        "phi_post": phi_post.tolist(),
        "success": int(success),
        "done": bool(done),
        "n_steps": n_steps,
        "valid": valid,
        "snapshot": snapshot,
    }


def state_branches(demo_dir, entry, delta_bar, seed, camera=None):
    """All branches for one state; returns (state_id, records)."""
    path = str(Path(demo_dir) / entry["suite"] / f"{entry['task']}.hdf5")
    states, actions = task_episode(path, entry["demo"])
    env = _get_env(demo_dir, entry["task"], camera)
    object_names, object_source = descriptor.object_bodies(env)
    layout = descriptor.feature_layout(object_names)
    chunk = actions[entry["t0"] : entry["t0"] + CHUNK_H]
    post = actions[entry["t0"] + CHUNK_H : entry["t0"] + CHUNK_H + POST_H]
    state_id = f"{entry['suite']}__{entry['task']}__{entry['demo']}__{entry['t0']}"
    state_seed = int(seed + zlib.crc32(state_id.encode("utf-8")))

    base = run_branch(env, states[entry["t0"]], state_seed, chunk, post, layout)
    records = [
        {
            "state_id": state_id,
            "kind": "baseline",
            "rep": r,
            "eps": None,
            "dir": None,
            "fold": None,
            "object_names": object_names if r == 0 else None,
            "object_source": object_source if r == 0 else None,
            "success": base["success"],
            "done": base["done"],
            "n_steps": base["n_steps"],
            "valid": base["valid"],
            "phi_chunk": base["phi_chunk"],
            "phi_post": base["phi_post"],
            "snapshot": base["snapshot"] if r == 0 else None,
        }
        for r in range(BASELINE_REPS)
    ]

    rng = np.random.default_rng(state_seed)
    plan = [(1.0, "train") for _ in range(K_MAIN)] + [(1.0, "val") for _ in range(K_VAL)]
    plan += [(eps, "lin") for eps in EPS_LEVELS for _ in range(K_LIN // len(EPS_LEVELS))]
    for i, (eps, fold) in enumerate(plan):
        perturb = estimate.sample_direction(rng, delta_bar, eps).reshape(chunk.shape)
        out = run_branch(env, states[entry["t0"]], state_seed, chunk + perturb, post, layout)
        records.append(
            {
                "state_id": state_id,
                "kind": "perturb",
                "rep": i,
                "eps": eps,
                "dir": perturb.reshape(-1).tolist(),
                "fold": fold,
                "success": out["success"],
                "done": out["done"],
                "n_steps": out["n_steps"],
                "valid": out["valid"],
                "phi_chunk": out["phi_chunk"],
                "phi_post": out["phi_post"],
                "snapshot": None,
            }
        )
    return state_id, records


def _worker(payload):
    demo_dir, entry, delta_bar, seed, camera = payload
    try:
        return state_branches(demo_dir, entry, delta_bar, seed, camera)
    except Exception:
        return {"state_id": None, "error": traceback.format_exc()}


def generate(demo_dir, out_dir, suites, per_suite, seed, workers, camera=None):
    """Orchestrate branch generation; writes manifest.json, states.json and
    branches/<state_id>.jsonl. Object names are discovered per task env inside
    the worker; per-worker env caches make the rest embarrassingly parallel."""
    out = Path(out_dir)
    (out / "branches").mkdir(parents=True, exist_ok=True)
    (out / "branches" / "errors.log").unlink(missing_ok=True)
    sel = select_states(demo_dir, suites, per_suite, seed)
    entries = sel["states"]
    delta_bar = calibrate_delta_bar(entries, demo_dir)
    path0 = str(Path(demo_dir) / entries[0]["suite"] / f"{entries[0]['task']}.hdf5")
    _, _, env_args = load_task_file(path0)

    entry_by_id = {f"{e['suite']}__{e['task']}__{e['demo']}__{e['t0']}": e for e in entries}
    objects, payloads = {}, []
    for e in entries:
        payloads.append((demo_dir, e, delta_bar, seed, camera))
    n_fail = 0
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(workers) as pool:
        for res in pool.imap_unordered(_worker, payloads):
            if "error" in res:
                n_fail += 1
                with (out / "branches" / "errors.log").open("a") as f:
                    f.write(res["error"] + "\n")
                continue
            state_id, records = res
            entry = entry_by_id[state_id]
            rec0 = records[0]
            entry["object_names"] = rec0["object_names"]
            entry["object_source"] = rec0["object_source"]
            objects[entry["task"]] = {
                "names": rec0["object_names"],
                "source": rec0["object_source"],
            }
            with (out / "branches" / f"{state_id}.jsonl").open("w") as f:
                for rec in records:
                    f.write(json.dumps(rec) + "\n")
    manifest = {
        "stage": "generate",
        "demo_dir": str(demo_dir),
        "suites": suites,
        "per_suite": per_suite,
        "seed": seed,
        "eps_levels": list(EPS_LEVELS),
        "k": {"main": K_MAIN, "val": K_VAL, "lin": K_LIN},
        "chunk_h": CHUNK_H,
        "post_h": POST_H,
        "baseline_reps": BASELINE_REPS,
        "delta_bar": delta_bar.tolist(),
        "objects": objects,
        "camera": camera,
        "env_args_sample": env_args,
        "states": len(entries),
        "failed_states": n_fail,
        "oevla_version": getattr(oevla, "__version__", "0.1"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / "states.json").write_text(json.dumps(sel, indent=2))
    print(json.dumps({"states": len(entries), "failed_states": n_fail}))
    return manifest
