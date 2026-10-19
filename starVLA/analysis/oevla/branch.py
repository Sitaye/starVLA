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

For each selected demo state: reconstruct the mid-episode simulator state by
replay (init state matched against the recorded frame-0 proprio, then the demo
actions up to t0), execute the expert chunk (H steps) zero-perturbed (repeated
baselines = determinism noise floor) or with one pre-registered probe
perturbation, then the fixed expert continuation (H_post steps); record phi at
chunk end and post-horizon, the success flag and bookkeeping. Branches of a
state share the restored state and env seed. Parallelized over states with
multiprocessing (spawn).
"""

import json
import multiprocessing
import traceback
import zlib
from pathlib import Path

import numpy as np

from starVLA.analysis import oevla
from starVLA.analysis.oevla import descriptor, estimate, lerobot_source, state_hook

BASELINE_REPS = 3
CHUNK_H = 8
POST_H = 32
EPS_LEVELS = (0.5, 1.0, 2.0)
K_MAIN = 48
K_VAL = 16
K_LIN = 12
TRANSFER_P = 50.0 / 350.0
MAX_SCAN_K = 600

# Init-state matching: frame-0 proprio against candidate benchmark init states.
INIT_POS_TOL = 0.05
INIT_ROT_TOL = 0.20
INIT_IMG_TOL = 12.0
REPLAY_FIDELITY_TOL = 0.05

_ENV_CACHE = {}
_SUITE_CACHE = {}
_MATCH_CACHE = {}


def _benchmark_suite(key):
    from libero.libero import benchmark

    bench = benchmark.get_benchmark_dict()
    if key not in bench:
        raise ValueError(f"unknown LIBERO benchmark suite {key!r}")
    return bench[key]()


def _norm_task_map(suite_obj):
    """Normalized instruction -> benchmark task_id."""
    out = {}
    for task_id in range(suite_obj.n_tasks):
        out[lerobot_source.norm_text(suite_obj.get_task(task_id).language)] = task_id
    return out


def _suite_ctx(dataset_dir, camera):
    """Worker-cached (LeRobotSuite, benchmark suite, task map)."""
    key = str(dataset_dir)
    if key not in _SUITE_CACHE:
        suite = lerobot_source.LeRobotSuite(dataset_dir)
        bench = _benchmark_suite(suite.benchmark_key)
        mapping = _norm_task_map(bench)
        task_ids = {}
        for task in {row["task"] for row in suite.episode_rows()}:
            tid = mapping.get(lerobot_source.norm_text(task))
            if tid is None:
                raise ValueError(f"{dataset_dir}: task {task!r} not in benchmark {suite.benchmark_key}")
            task_ids[task] = tid
        _SUITE_CACHE[key] = (suite, bench, task_ids)
    return _SUITE_CACHE[key]


def _get_env(dataset_dir, task, camera):
    """Benchmark-built env (eval parity: bddl + OffScreenRenderEnv) + init states."""
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    _, bench, task_ids = _suite_ctx(dataset_dir, camera)
    task_id = task_ids[task]
    key = (str(dataset_dir), task_id)
    if key not in _ENV_CACHE:
        bench_task = bench.get_task(task_id)
        bddl = Path(get_libero_path("bddl_files")) / bench_task.problem_folder / bench_task.bddl_file
        env_args = {
            "bddl_file_name": str(bddl),
            "camera_heights": camera or 256,
            "camera_widths": camera or 256,
        }
        env = OffScreenRenderEnv(**env_args)
        env.seed(zlib.crc32(task.encode("utf-8")) + task_id)
        _ENV_CACHE[key] = (env, np.asarray(bench.get_task_init_states(task_id), dtype=float))
    return _ENV_CACHE[key]


def _env_proprio(obs):
    pos = np.asarray(obs["robot0_eef_pos"], dtype=float)
    rot = descriptor.quat2axisangle(descriptor.eef_quat(obs))
    return np.concatenate([pos, rot])


def _env_image(obs):
    """Agentview-style camera frame, rotated 180 deg to the recorded convention.

    The no_noops regeneration rendered upside-down and saved the compensated
    frames (probe round 1: recorded vs rendered pairs are inverted), so the
    rendered frame is rotated 180 deg before any comparison with recorded data.
    """
    keys = [k for k in obs if k.endswith("_image") and "eye_in_hand" not in k]
    if not keys:
        raise ValueError(f"no agentview-style camera in obs: {sorted(obs)[:6]}")
    return np.asarray(obs[keys[0]])[::-1, ::-1].copy()


def match_init_state(env, init_states, target_proprio, target_img):
    """Identify which benchmark init state generated the episode.

    Arm proprio is nearly constant across candidates (every reset homes the
    arm), so selection is image-primary (object layout differs per candidate);
    pos/rot residuals are sanity gates on the frame-0 offset. Raises when no
    candidate satisfies all three pre-registered tolerances."""
    env.reset()
    candidates = []
    for j, init in enumerate(init_states):
        obs = env.set_init_state(np.asarray(init, dtype=float).flatten())
        prop = _env_proprio(obs)
        img = _env_image(obs)
        diff = min(
            float(np.mean(np.abs(img.astype(np.float32) - target_img.astype(np.float32)))),
            float(np.mean(np.abs(img.astype(np.float32)[..., ::-1] - target_img.astype(np.float32)))),
        )
        candidates.append(
            {
                "init": j,
                "pos_err": float(np.linalg.norm(prop[:3] - target_proprio[:3])),
                "rot_err": float(np.linalg.norm(prop[3:6] - target_proprio[3:6])),
                "img_err": diff,
            }
        )
    candidates.sort(key=lambda c: (c["img_err"], c["pos_err"]))
    for c in candidates:
        if c["pos_err"] < INIT_POS_TOL and c["rot_err"] < INIT_ROT_TOL and c["img_err"] < INIT_IMG_TOL:
            return c
    top = ", ".join(
        f"init{c['init']}(pos={c['pos_err']:.3f},rot={c['rot_err']:.3f},img={c['img_err']:.1f})"
        for c in candidates[:3]
    )
    raise ValueError(f"no init state matches recorded frame 0 ({top})")


def replay_to(env, init_state, actions, t0):
    """Reconstruct the mid-episode sim state: init -> demo actions[:t0]."""
    env.reset()
    env.set_init_state(np.asarray(init_state, dtype=float).flatten())
    for action in actions[:t0]:
        env.step(np.asarray(action, dtype=float))
    return state_hook.snapshot(env)


def select_states(data_root, suites, per_suite, seed, transfer_p=TRANSFER_P):
    """Pre-registered state selection: per suite, episodes round-robin over
    tasks, one state per (task, episode, temporal third); a seeded fraction is
    tagged transfer (held out from Jacobian fitting, A6 gate only).
    """
    entries = []
    for suite_name in suites:
        dataset_dir = Path(data_root) / suite_name
        suite = lerobot_source.LeRobotSuite(dataset_dir)
        rows = suite.episode_rows()
        by_task = {}
        for row in rows:
            by_task.setdefault(row["task"], []).append(row)
        tasks = sorted(by_task)
        rng = np.random.default_rng(seed + sum(ord(c) for c in suite_name))
        picked, k = 0, 0
        seen = set()
        while picked < per_suite:
            if k > MAX_SCAN_K:
                raise RuntimeError(f"{suite_name}: too many unusable episodes")
            task = tasks[k % len(tasks)]
            eps = by_task[task]
            row = eps[(k // len(tasks)) % len(eps)]
            t_total = row["length"]
            lo = CHUNK_H + 1
            hi = t_total - CHUNK_H - POST_H - 2
            if hi < lo:
                k += 1
                continue
            third = (k // (3 * len(tasks))) % 3
            t0 = int(lo + (hi - lo) * (2 * third + 1) / 6.0)
            state_id = f"{suite_name}__{lerobot_source.norm_text(task)}__ep{row['episode_index']}__t{t0}"
            if state_id in seen:
                k += 1
                continue
            seen.add(state_id)
            entries.append(
                {
                    "suite": suite_name,
                    "task": task,
                    "task_index": row["task_index"],
                    "episode_index": row["episode_index"],
                    "t0": t0,
                    "third": third,
                    "transfer": bool(rng.random() < transfer_p),
                }
            )
            picked += 1
            k += 1
    return {"states": entries}


def calibrate_delta_bar(entries, data_root):
    """Pre-registered probe scale: per-(step, dim) mean |chunk| over states."""
    suite_cache = {}
    chunks = []
    for e in entries:
        dir_ = str(Path(data_root) / e["suite"])
        if dir_ not in suite_cache:
            suite_cache[dir_] = lerobot_source.LeRobotSuite(dir_)
        actions = suite_cache[dir_].actions(e["episode_index"])
        chunks.append(np.abs(actions[e["t0"] : e["t0"] + CHUNK_H]).reshape(-1))
    return np.mean(np.stack(chunks), axis=0)


def run_branch(env, state, seed, chunk, post, layout):
    """One rollout: restore -> (perturbed) chunk -> fixed expert continuation.

    Returns phi at chunk end and post-horizon. success = step's done flag
    (repo eval convention: in this LIBERO build env.step's done encodes
    task success, cf. eval_libero.py done-as-success counting).
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
    success = bool(done)
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
        "n_steps": n_steps,
        "valid": valid,
        "snapshot": snapshot,
    }


def state_branches(dataset_dir, entry, delta_bar, seed, camera=None):
    """All branches for one state; returns (state_id, records)."""
    suite, _, _ = _suite_ctx(dataset_dir, camera)
    actions = suite.actions(entry["episode_index"])
    env, init_states = _get_env(dataset_dir, entry["task"], camera)
    object_names, object_source = descriptor.object_bodies(env)
    layout = descriptor.feature_layout(object_names)
    chunk = actions[entry["t0"] : entry["t0"] + CHUNK_H]
    post = actions[entry["t0"] + CHUNK_H : entry["t0"] + CHUNK_H + POST_H]
    state_id = f"{entry['suite']}__{lerobot_source.norm_text(entry['task'])}__ep{entry['episode_index']}__t{entry['t0']}"
    state_seed = int(seed + zlib.crc32(state_id.encode("utf-8")))

    match_key = (str(dataset_dir), entry["episode_index"])
    if match_key not in _MATCH_CACHE:
        target = suite.proprio(entry["episode_index"])[0]
        target_img = suite.frame_rgb(entry["episode_index"])
        _MATCH_CACHE[match_key] = match_init_state(env, init_states, target, target_img)
    match = _MATCH_CACHE[match_key]
    rec = suite.proprio(entry["episode_index"])
    f0 = _env_proprio(env.set_init_state(np.asarray(init_states[match["init"]], dtype=float).flatten()))
    state = replay_to(env, init_states[match["init"]], actions, entry["t0"])
    prop_t0 = _env_proprio(descriptor.env_obs(env))
    # Fidelity excess over the frame-0 residual: the constant convention/pose
    # offset cancels, only genuine replay divergence survives.
    replay_excess = {
        "pos": float(np.linalg.norm((prop_t0[:3] - rec[entry["t0"]][:3]) - (f0[:3] - rec[0][:3]))),
        "rot": float(np.linalg.norm((prop_t0[3:6] - rec[entry["t0"]][3:6]) - (f0[3:6] - rec[0][3:6]))),
    }

    base = run_branch(env, state, state_seed, chunk, post, layout)
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
            "init_index": match["init"] if r == 0 else None,
            "init_pos_err": match["pos_err"] if r == 0 else None,
            "init_rot_err": match["rot_err"] if r == 0 else None,
            "init_img_err": match["img_err"] if r == 0 else None,
            "replay_excess": replay_excess if r == 0 else None,
            "success": base["success"],
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
        out = run_branch(env, state, state_seed, chunk + perturb, post, layout)
        records.append(
            {
                "state_id": state_id,
                "kind": "perturb",
                "rep": i,
                "eps": eps,
                "dir": perturb.reshape(-1).tolist(),
                "fold": fold,
                "success": out["success"],
                "n_steps": out["n_steps"],
                "valid": out["valid"],
                "phi_chunk": out["phi_chunk"],
                "phi_post": out["phi_post"],
                "snapshot": None,
            }
        )
    return state_id, records


def _worker(payload):
    dataset_dir, entry, delta_bar, seed, camera = payload
    try:
        return state_branches(dataset_dir, entry, delta_bar, seed, camera)
    except Exception:
        return {"state_id": None, "error": traceback.format_exc()}


def generate(data_root, out_dir, suites, per_suite, seed, workers, camera=None):
    """Orchestrate branch generation; writes manifest.json, states.json and
    branches/<state_id>.jsonl. Envs and episode init-matching are per-worker;
    the rest is embarrassingly parallel."""
    out = Path(out_dir)
    (out / "branches").mkdir(parents=True, exist_ok=True)
    (out / "branches" / "errors.log").unlink(missing_ok=True)
    sel = select_states(data_root, suites, per_suite, seed)
    entries = sel["states"]
    delta_bar = calibrate_delta_bar(entries, data_root)

    entry_by_id = {}
    for e in entries:
        sid = f"{e['suite']}__{lerobot_source.norm_text(e['task'])}__ep{e['episode_index']}__t{e['t0']}"
        entry_by_id[sid] = e
    payloads = [(str(Path(data_root) / e["suite"]), e, delta_bar, seed, camera) for e in entries]
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
            entry["init_index"] = rec0["init_index"]
            entry["init_arm_err"] = rec0["init_arm_err"]
            entry["init_img_err"] = rec0["init_img_err"]
            entry["replay_drift"] = rec0["replay_drift"]
            with (out / "branches" / f"{state_id}.jsonl").open("w") as f:
                for rec in records:
                    f.write(json.dumps(rec) + "\n")
    manifest = {
        "stage": "generate",
        "data_root": str(data_root),
        "source": "lerobot_v2.1",
        "suites": suites,
        "per_suite": per_suite,
        "seed": seed,
        "eps_levels": list(EPS_LEVELS),
        "k": {"main": K_MAIN, "val": K_VAL, "lin": K_LIN},
        "chunk_h": CHUNK_H,
        "post_h": POST_H,
        "baseline_reps": BASELINE_REPS,
        "delta_bar": delta_bar.tolist(),
        "objects": {
            f"{e['suite']}__{e['task']}": {"names": e.get("object_names"), "source": e.get("object_source")}
            for e in entries
        },
        "camera": camera,
        "states": len(entries),
        "failed_states": n_fail,
        "replay_drift_max": max([e.get("replay_drift") or 0.0 for e in entries], default=0.0),
        "oevla_version": getattr(oevla, "__version__", "0.1"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / "states.json").write_text(json.dumps(sel, indent=2))
    print(json.dumps({"states": len(entries), "failed_states": n_fail}))
    return manifest
