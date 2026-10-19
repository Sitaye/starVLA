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
"""Alignment diagnostics for gate A on the LeRobot LIBERO data.

Prints/saves enough ground truth to explain why benchmark init states do not
match recorded frame-0 proprio (suspects: world-vs-base frame, OXE conversion
conventions, no_noops trimming). Cluster-only; run with the LIBERO venv python:

  python -m starVLA.analysis.oevla.probe_alignment --data-root ... --out-dir bar/probe
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from starVLA.analysis.oevla import branch, lerobot_source, run_gate_a


def _base_candidates(env):
    names = list(env.sim.model.body_names)
    return [n for n in names if "base" in n.lower() or "root" in n.lower()]


def _probe_suite(suite_name, dataset_dir, out_dir, n_candidates):
    suite = lerobot_source.LeRobotSuite(dataset_dir)
    rows = suite.episode_rows()
    row = max(rows, key=lambda r: r["length"])
    ep = row["episode_index"]
    state = suite.proprio(ep)
    actions = suite.actions(ep)
    env, init_states = branch._get_env(str(dataset_dir), row["task"], None)
    base_bodies = {
        n: np.asarray(env.sim.data.body_xpos[env.sim.model.body_name2id(n)]).tolist()
        for n in _base_candidates(env)
    }
    base_pos = np.asarray(next(iter(base_bodies.values()))) if base_bodies else np.zeros(3)
    candidates = []
    for j, init in enumerate(init_states[:n_candidates]):
        obs = env.set_init_state(np.asarray(init, dtype=float).flatten())
        prop = branch._env_proprio(obs)
        img = branch._env_image(obs)
        target = suite.frame_rgb(ep)
        candidates.append(
            {
                "init": j,
                "env_proprio": prop.tolist(),
                "abs_err": float(np.linalg.norm(prop - state[0][:6])),
                "rel_err": float(np.linalg.norm((prop[:3] - base_pos) - state[0][:3])),
                "rot_err": float(np.linalg.norm(prop[3:6] - state[0][3:6])),
                "img_err": float(np.mean(np.abs(img.astype(np.float32) - target.astype(np.float32)))),
            }
        )
    report = {
        "suite": suite_name,
        "episode": ep,
        "length": int(row["length"]),
        "task": row["task"],
        "n_candidates": len(candidates),
        "base_bodies": base_bodies,
        "recorded_state_first": state[0].tolist(),
        "recorded_state_mid": state[min(9, len(state) - 1)].tolist(),
        "action_first": actions[0].tolist(),
        "action_absmax_mean": float(np.mean(np.abs(actions))),
        "best_abs": min(candidates, key=lambda c: c["abs_err"]),
        "best_rel": min(candidates, key=lambda c: c["rel_err"]),
        "best_rot": min(candidates, key=lambda c: c["rot_err"]),
        "best_img": min(candidates, key=lambda c: c["img_err"]),
    }
    best = report["best_rel"]
    branch.replay_to(env, init_states[best["init"]], actions, t0=branch.CHUNK_H + 1)
    prop = branch._env_proprio(env._get_observations())
    report["replay_from_best_rel"] = {
        "init": best["init"],
        "drift_abs": float(np.linalg.norm(prop - state[branch.CHUNK_H + 1][:6])),
        "drift_rel": float(np.linalg.norm((prop[:3] - base_pos) - state[branch.CHUNK_H + 1][:3])),
    }
    from PIL import Image

    pair = np.concatenate([np.asarray(suite.frame_rgb(ep)), branch._env_image(env._get_observations())], axis=1)
    Image.fromarray(pair).save(out_dir / f"{suite_name}.png")
    env.close()
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--suites", nargs="+", default=run_gate_a.DEFAULT_SUITES)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--candidates", type=int, default=50)
    args = ap.parse_args()
    run_gate_a._setup_libero_env(None)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for suite_name in args.suites:
        try:
            reports.append(_probe_suite(suite_name, Path(args.data_root) / suite_name, out_dir, args.candidates))
        except Exception as e:  # one suite failing must not kill the others
            reports.append({"suite": suite_name, "error": repr(e)})
    (out_dir / "probe_alignment.json").write_text(json.dumps(reports, indent=2))
    print(json.dumps(reports, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
