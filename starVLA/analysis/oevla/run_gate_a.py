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
"""OE-VLA gate-A entry point (cluster-only; needs the LIBERO eval env).

Stages:
  verify-determinism  init-state matching, replay identity, snapshot roundtrip
                      and repeated-rollout spread for one state per suite
                      (prerequisite spike, ~minutes);
  generate            counterfactual branches for all selected states;
  analyze             pre-registered gates G0/A1/A2/A4/A6 -> results.json.

Example:
  python -m starVLA.analysis.oevla.run_gate_a --stage generate \
      --data-root /path/to/OXE_LEROBOT_DATASET/libero_no_noops_1.0.0_lerobot \
      --per-suite 75 --workers 40 --out-dir bar/oevla_gate_a
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from starVLA.analysis import oevla
from starVLA.analysis.oevla import branch, lerobot_source, state_hook

DEFAULT_SUITES = [
    "libero_spatial_no_noops_1.0.0_lerobot",
    "libero_object_no_noops_1.0.0_lerobot",
    "libero_goal_no_noops_1.0.0_lerobot",
    "libero_10_no_noops_1.0.0_lerobot",
]
DETERMINISM_REPS = 3


def _setup_libero_env(libero_home):
    """Point the libero package at its repo (eval_libero.sh parity)."""
    home = libero_home or os.environ.get("LIBERO_HOME")
    if not home:
        raise SystemExit("LIBERO location required: pass --libero-home or export LIBERO_HOME")
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ["LIBERO_CONFIG_PATH"] = os.path.join(home, "libero")
    parts = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    if home not in parts:
        os.environ["PYTHONPATH"] = os.pathsep.join([*parts, home])
    # os.environ edits do not affect sys.path of the running interpreter.
    if home not in sys.path:
        sys.path.append(home)


def verify_determinism(args):
    from starVLA.analysis.oevla import descriptor

    data_root = Path(args.data_root)
    report = {
        "suites": [],
        "all_init_ok": True,
        "all_replay_ok": True,
        "all_roundtrip_ok": True,
        "all_rollouts_ok": True,
        "all_fidelity_ok": True,
    }
    for suite_name in args.suites:
        dataset_dir = data_root / suite_name
        suite = lerobot_source.LeRobotSuite(dataset_dir)
        row = next((r for r in suite.episode_rows() if r["length"] >= 2 * branch.CHUNK_H + branch.POST_H + 3), None)
        if row is None:
            raise ValueError(f"{suite_name}: no episode long enough for the spike")
        t0 = branch.CHUNK_H + 1
        actions = suite.actions(row["episode_index"])
        env, init_states = branch._get_env(str(dataset_dir), row["task"], args.camera)
        names, source = descriptor.object_bodies(env)
        layout = descriptor.feature_layout(names)
        chunk = actions[t0 : t0 + branch.CHUNK_H]

        def extract(env=env, layout=layout):
            phi, *_ = descriptor.snapshot_features(env, layout)
            return phi

        entry = {"suite": suite_name, "episode": row["episode_index"], "task": row["task"], "object_source": source}
        rec = suite.proprio(row["episode_index"])
        try:
            match = branch.match_by_replay(
                env,
                init_states,
                rec[0],
                suite.frame_rgb(row["episode_index"]),
                actions,
                t0,
                rec,
            )
            entry["init_match"] = {k: match[k] for k in ("init", "pos_err", "rot_err", "img_err")}
            report["all_init_ok"] &= True
        except ValueError as e:
            entry["init_match"] = {"error": str(e)}
            report["all_init_ok"] = False
            report["suites"].append(entry)
            env.close()
            continue

        snap1 = branch.replay_to(env, init_states[match["init"]], actions, t0)
        snap2 = branch.replay_to(env, init_states[match["init"]], actions, t0)
        entry["replay_err"] = float(np.max(np.abs(snap1 - snap2)))
        entry["replay_excess"] = match["replay_excess"]
        out = state_hook.verify_determinism(env, snap1, list(chunk), extract, n_reps=DETERMINISM_REPS)
        entry["roundtrip_err"] = out["roundtrip_err"]
        entry["rollout_spread"] = out["rollout_spread"]
        entry["phi_dim"] = len(out["phi"])
        report["suites"].append(entry)
        report["all_replay_ok"] &= entry["replay_err"] <= state_hook.SPIKE_SPREAD_TOL
        report["all_roundtrip_ok"] &= entry["roundtrip_err"] <= state_hook.ROUNDTRIP_TOL
        report["all_rollouts_ok"] &= entry["rollout_spread"] <= state_hook.SPIKE_SPREAD_TOL
        report["all_fidelity_ok"] &= (
            entry["replay_excess"]["pos"] <= branch.REPLAY_FIDELITY_TOL
            and entry["replay_excess"]["rot"] <= branch.REPLAY_FIDELITY_TOL
        )
        env.close()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "determinism.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    gates = [k for k in report if k.startswith("all_")]
    if not all(report[k] for k in gates):
        sys.exit(1)


def generate(args):
    manifest = branch.generate(
        args.data_root,
        args.out_dir,
        args.suites,
        args.per_suite,
        args.seed,
        args.workers,
        args.camera,
    )
    print(json.dumps({k: manifest[k] for k in ("states", "failed_states")}, indent=2))


def analyze(args):
    from starVLA.analysis.oevla.analyze import analyze as run_analyze

    results = run_analyze(args.out_dir)
    print(json.dumps(results["verdict"], indent=2))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", required=True, choices=["verify-determinism", "generate", "analyze"])
    parser.add_argument("--data-root", help="root containing the LeRobot LIBERO suite dirs")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--suites", nargs="+", default=DEFAULT_SUITES)
    parser.add_argument("--per-suite", type=int, default=75)
    parser.add_argument("--seed", type=int, default=getattr(oevla, "SEED", 42))
    parser.add_argument("--workers", type=int, default=40)
    parser.add_argument("--camera", type=int, default=None, help="render resolution (default: dataset 256)")
    parser.add_argument("--libero-home", default=None, help="LIBERO repo root (default: $LIBERO_HOME)")
    return parser


def main():
    args = build_parser().parse_args()
    if args.stage in ("verify-determinism", "generate"):
        if not args.data_root:
            raise SystemExit("--data-root is required for this stage")
        _setup_libero_env(args.libero_home)
    {"verify-determinism": verify_determinism, "generate": generate, "analyze": analyze}[args.stage](args)


if __name__ == "__main__":
    main()
