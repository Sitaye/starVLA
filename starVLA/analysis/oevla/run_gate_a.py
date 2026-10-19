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
  verify-determinism  restore mid-episode states and check roundtrip/replay
                      identity (prerequisite spike, ~minutes);
  generate            counterfactual branches for all selected states;
  analyze             pre-registered gates G0/A1/A2/A4/A6 -> results.json.

Example:
  python -m starVLA.analysis.oevla.run_gate_a --stage generate \
      --demo-dir /path/to/libero_datasets --per-suite 75 --workers 40 \
      --out-dir bar/oevla_gate_a
"""

import argparse
import json
import sys
from pathlib import Path

from starVLA.analysis import oevla
from starVLA.analysis.oevla import branch, state_hook

DEFAULT_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_long"]
DETERMINISM_STATES = 3
DETERMINISM_REPS = 3


def verify_determinism(args):
    from starVLA.analysis.oevla import descriptor

    demo_dir = Path(args.demo_dir)
    entries = []
    for suite in args.suites:
        files = sorted((demo_dir / suite).glob("*.hdf5"))
        if not files:
            raise FileNotFoundError(f"no hdf5 under {demo_dir / suite}")
        entries.append((suite, files[0].stem, files[0]))
    report = {"states": [], "all_roundtrip_ok": True, "all_rollouts_ok": True}
    for suite, task, path in entries:
        states, actions = branch.task_episode(str(path), branch.load_task_file(str(path))[1][0])
        env = branch._get_env(str(demo_dir), f"{suite}/{task}", args.camera)
        names, source = descriptor.object_bodies(env)
        layout = descriptor.feature_layout(names)
        t0 = branch.CHUNK_H + 1
        chunk = actions[t0 : t0 + branch.CHUNK_H]

        def extract(env=env, layout=layout):
            phi, *_ = descriptor.snapshot_features(env, layout)
            return phi

        out = state_hook.verify_determinism(
            env, states[t0], list(chunk), extract, n_reps=DETERMINISM_REPS
        )
        out.update({"suite": suite, "task": task, "object_source": source, "phi_dim": len(out["phi"])})
        del out["phi"]
        report["states"].append(out)
        report["all_roundtrip_ok"] &= out["roundtrip_err"] <= state_hook.ROUNDTRIP_TOL
        report["all_rollouts_ok"] &= out["rollout_spread"] <= state_hook.SPIKE_SPREAD_TOL
        env.close()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "determinism.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not (report["all_roundtrip_ok"] and report["all_rollouts_ok"]):
        sys.exit(1)


def generate(args):
    manifest = branch.generate(
        args.demo_dir,
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


def main():
    parser = argparse.ArgumentParser(description="OE-VLA gate A")
    parser.add_argument("--stage", required=True, choices=["verify-determinism", "generate", "analyze"])
    parser.add_argument("--demo-dir", required=True)
    parser.add_argument("--out-dir", default="bar/oevla_gate_a")
    parser.add_argument("--suites", nargs="+", default=DEFAULT_SUITES)
    parser.add_argument("--per-suite", type=int, default=75)
    parser.add_argument("--seed", type=int, default=oevla.SEED)
    parser.add_argument("--workers", type=int, default=40)
    parser.add_argument("--camera", type=int, default=None)
    args = parser.parse_args()
    {"verify-determinism": verify_determinism, "generate": generate, "analyze": analyze}[args.stage](args)


if __name__ == "__main__":
    main()
