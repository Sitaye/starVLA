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
"""Simulator state snapshot/restore and determinism verification.

LIBERO follows robosuite conventions: eval restores simulator states with
env.set_init_state (eval_libero.py); Gate A restores MID-EPISODE states the
same way, where the flat state is reconstructed by demo replay (branch.replay_to,
LeRobot demos carry no simulator states). The determinism spike
(run_gate_a.py --stage verify-determinism) must validate this usage before any
branch generation. Restore call order is pre-registered:
set_state_from_flattened (exact state API) falling back to set_init_state.
"""

import numpy as np

ROUNDTRIP_TOL = 1e-9
SPIKE_SPREAD_TOL = 1e-6


def snapshot(env):
    """Flat simulator state (the unit stored in hdf5 demo files)."""
    return np.asarray(env.sim.get_state().flatten(), dtype=float)


def restore(env, state):
    state = np.asarray(state, dtype=float)
    if hasattr(env.sim, "set_state_from_flattened"):
        env.sim.set_state_from_flattened(state)
    else:
        env.set_init_state(state)
    env.sim.forward()


def verify_determinism(env, state, actions, extract_phi, n_reps=3):
    """Roundtrip exactness + repeated-rollout identity for one state.

    Returns {"roundtrip_err", "rollout_spread", "phi"}; caller applies the
    pre-registered tolerances (roundtrip <= ROUNDTRIP_TOL, replay spread of the
    SAME chunk <= SPIKE_SPREAD_TOL).
    """
    restore(env, state)
    roundtrip_err = float(np.max(np.abs(snapshot(env) - state)))

    phis = []
    for _ in range(n_reps):
        restore(env, state)
        for action in actions:
            env.step(np.asarray(action, dtype=float))
        phis.append(extract_phi())
    spread = float(np.max(np.abs(phis[-1] - phis[0]))) if len(phis) > 1 else 0.0
    return {"roundtrip_err": roundtrip_err, "rollout_spread": spread, "phi": phis[-1]}
