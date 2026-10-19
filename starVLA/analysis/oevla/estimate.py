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
"""OE-VLA gate-A estimation core (pure numpy, protocol section 4).

Jacobians are probed with random direction rollouts and fitted by ridge
least squares in RAW action units (m, rad per step) against the RAW outcome
descriptor; the outcome metric is G = J^T W J with the pre-registered W.
Also implements the contact-free analytic (kinematics) baseline Jacobian.
"""

import numpy as np

from starVLA.analysis.oevla import descriptor as desc

RIDGE = 1e-8
GRIPPER_OPEN_WIDTH = 0.05


def fit_jacobian(E, D, ridge=RIDGE):
    """J^T = (E^T E + ridge I)^-1 E^T D  ->  J [m, d].

    E [K, d]: probe action perturbations; D [K, m]: outcome deviations
    delta phi = phi(perturbed) - phi(baseline).
    """
    E = np.asarray(E, dtype=float)
    D = np.asarray(D, dtype=float)
    gram = E.T @ E + ridge * np.eye(E.shape[1])
    return np.linalg.solve(gram, E.T @ D).T


def outcome_metric(J, w):
    """G = J^T W J with W = diag(w) over the m outcome features (RAW units)."""
    Jw = np.asarray(J, dtype=float) * np.sqrt(np.asarray(w, dtype=float))[:, None]
    return Jw.T @ Jw


def effective_rank(G, energy=0.9):
    """Number of eigenvalues covering `energy` of the positive spectrum."""
    ev = np.clip(np.linalg.eigvalsh(0.5 * (G + G.T))[::-1], 0.0, None)
    total = float(ev.sum())
    if total <= 0.0:
        return 0
    return int(np.searchsorted(np.cumsum(ev) / total, energy) + 1)


def direction_cost(G, e):
    """Directional cost e^T G e (RAW units)."""
    return float(np.asarray(e) @ np.asarray(G) @ np.asarray(e))


def trace_only_cost(G, e):
    """Isotropic baseline tr(G)/d * ||e||^2 (what a scalar weight knows)."""
    return float(np.trace(np.asarray(G)) / np.asarray(G).shape[0] * np.dot(e, e))


def sample_direction(rng, delta_bar, eps):
    """Pre-registered probe: e = eps * delta_bar * d, d = g/||g||_db (unit in
    the db-metric), g ~ N(0, I_d); ||g||_db = sqrt(sum_k (g_k delta_bar_k)^2)
    makes directions comparable across action dimensions with heterogeneous
    step sizes (protocol section 4)."""
    db = np.asarray(delta_bar, dtype=float).ravel()
    g = rng.standard_normal(db.size)
    norm_db = float(np.sqrt(np.sum((g * db) ** 2)))
    d = g / norm_db if norm_db > 0 else np.zeros_like(g)
    return eps * db * d


def _phi_analytic(snapshot, actions, layout):
    """Contact-free outcome map: dead-reckoned kinematics from the snapshot.

    eef: pos += per-step delta; rot: quat o exp(world-frame axis-angle delta).
    Objects: the pre-registered grasped object (rigid offset) follows the eef;
    all others are static. gripper feature from the last chunk command.
    Continuation is identical across branches under this model, so only the
    chunk contributes to delta phi.
    """
    pos = np.array(snapshot["eef_pos"], dtype=float)
    quat = np.array(snapshot["eef_quat"], dtype=float)
    actions = np.atleast_2d(np.asarray(actions, dtype=float))
    for step in actions:
        pos = pos + step[:3]
        quat = desc.quat_mul_axisangle(quat, step[3:6])
    grasp = snapshot.get("grasp_name")
    feats = [pos, desc.quat2axisangle(quat)]
    q0 = np.asarray(snapshot["eef_quat"], dtype=float)
    for name in layout["object_names"]:
        pos0, quat0 = snapshot["objects"][name]
        if name == grasp:
            # rigid offset rotates with the eef: rel_world = q o (q0^-1 rel0)
            rel = desc.quat_rotate(
                desc.quat_conj(q0), np.asarray(pos0, dtype=float) - snapshot["eef_pos"]
            )
            op = pos + desc.quat_rotate(quat, rel)
            oq = desc.quat_mul(quat, desc.quat_mul(desc.quat_conj(q0), quat0))
        else:
            op, oq = pos0, quat0
        feats += [op, desc.quat2axisangle(oq), op - pos]
    feats.append(np.array([GRIPPER_OPEN_WIDTH if actions[-1][6] >= 0.5 else 0.0]))
    return np.concatenate(feats)


def analytic_jacobian(snapshot, chunk, layout, eps=1e-5):
    """Finite-difference Jacobian of the analytic map over the chunk dims."""
    base = _phi_analytic(snapshot, chunk, layout)
    flat = np.asarray(chunk, dtype=float).reshape(-1)
    J = np.zeros((base.size, flat.size))
    for i in range(flat.size):
        up = flat.copy()
        up[i] += eps
        J[:, i] = (_phi_analytic(snapshot, up.reshape(chunk.shape), layout) - base) / eps
    return J
