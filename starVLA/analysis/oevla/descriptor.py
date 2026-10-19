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
"""OE-VLA outcome descriptor phi and pre-registered metric (protocol section 3).

phi(s) is the continuous outcome block
    [eef_pos(3), eef_rot_axisangle(3),
     per object (name-sorted): obj_pos(3), obj_rot_axisangle(3), rel_pos(3),
     gripper_width(1)]
plus a separate binary success flag (tail metric, excluded from directional
geometry). Feature scales and metric weights W = 1/scale^2 are frozen
constants, not tuned: positions 0.05 m, rotations 0.15 rad, gripper_width
1.0 (raw meters). Grasp flags for the analytic baseline are pre-registered:
gripper mean width < GRIPPER_CLOSED and nearest object within GRASP_DIST.
"""

import numpy as np

POS_SCALE = 0.05
ROT_SCALE = 0.15
GRIPPER_SCALE = 1.0
GRIPPER_CLOSED = 0.03
GRASP_DIST = 0.075

STATIC_BODY_PREFIXES = ("robot", "worldbody", "table", "floor", "wall", "background", "skybox")


def quat2axisangle(quat):
    """Convert quaternion (w, x, y, z) to axis-angle; robosuite-compatible."""
    quat = np.asarray(quat, dtype=float)
    den = np.sqrt(max(np.dot(quat[1:], quat[1:]), 1e-12))
    if den == 0.0:
        return np.zeros(3)
    return (quat[1:] / den) * 2.0 * np.arccos(np.clip(quat[0], -1.0, 1.0))


def quat_mul(q1, q2):
    """Hamilton product (world-frame composition)."""
    w1, v1 = q1[0], q1[1:]
    w2, v2 = q2[0], q2[1:]
    return np.array(
        [w1 * w2 - np.dot(v1, v2), *(w1 * v2 + w2 * v1 + np.cross(v1, v2))]
    )


def quat_conj(q):
    return np.array([q[0], *(-q[1:])])


def quat_rotate(q, v):
    """Rotate vector v by quaternion q (world frame)."""
    w, u = q[0], q[1:]
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def quat_mul_axisangle(quat, axisangle):
    """Left-multiply quat by the rotation exp(axisangle) (world-frame delta)."""
    axisangle = np.asarray(axisangle, dtype=float)
    angle = np.linalg.norm(axisangle)
    if angle < 1e-12:
        return np.asarray(quat, dtype=float).copy()
    axis = axisangle / angle
    dq = np.concatenate([[np.cos(angle / 2.0)], axis * np.sin(angle / 2.0)])
    return quat_mul(quat, dq)


def feature_layout(object_names):
    """Deterministic descriptor layout: objects sorted by name."""
    return {"object_names": sorted(object_names)}


def phi_dim(layout):
    """Continuous-block dimension: 6 + 9 * n_objects + 1 (gripper)."""
    return 6 + 9 * len(layout["object_names"]) + 1


def object_bodies(env):
    """Task-relevant object body names, discovered in a pre-registered order.

    1. env.objects (LIBERO exposes root_body per object);
    2. fallback: sim body names minus robot/world/table prefixes.
    Returns (names, source); the source and names are recorded in the manifest.
    """
    names = []
    for obj in getattr(env, "objects", None) or []:
        name = getattr(obj, "root_body", None)
        if isinstance(name, (list, tuple)):
            name = name[0] if name else None
        if name:
            names.append(name)
    if names:
        return sorted(set(names)), "env.objects"
    names = [
        n
        for n in env.sim.model.body_names
        if not n.lower().startswith(STATIC_BODY_PREFIXES)
    ]
    return sorted(set(names)), "body_names_filter"


def env_obs(env):
    """Observation dict from whichever accessor this robosuite build exposes.

    The LIBERO wrapper delegates nothing, so walk env -> env.env and try the
    accessors seen across robosuite/LIBERO versions.
    """
    for name in ("_get_observations", "get_observation", "get_obs"):
        target = env
        while target is not None:
            if hasattr(target, name):
                return getattr(target, name)()
            target = getattr(target, "env", None)
    raise AttributeError(f"no observation accessor on {type(env).__name__}")


def snapshot_features(env, layout):
    """Read eef pose, object poses and gripper width from a LIBERO env."""
    obs = env_obs(env)
    eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=float)
    eef_quat = np.asarray(obs["robot0_eef_quat"], dtype=float)
    ids = env.sim.model.body_name2id
    feats = [eef_pos, quat2axisangle(eef_quat)]
    objects = {}
    for name in layout["object_names"]:
        i = ids(name)
        pos = np.asarray(env.sim.data.body_xpos[i], dtype=float)
        quat = np.asarray(env.sim.data.body_xquat[i], dtype=float)
        objects[name] = (pos, quat)
        feats += [pos, quat2axisangle(quat), pos - eef_pos]
    gripper = float(np.mean(np.asarray(obs["robot0_gripper_qpos"], dtype=float)))
    feats.append(np.array([gripper]))
    return np.concatenate(feats), objects, eef_pos, eef_quat, gripper


def grasp_object(objects, eef_pos, gripper_width):
    """Pre-registered grasp detection: gripper closed and nearest object in range.

    Returns the grasped object name or None. Detection errors only affect the
    analytic baseline (a control), and the feature-level breakdown in the
    analysis reports where an analytic/CF gap concentrates.
    """
    if gripper_width >= GRIPPER_CLOSED:
        return None
    best, best_dist = None, GRASP_DIST
    for name, (pos, _quat) in objects.items():
        d = float(np.linalg.norm(pos - eef_pos))
        if d < best_dist:
            best, best_dist = name, d
    return best


def metric_weights(layout):
    """W = diag(1/scale^2) over the continuous block (frozen, pre-registered)."""
    w = [1.0 / POS_SCALE**2] * 3
    w += [1.0 / ROT_SCALE**2] * 3
    for _ in layout["object_names"]:
        w += [1.0 / POS_SCALE**2] * 3
        w += [1.0 / ROT_SCALE**2] * 3
        w += [1.0 / POS_SCALE**2] * 3
    w += [1.0 / GRIPPER_SCALE**2]
    return np.array(w)


def normalized_features(phi, layout):
    """phi divided by per-feature scales (for NN distance in A6)."""
    return phi * np.sqrt(metric_weights(layout))
