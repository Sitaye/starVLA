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
"""Unit tests for the OE-VLA gate-A package.

Estimation/analysis layers are pure numpy and tested directly; the selection
path is tested against a tiny synthetic LeRobot v2.1 suite (skipped when
pandas/pyarrow are unavailable). End-to-end gate behavior is exercised on
fabricated branches covering the GO world (kinematics + object coupling), the
kinematic-confound NO-GO world and the instrument-failure path.
"""

import json
import math
import zlib
from pathlib import Path

import numpy as np
import pytest

from starVLA.analysis.oevla import analyze as analyze_mod
from starVLA.analysis.oevla import branch as branch_mod
from starVLA.analysis.oevla import descriptor, estimate, lerobot_source

PHI_DIM = 25  # 6 + 9 * 2 objects + 1
DIR_DIM = 56
LAYOUT = {"object_names": ["obj_a", "obj_b"]}


def _metric():
    return descriptor.metric_weights(LAYOUT)


def _kinematic_jacobian():
    """Translation-only world: eef pos rows track the per-step pos dims."""
    J = np.zeros((PHI_DIM, DIR_DIM))
    for step in range(branch_mod.CHUNK_H):
        J[0:3, step * 7 : step * 7 + 3] = np.eye(3)
        J[12:15, step * 7 : step * 7 + 3] = -np.eye(3)
    return J


def _go_jacobian():
    """Kinematics + object-coupling structure beyond the analytic baseline."""
    rng = np.random.default_rng(123)
    J = _kinematic_jacobian()
    J[6:12, :] = rng.standard_normal((6, DIR_DIM)) * 1.5
    return J


def _quat_from_axisangle(a):
    angle = float(np.linalg.norm(a))
    axis = a / angle
    return np.concatenate([[np.cos(angle / 2)], axis * np.sin(angle / 2)])


def test_fit_jacobian_exact():
    rng = np.random.default_rng(0)
    J = rng.standard_normal((4, 10))
    E = rng.standard_normal((200, 10))
    D = E @ J.T
    assert np.allclose(estimate.fit_jacobian(E, D), J, atol=1e-6)


def test_effective_rank():
    assert estimate.effective_rank(np.diag([2.0, 1.0, 0.0, 0.0]), energy=0.9) == 2
    assert estimate.effective_rank(np.zeros((3, 3))) == 0
    assert estimate.effective_rank(np.eye(5), energy=0.9) == 5


def test_sample_direction_norm_and_seed():
    db = np.abs(np.random.default_rng(1).standard_normal(DIR_DIM)) + 0.1
    e1 = estimate.sample_direction(np.random.default_rng(7), db, 1.0)
    e2 = estimate.sample_direction(np.random.default_rng(7), db, 1.0)
    assert np.allclose(e1, e2)
    d = e1 / 1.0 / db
    assert np.isclose(np.sqrt(np.sum((d * db) ** 2)), 1.0)  # d unit in db-metric
    e3 = estimate.sample_direction(np.random.default_rng(7), db, 0.5)
    assert np.allclose(e3 / 0.5 / db, d)


def test_quat_roundtrip_and_rotate():
    a = np.array([0.3, -0.4, 0.5])
    q = _quat_from_axisangle(a)
    assert np.allclose(descriptor.quat2axisangle(q), a, atol=1e-9)
    v = np.array([1.0, 0.0, 0.0])
    assert np.allclose(descriptor.quat_rotate([1.0, 0, 0, 0], v), v)
    qz = _quat_from_axisangle(np.array([0.0, 0.0, np.pi / 2]))
    assert np.allclose(descriptor.quat_rotate(qz, v), [0.0, 1.0, 0.0], atol=1e-9)


def test_obs_quat_is_xyzw():
    # Cluster probe round 1 regression: robosuite obs quats are (x, y, z, w).
    # These slots are the gripper-down home pose of libero_spatial ep372; read
    # as (w, x, y, z) they produce an absurd near-identity rotation.
    q = np.array([0.99870986, 0.04726228, -0.01817629, 0.00380947])
    aa = descriptor.quat2axisangle(descriptor.eef_quat({"robot0_eef_quat": q}))
    assert np.allclose(aa, [3.12995315, 0.14811983, -0.05696444], atol=1e-6)
    assert np.linalg.norm(aa) > 3.0


def test_env_image_rot180():
    img = np.arange(16, dtype=np.uint8).reshape(4, 4)[..., None].repeat(3, -1)
    out = branch_mod._env_image({"agentview_image": img})
    assert np.array_equal(out, img[::-1, ::-1])


class _FakeMatchEnv:
    """set_init_state returns the pre-built obs for each candidate in order."""

    def __init__(self, obs_list):
        self._obs_list = obs_list
        self._i = -1

    def reset(self):
        self._i = -1

    def set_init_state(self, _flat):
        self._i += 1
        return self._obs_list[self._i]


def test_match_init_state_img_primary():
    rng = np.random.default_rng(7)
    target_img = rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)
    identity_xyzw = np.array([0.0, 0.0, 0.0, 1.0])  # (x, y, z, w)
    target_proprio = np.zeros(6)

    def obs(img):
        # obs images are pre-flip: _env_image rotates 180 deg to compare.
        return {"robot0_eef_pos": np.zeros(3), "robot0_eef_quat": identity_xyzw, "agentview_image": img[::-1, ::-1]}

    off = target_img.copy()
    off[0, 0] += 40
    env = _FakeMatchEnv([obs(off), obs(target_img), obs(off)])
    match = branch_mod.match_init_state(env, [np.zeros(1)] * 3, target_proprio, target_img)
    assert match["init"] == 1
    assert match["img_err"] == pytest.approx(0.0, abs=1e-6)
    assert match["pos_err"] == pytest.approx(0.0, abs=1e-9)


def test_match_init_state_rot_gate_rejects():
    rng = np.random.default_rng(8)
    target_img = rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)
    target_proprio = np.concatenate([np.zeros(3), [0.3, 0.0, 0.0]])  # recorded rot off home
    obs = {
        "robot0_eef_pos": np.zeros(3),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "agentview_image": target_img[::-1, ::-1],
    }
    env = _FakeMatchEnv([obs])
    with pytest.raises(ValueError, match="no init state matches"):
        branch_mod.match_init_state(env, [np.zeros(1)], target_proprio, target_img)


def _static_snapshot(grasp_name=None):
    rng = np.random.default_rng(2)
    return {
        "eef_pos": rng.standard_normal(3),
        "eef_quat": _quat_from_axisangle(rng.standard_normal(3) * 0.1),
        "gripper_width": 0.01 if grasp_name else 0.06,
        "grasp_name": grasp_name,
        "objects": {
            "obj_a": (rng.standard_normal(3), _quat_from_axisangle(rng.standard_normal(3) * 0.1)),
            "obj_b": (rng.standard_normal(3), _quat_from_axisangle(rng.standard_normal(3) * 0.1)),
        },
    }


def test_analytic_jacobian_translation():
    snap = _static_snapshot()
    J = estimate.analytic_jacobian(snap, np.zeros((branch_mod.CHUNK_H, 7)), LAYOUT)
    for step in range(branch_mod.CHUNK_H):
        cols = slice(step * 7, step * 7 + 3)
        assert np.allclose(J[0:3, cols], np.eye(3), atol=1e-4)
    assert np.allclose(J[6:9, :], 0.0, atol=1e-8)  # static object pos
    assert np.allclose(J[12:15, 0:3], -np.eye(3), atol=1e-4)  # rel_pos


def test_analytic_jacobian_rotation():
    snap = _static_snapshot()
    u = np.array([0.3, -0.5, 0.81])
    u = u / np.linalg.norm(u)
    theta = 0.05
    action = np.zeros(7)
    action[3:6] = u * theta
    chunk = np.tile(action, (branch_mod.CHUNK_H, 1))
    phi = estimate._phi_analytic(snap, chunk, LAYOUT)
    # Independent Hamilton product: q1 = q0 ⊗ exp(u, CHUNK_H·theta).
    q0 = np.asarray(snap["eef_quat"], dtype=float)
    ang = branch_mod.CHUNK_H * theta
    qd = [math.cos(ang / 2), *(u * math.sin(ang / 2))]
    a, b = q0, qd  # layout (w, x, y, z), q1 = a ⊗ b (Hamilton product)
    q1 = [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ]
    assert np.allclose(phi[3:6], descriptor.quat2axisangle(q1), atol=1e-9)


def test_analytic_jacobian_grasp_follow():
    snap = _static_snapshot(grasp_name="obj_a")
    J = estimate.analytic_jacobian(snap, np.zeros((branch_mod.CHUNK_H, 7)), LAYOUT)
    for step in range(branch_mod.CHUNK_H):
        assert np.allclose(J[6:9, step * 7 : step * 7 + 3], np.eye(3), atol=1e-4)
    assert np.allclose(J[12:15, 0:3], 0.0, atol=1e-6)  # translation-invariant


def test_spearman_and_rankdata():
    assert analyze_mod.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert analyze_mod.spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)
    assert np.allclose(analyze_mod.rankdata([1, 2, 2, 3]), [1, 2.5, 2.5, 4])


def test_cluster_bootstrap_ci_constant():
    lo, hi, n_nan = analyze_mod.cluster_bootstrap_ci(lambda c: 0.5, [1, 2, 3], n_reps=50)
    assert lo == pytest.approx(0.5) and hi == pytest.approx(0.5) and n_nan == 0


def test_select_states(tmp_path):
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    name = "libero_spatial_no_noops_1.0.0_lerobot"
    _write_lerobot_suite(tmp_path, name)
    sel = branch_mod.select_states(str(tmp_path), [name], 4, seed=42)
    sel2 = branch_mod.select_states(str(tmp_path), [name], 4, seed=42)
    assert sel == sel2
    for e in sel["states"]:
        assert 9 <= e["t0"] <= 60 - branch_mod.CHUNK_H - branch_mod.POST_H - 2
        assert e["task"] in {"task a", "task b"}
    assert len(sel["states"]) == 4
    ids = {
        f"{e['suite']}__{lerobot_source.norm_text(e['task'])}__ep{e['episode_index']}__t{e['t0']}"
        for e in sel["states"]
    }
    assert len(ids) == 4


def test_calibrate_delta_bar_shape(tmp_path):
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    name = "libero_spatial_no_noops_1.0.0_lerobot"
    _write_lerobot_suite(tmp_path, name)
    sel = branch_mod.select_states(str(tmp_path), [name], 4, seed=42)
    delta_bar = branch_mod.calibrate_delta_bar(sel["states"], str(tmp_path))
    assert delta_bar.shape == (56,)
    assert np.allclose(delta_bar, 0.01)
    chunk = estimate.sample_direction(np.random.default_rng(0), delta_bar, 1.0)
    assert chunk.reshape(8, 7).shape == (8, 7)


def test_lerobot_helpers():
    pytest.importorskip("pandas")
    name = "libero_goal_no_noops_1.0.0_lerobot"
    assert lerobot_source.suite_benchmark_key(name) == "libero_goal"
    assert lerobot_source.norm_text("Turn ON the Stove!") == "turnonthestove"


def _write_lerobot_suite(root, name, tasks=("task a", "task b"), n_eps=4, length=60):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    d = root / name
    (d / "meta").mkdir(parents=True)
    (d / "data" / "chunk-000").mkdir(parents=True)
    tasks_sorted = sorted(tasks)
    (d / "meta" / "tasks.jsonl").write_text(
        "".join(json.dumps({"task_index": i, "task": t}) + "\n" for i, t in enumerate(tasks_sorted))
    )
    episodes = []
    for ep in range(n_eps):
        task = tasks_sorted[ep % len(tasks_sorted)]
        actions = np.full((length, 7), 0.01)
        state = np.zeros((length, 8))
        # Real v2.1 episodes.jsonl rows carry `tasks` (list), not task_index.
        episodes.append({"episode_index": ep, "tasks": [task], "length": length})
        df = pd.DataFrame(
            {
                "action": [a.tolist() for a in actions],
                "observation.state": [s.tolist() for s in state],
                "frame_index": np.arange(length),
                "episode_index": np.full(length, ep),
                "task_index": np.full(length, tasks_sorted.index(task)),
            }
        )
        df.to_parquet(d / "data" / "chunk-000" / f"episode_{ep:06d}.parquet", index=False)
    (d / "meta" / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in episodes))
    info = {
        "codebase_version": "v2.1",
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {"observation.images.image": {"dtype": "video"}},
    }
    (d / "meta" / "info.json").write_text(json.dumps(info))


def _write_synthetic(out_dir, response_fn, spread=1e-5):
    """Fabricate a full gate-A output directory (states.json + branches/)."""
    rng = np.random.default_rng(11)
    entries = []
    for t in range(4):
        for k in range(8):
            e = {
                "suite": f"suite{t % 2}",
                "task": f"task {t}",
                "task_index": t,
                "episode_index": 10 * t + k,
                "t0": 100 + k,
                "third": k % 3,
                "transfer": k in (3, 7),
            }
            e["state_id"] = analyze_mod._state_id(e)
            entries.append(e)
    for e in entries:
        phi0 = rng.standard_normal(PHI_DIM) * 0.1
        snap0 = _snapshot_record()
        srng = np.random.default_rng(zlib.crc32(e["state_id"].encode()))
        records = []
        for r in range(3):
            records.append(
                {
                    "state_id": e["state_id"],
                    "kind": "baseline",
                    "rep": r,
                    "eps": None,
                    "dir": None,
                    "fold": None,
                    "success": 0,
                    "n_steps": 40,
                    "valid": True,
                    "phi_chunk": phi0.tolist(),
                    "phi_post": (phi0 + spread * rng.standard_normal(PHI_DIM)).tolist(),
                    "snapshot": snap0 if r == 0 else None,
                }
            )
        for i, (eps, fold) in enumerate(
            [(1.0, "train")] * 48 + [(1.0, "val")] * 16
            + [(lev, "lin") for lev in (0.5, 1.0, 2.0) for _ in range(4)]
        ):
            e_dir = estimate.sample_direction(srng, np.abs(srng.standard_normal(DIR_DIM)) + 0.1, eps)
            dphi = np.asarray(response_fn(e_dir, snap0), dtype=float) + 1e-6 * rng.standard_normal(PHI_DIM)
            records.append(
                {
                    "state_id": e["state_id"],
                    "kind": "perturb",
                    "rep": i,
                    "eps": eps,
                    "dir": e_dir.tolist(),
                    "fold": fold,
                    "success": int(srng.random() < 0.02),
                    "n_steps": 40,
                    "valid": True,
                    "phi_chunk": phi0.tolist(),
                    "phi_post": (phi0 + dphi).tolist(),
                    "snapshot": None,
                }
            )
        out = Path(out_dir) / "branches"
        out.mkdir(parents=True, exist_ok=True)
        with (out / f"{e['state_id']}.jsonl").open("w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
    (Path(out_dir) / "states.json").write_text(json.dumps({"states": entries}, indent=2))


def _snapshot_record():
    rng = np.random.default_rng(5)
    chunk = (rng.standard_normal(DIR_DIM) * 0.02).tolist()
    return {
        "eef_pos": rng.standard_normal(3).tolist(),
        "eef_quat": [1.0, 0.0, 0.0, 0.0],
        "gripper_width": 0.06,
        "grasp_name": None,
        "chunk": chunk,
        "objects": {
            "obj_a": {"pos": rng.standard_normal(3).tolist(), "quat": [1.0, 0, 0, 0]},
            "obj_b": {"pos": rng.standard_normal(3).tolist(), "quat": [1.0, 0, 0, 0]},
        },
    }


def _kinematic_response(e, snap):
    """The confound world IS the dead-reckoning map: truth = analytic delta."""
    e = np.asarray(e, dtype=float)
    chunk = np.asarray(snap["chunk"], dtype=float)
    snap = analyze_mod._hydrate_snapshot(snap)
    up = estimate._phi_analytic(snap, (chunk + e).reshape(-1, analyze_mod.ACTION_DIM), LAYOUT)
    base = estimate._phi_analytic(snap, chunk.reshape(-1, analyze_mod.ACTION_DIM), LAYOUT)
    return up - base


@pytest.fixture
def fast_bootstrap(monkeypatch):
    real_ci = analyze_mod.cluster_bootstrap_ci
    monkeypatch.setattr(
        analyze_mod,
        "cluster_bootstrap_ci",
        lambda fn, cl, n_reps=0, seed=0: real_ci(fn, cl, n_reps=100, seed=seed),
    )


def test_analyze_end_to_end_go(tmp_path, fast_bootstrap):
    J = _go_jacobian()
    _write_synthetic(tmp_path, lambda e, s: J @ e)
    results = analyze_mod.analyze(tmp_path)
    assert results["verdict"]["gate_b_go"] is True
    assert results["verdict"]["failed"] == []
    assert results["gates"]["A1"]["median_rho"] > 0.9
    assert results["gates"]["A2"]["delta_rho"] > 0.5
    assert results["gates"]["A4"]["delta_rho"] > 0.3
    assert (tmp_path / "results.json").exists() and (tmp_path / "report.md").exists()


def test_analyze_end_to_end_kinematic_confound(tmp_path, fast_bootstrap):
    _write_synthetic(tmp_path, _kinematic_response)
    results = analyze_mod.analyze(tmp_path)
    assert results["verdict"]["gate_b_go"] is False
    assert "A4" in results["verdict"]["failed"]


def test_analyze_g0_noise_fail(tmp_path, fast_bootstrap):
    _write_synthetic(tmp_path, lambda e, s: (_go_jacobian() @ e), spread=100.0)
    results = analyze_mod.analyze(tmp_path)
    assert results["verdict"]["gate_b_go"] is False
    assert "G0" in results["verdict"]["failed"]
