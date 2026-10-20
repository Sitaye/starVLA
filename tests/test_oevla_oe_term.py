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
"""Gate-B OE term + geometry-cache unit tests (torch/numpy only, no VLM)."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

CHUNK, ACT = 2, 2
D = CHUNK * ACT


def _oe_loss_mod():
    """File-load the standalone loss module (torch+numpy only; no package chain)."""
    spec = importlib.util.spec_from_file_location(
        "oe_loss_under_test", Path(__file__).resolve().parents[1] / "starVLA/model/modules/oe_loss.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _oe_cache_mod():
    """File-load the cache module (json+pathlib+numpy only; no package chain)."""
    spec = importlib.util.spec_from_file_location(
        "oe_cache_under_test", Path(__file__).resolve().parents[1] / "starVLA/dataloader/oe_cache.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


OE_LOSS = _oe_loss_mod()
OE_MODES = OE_LOSS.OE_MODES
oe_quadratic = OE_LOSS.oe_quadratic


def _payload(g_a, invalid_b=False, c_bar=2.0, task_mean=None):
    ga = np.asarray(g_a, dtype=np.float32)
    gb = np.full((D, D), np.nan, dtype=np.float32) if invalid_b else np.asarray(g_a, dtype=np.float32) * 2.0
    tm = np.asarray(task_mean, dtype=np.float32)
    return [
        {"G": ga, "valid": True, "half_range": np.array([2.0, 0.5], dtype=np.float32), "task_mean": tm,
         "trace_mean": c_bar},
        {"G": gb, "valid": not invalid_b, "half_range": np.array([2.0, 0.5], dtype=np.float32), "task_mean": tm,
         "trace_mean": c_bar},
    ]


def _e_norm():
    return torch.tensor([[1.0, 0.25, 0.5, 2.0], [-0.5, 1.0, 0.0, -1.0]])


def _e_raw():
    return _e_norm().numpy() * np.tile(np.array([2.0, 0.5], dtype=np.float32), CHUNK)


def _blocks(g):
    """Step-major block-diagonal G for a per-step 2x2 geometry."""
    return np.kron(np.eye(CHUNK), np.asarray(g, dtype=np.float32))


@pytest.fixture(scope="module")
def payload():
    step_g = np.kron(np.eye(CHUNK), np.array([[4.0, 1.0], [1.0, 2.0]]))
    task_mean = np.kron(np.eye(CHUNK), np.array([[1.5, 0.25], [0.25, 1.0]]))
    return _payload(step_g, invalid_b=True, task_mean=task_mean)


def _manual_full(payload):
    e = _e_raw()
    out = np.zeros(2)
    for i, ex in enumerate(payload):
        if not ex["valid"]:
            continue
        g = np.asarray(ex["G"], dtype=np.float32)
        c = np.trace(g) / D
        out[i] = e[i] @ (g / c) @ e[i]
    return out


def _manual_diag(payload):
    e = _e_raw()
    out = np.zeros(2)
    for i, ex in enumerate(payload):
        if not ex["valid"]:
            continue
        g = np.asarray(ex["G"], dtype=np.float32)
        dg = np.diag(g)
        out[i] = (e[i] * (dg / dg.mean())) @ e[i]
    return out


def _run(mode, payload):
    loss, valid_frac = oe_quadratic(payload, _e_norm(), mode, CHUNK)
    return float(loss), float(valid_frac)


def test_oe_modes_registry():
    assert OE_MODES == ("task_fixed", "trace", "diag", "full", "shuffled")


def test_full_arm_matches_manual(payload):
    loss, valid_frac = _run("full", payload)
    np.testing.assert_allclose(loss, _manual_full(payload).sum() / 1, rtol=1e-5)
    assert valid_frac == pytest.approx(0.5)


def test_diag_arm_ignores_offdiagonal(payload):
    loss, _ = _run("diag", payload)
    np.testing.assert_allclose(loss, _manual_diag(payload).sum() / 1, rtol=1e-5)


def test_task_fixed_is_state_free(payload):
    """task_fixed must not read G (works even when every G row is NaN)."""
    broken = [dict(ex, G=np.full((D, D), np.nan, dtype=np.float32)) for ex in payload]
    loss_broken, frac_broken = _run("task_fixed", broken)
    loss_clean, frac_clean = _run("task_fixed", payload)
    assert loss_broken == pytest.approx(loss_clean, rel=1e-6)
    assert frac_broken == pytest.approx(frac_clean)


def test_trace_arm_is_scaled_norm(payload):
    e = _e_raw()
    w0 = (np.trace(np.asarray(payload[0]["G"])) / D) / payload[0]["trace_mean"]
    expected = w0 * (e[0] @ e[0])  # only state 0 valid
    loss, _ = _run("trace", payload)
    assert loss == pytest.approx(expected, rel=1e-5)


def test_invalid_rows_excluded(payload):
    """Invalid transitions contribute 0 to the term and shrink the denominator."""
    loss, valid_frac = _run("full", payload)
    manual = _manual_full(payload)
    assert manual[1] == 0.0
    assert loss == pytest.approx(manual[0], rel=1e-5)  # mean over 1 valid row
    assert valid_frac == pytest.approx(0.5)


def test_all_invalid_is_zero():
    step_g = np.kron(np.eye(CHUNK), np.array([[4.0, 1.0], [1.0, 2.0]]))
    task_mean = np.kron(np.eye(CHUNK), np.array([[1.5, 0.25], [0.25, 1.0]]))
    payload = _payload(step_g, task_mean=task_mean)
    payload = [dict(ex, valid=False, G=np.full((D, D), np.nan, dtype=np.float32)) for ex in payload]
    for mode in ("task_fixed", "trace", "diag", "full", "shuffled"):
        loss, valid_frac = _run(mode, payload)
        assert loss == 0.0
        assert valid_frac == 0.0


def test_shuffled_matches_full_math():
    """shuffled differs from full only by the cache-level row permutation."""
    step_g = np.kron(np.eye(CHUNK), np.array([[4.0, 1.0], [1.0, 2.0]]))
    payload = _payload(step_g, invalid_b=False, task_mean=np.kron(np.eye(CHUNK), [[1.5, 0.25], [0.25, 1.0]]))
    loss_full, _ = _run("full", payload)
    loss_shuf, _ = _run("shuffled", payload)
    assert loss_shuf == pytest.approx(loss_full, rel=1e-6)


class TestOEGeometryCache:
    def _build_suite(self, tmp_path, name="suite_a", n_ep=2, lengths=(12, 15), d=4):
        root = tmp_path / name
        root.mkdir()
        n = sum(t - 1 for t in lengths)  # valid domain t <= T-chunk_h, step-major rows
        g = np.full((n, d, d), np.nan, dtype=np.float16)
        keys = []
        offsets = {}
        row = 0
        for tid, t in enumerate(lengths):
            offsets[tid] = row
            for step in range(t - 1):
                g[row] = np.eye(d) * (row + 1.0)
                keys.append(tid * 100000 + step)
                row += 1
        np.save(root / "G.npy", g)
        np.save(root / "half_range.npy", np.full(7, 0.25, dtype=np.float32))
        np.save(root / "task_mean.npy", np.eye(d, dtype=np.float16))
        np.save(root / "trace_mean.npy", np.float32(3.0))
        (root / "meta.json").write_text(json.dumps({"chunk_h": CHUNK, "action_dim": ACT}))
        perm = np.random.default_rng(1234).permutation(n).astype(np.int64)
        rank = np.argsort(perm)  # rank[perm[j]] == j
        np.save(root / "rank.npy", rank)
        np.save(root / "perm.npy", perm)
        (root / "lengths.json").write_text(json.dumps({str(tid): t for tid, t in enumerate(lengths)}))
        return root, offsets, n

    def test_get_roundtrip_and_invalid(self, tmp_path):
        self._build_suite(tmp_path)
        cache = _oe_cache_mod().OutcomeGeometryCache(str(tmp_path), shuffle=False)
        pl = cache.get("suite_a", 0, 2)
        assert pl["valid"] and np.isfinite(np.asarray(pl["G"], dtype=np.float32)).all()
        np.testing.assert_allclose(np.asarray(pl["G"], dtype=np.float32), np.eye(4) * 3.0, rtol=1e-3)  # row 2
        bad = cache.get("suite_a", 0, 99)  # beyond valid domain
        assert not bad["valid"]
        bad2 = cache.get("suite_a", 7, 0)  # unknown episode
        assert not bad2["valid"]
        np.testing.assert_allclose(np.asarray(bad2["half_range"]), np.full(7, 0.25), rtol=1e-6)
        ep1 = cache.get("suite_a", 1, 3)  # ep1 rows start at offset 11 (T=12 -> 11 rows)
        assert ep1["valid"]
        np.testing.assert_allclose(np.asarray(ep1["G"], dtype=np.float32), np.eye(4) * 15.0, rtol=1e-3)

    def test_shuffle_permutation_is_within_suite(self, tmp_path):
        mod = _oe_cache_mod()
        root, _, n = self._build_suite(tmp_path)
        perm = np.load(root / "perm.npy")
        rank = np.load(root / "rank.npy")
        assert (perm != np.arange(n)).any()  # the pairing must actually be broken
        shuffled = mod.OutcomeGeometryCache(str(tmp_path), shuffle=True)
        for step in range(5):  # ep0 rows (T-1 = 11) are all valid; first 5 suffice
            expected_row = int(perm[int(rank[step])])
            b = np.asarray(shuffled.get("suite_a", 0, step)["G"], dtype=np.float32)
            # contract: idx' = perm[rank[idx]]
            np.testing.assert_allclose(b, np.eye(4) * (expected_row + 1.0), rtol=1e-3)
        # determinism
        fresh = mod.OutcomeGeometryCache(str(tmp_path), shuffle=True)
        again = np.asarray(fresh.get("suite_a", 0, 2)["G"], dtype=np.float32)
        assert np.allclose(again, np.asarray(shuffled.get("suite_a", 0, 2)["G"], dtype=np.float32))

    def test_validate_suites_fail_fast(self, tmp_path):
        self._build_suite(tmp_path)
        cache = _oe_cache_mod().OutcomeGeometryCache(str(tmp_path), shuffle=False)
        with pytest.raises(ValueError):
            cache.validate_suites(["suite_a", "missing_suite"])
