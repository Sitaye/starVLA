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
"""Unit tests for the per-dimension error probe (numpy only; no GPU, no data)."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

GRIPPER_IDX = 6


def _probe_mod():
    """File-load the probe module (numpy+torch only; no package chain)."""
    spec = importlib.util.spec_from_file_location(
        "probe_policy_dims_under_test",
        Path(__file__).resolve().parents[1] / "starVLA/analysis/oevla/probe_policy_dims.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PROBE = _probe_mod()


def _world(n, chunk=8, dim=7, eef_err=0.05, grip_dead=True, seed=0):
    rng = np.random.default_rng(seed)
    target = rng.normal(scale=0.2, size=(n, chunk, dim))
    pred = target.copy()
    if dim <= GRIPPER_IDX:
        return pred, target
    target[..., GRIPPER_IDX] = (rng.random((n, chunk)) < 0.5).astype(np.float64)  # binary channel
    pred = target.copy()
    pred[..., :GRIPPER_IDX] += rng.normal(scale=eef_err, size=(n, chunk, GRIPPER_IDX))
    if grip_dead:
        pred[..., GRIPPER_IDX] = 0.1  # constant, never crosses the 0.5 close threshold
    return pred, target


def test_healthy_world_has_low_gripper_share():
    pred, target = _world(64, grip_dead=False)
    out = PROBE.summarize(pred, target)
    assert out["gripper_error_share"] < 0.35
    assert out["gripper"]["closed_accuracy"] > 0.9
    assert min(out["state_blindness"]["gripper_r2_across_samples"]) > 0.8


def test_dead_gripper_dominates_error_and_loses_state_tracking():
    pred, target = _world(64, grip_dead=True)
    out = PROBE.summarize(pred, target)
    # EEF cells are near-perfect, so nearly all squared error sits in the gripper
    assert out["gripper_error_share"] > 0.95
    assert out["eef_l1"] < 0.1
    assert out["gripper_l1"] == pytest.approx(float(np.abs(0.1 - target[..., GRIPPER_IDX]).mean()), rel=1e-6)
    # the channel never closes, while the demonstrations close half the time
    assert max(out["gripper"]["close_rate_pred"]) < 1e-9
    assert np.mean(out["gripper"]["close_rate_target"]) == pytest.approx(0.5, abs=0.1)
    assert out["gripper"]["closed_accuracy"] == pytest.approx(0.5, abs=0.12)
    # a constant prediction carries no scene information at any chunk step
    assert max(out["state_blindness"]["gripper_r2_across_samples"]) < 1e-9


def test_shapes_and_layout():
    pred, target = _world(9, chunk=3, dim=7)
    out = PROBE.summarize(pred, target)
    assert out["n_samples"] == 9 and out["chunk_len"] == 3 and out["action_dim"] == 7
    assert len(out["per_dim_l1"]) == 7
    assert len(out["per_step_l1"]) == 3
    assert np.asarray(out["per_dim_step_l1"]).shape == (3, 7)
    assert out["action_loss"] == pytest.approx(float(np.abs(pred - target).mean()), rel=1e-9)


def test_shape_mismatch_raises():
    pred, target = _world(4)
    with pytest.raises(ValueError, match="shape mismatch"):
        PROBE.summarize(pred, target[:, :4])


def test_too_few_dims_raises():
    pred, target = _world(4, dim=5)
    with pytest.raises(ValueError, match="at least 7 action dims"):
        PROBE.summarize(pred, target)


def test_pearson_handles_constant_side():
    assert PROBE._pearson(np.zeros(5), np.arange(5.0)) == 0.0
    assert PROBE._pearson(np.arange(5.0), np.arange(5.0)) == pytest.approx(1.0)


def test_to_device_reaches_into_a_userdict_batch_feature():
    """The processor returns a BatchFeature (UserDict): a Mapping, but not a dict."""
    import collections

    import torch

    class Feature(collections.UserDict):
        pass

    tensors = Feature({
        "input_ids": torch.zeros(2, dtype=torch.long),
        "attention_mask": torch.ones(2, dtype=torch.long),
        "pixel_values": torch.zeros(1),
    })
    batch = [{"qwen_inputs": tensors, "action": np.zeros((1, 8, 7)), "lang": "pick"}]
    moved = PROBE._to_device(batch, torch.device("meta"))

    assert isinstance(moved[0]["qwen_inputs"], Feature)
    for value in moved[0]["qwen_inputs"].values():
        assert value.device.type == "meta"
    assert moved[0]["lang"] == "pick"  # non-tensor leaves untouched


def test_to_device_keeps_plain_containers_and_numpy():
    import torch

    batch = [{"qwen_inputs": {"input_ids": torch.zeros(1, dtype=torch.long)}, "action": np.ones((1, 8, 7))}]
    moved = PROBE._to_device(batch, torch.device("meta"))
    assert isinstance(moved[0]["qwen_inputs"], dict)
    assert moved[0]["qwen_inputs"]["input_ids"].device.type == "meta"
    assert isinstance(moved[0]["action"], np.ndarray)


def test_check_preprocessed_fails_loudly():
    with pytest.raises(SystemExit, match="preprocess_in_dataloader"):
        PROBE._check_preprocessed([{"action": np.zeros((1, 8, 7))}])
    PROBE._check_preprocessed([{"qwen_inputs": {}}])  # present -> no raise


def test_variance_ratio_detects_mean_seeking():
    pred, target = _world(64, grip_dead=False)
    healthy = PROBE.summarize(pred, target)
    assert all(0.8 < r < 1.3 for r in healthy["variance_ratio"])
    assert 0.8 < healthy["overall_variance_ratio"] < 1.3

    # A squared-error head on a bimodal channel regresses to the conditional mean:
    # constant output -> zero variance on the gripper dim, +-1 target.
    flat = target.copy()
    flat[..., GRIPPER_IDX] = 0.06
    collapsed = PROBE.summarize(flat, target)
    assert collapsed["variance_ratio"][GRIPPER_IDX] == pytest.approx(0.0, abs=1e-9)
    assert collapsed["gripper"]["close_rate_pred"] == pytest.approx([0.0] * target.shape[1])
    # the continuous dims are untouched, so their ratio stays near 1
    assert all(0.8 < r < 1.3 for r in collapsed["variance_ratio"][:GRIPPER_IDX])
