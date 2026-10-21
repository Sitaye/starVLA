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
"""Unit tests for the shared policy/data plumbing (torch only; no GPU, no accelerate)."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _policy_io_mod():
    """File-load the module (its framework/dataloader imports are function-local)."""
    spec = importlib.util.spec_from_file_location(
        "policy_io_under_test",
        Path(__file__).resolve().parents[1] / "starVLA/analysis/oevla/policy_io.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


POLICY_IO = _policy_io_mod()


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
    moved = POLICY_IO.to_device(batch, torch.device("meta"))

    assert isinstance(moved[0]["qwen_inputs"], Feature)
    for value in moved[0]["qwen_inputs"].values():
        assert value.device.type == "meta"
    assert moved[0]["lang"] == "pick"  # non-tensor leaves untouched


def test_to_device_keeps_plain_containers_and_numpy():
    import torch

    batch = [{"qwen_inputs": {"input_ids": torch.zeros(1, dtype=torch.long)}, "action": np.ones((1, 8, 7))}]
    moved = POLICY_IO.to_device(batch, torch.device("meta"))
    assert isinstance(moved[0]["qwen_inputs"], dict)
    assert moved[0]["qwen_inputs"]["input_ids"].device.type == "meta"
    assert isinstance(moved[0]["action"], np.ndarray)


def test_check_preprocessed_fails_loudly():
    with pytest.raises(SystemExit, match="preprocess_in_dataloader"):
        POLICY_IO.check_preprocessed([{"action": np.zeros((1, 8, 7))}])
    POLICY_IO.check_preprocessed([{"qwen_inputs": {}}])  # present -> no raise


def test_load_state_dict_unwraps_and_strips_module_prefix(tmp_path):
    import torch

    path = tmp_path / "ckpt.pt"
    torch.save({"state_dict": {"module.a": torch.zeros(1), "module.b": torch.ones(1)}}, path)
    state = POLICY_IO.load_state_dict(path)
    assert sorted(state) == ["a", "b"]

    bad = tmp_path / "empty.pt"
    torch.save({}, bad)
    with pytest.raises(ValueError, match="not a state dict"):
        POLICY_IO.load_state_dict(bad)


def test_load_into_reports_missing_and_unexpected_keys():
    import torch

    model = torch.nn.Linear(2, 1)
    POLICY_IO.load_into(model, dict(model.state_dict()), "ok")  # exact -> no raise

    with pytest.raises(ValueError, match="missing keys"):
        POLICY_IO.load_into(model, {}, "empty")
    with pytest.raises(ValueError, match="unexpected keys"):
        POLICY_IO.load_into(model, {**model.state_dict(), "extra": torch.zeros(1)}, "extra")
