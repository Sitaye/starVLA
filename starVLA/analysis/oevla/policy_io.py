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
"""Shared policy/data plumbing for the gate-B offline instruments.

``calibrate_oe_lambda`` and ``probe_policy_dims`` must see exactly the model and
the sample stream that training sees, so the construction path lives in one
place instead of once per script: build the framework from an arm YAML, move it
to the device, load a checkpoint, and build the deterministic dataloader.

Cluster-only (torch, transformers, accelerate, omegaconf, pyav); never touches
robosuite, so it does not belong in the LIBERO eval environment.
"""

from collections.abc import Mapping
from pathlib import Path

import torch


def load_state_dict(path):
    """Read a StarVLA checkpoint, unwrap ``state_dict``, drop a ``module.`` prefix if present."""
    path = str(path)
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        state = load_file(path)
    else:
        state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
        state = state["state_dict"]
    if not isinstance(state, dict) or not state:
        raise ValueError(f"{path}: not a state dict")
    prefixed = sum(key.startswith("module.") for key in state)
    if prefixed and prefixed >= len(state) // 2:
        state = {key[len("module.") :] if key.startswith("module.") else key: value for key, value in state.items()}
    return state


def load_into(model, state, label):
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise ValueError(f"{label}: {len(missing)} missing keys, e.g. {list(missing)[:5]}")
    if unexpected:
        raise ValueError(f"{label}: {len(unexpected)} unexpected keys, e.g. {list(unexpected)[:5]}")


def load_checkpoint(model, path):
    """``load_into(load_state_dict(path))`` — the only way a checkpoint enters an instrument."""
    load_into(model, load_state_dict(path), Path(path).name)


def to_device(obj, device):
    """Recursively move tensors to ``device`` — what accelerate's dataloader does during training.

    ``predict_action``/``forward`` consume ``examples[0]["qwen_inputs"]`` as handed to them, so
    without accelerate (no DataLoaderShard) the nested tensors would stay on the CPU. The
    processor returns a ``BatchFeature`` (a ``UserDict``), which is a Mapping but *not* a dict,
    hence the Mapping branch rather than an isinstance(obj, dict) one.
    """
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, Mapping):
        moved = {k: to_device(v, device) for k, v in obj.items()}
        if type(obj) is dict:
            return moved
        try:
            return type(obj)(moved)
        except TypeError:
            return moved
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_device(v, device) for v in obj)
    return obj


def check_preprocessed(examples):
    """The collate must have built qwen_inputs; otherwise predict_action builds them on the CPU."""
    if "qwen_inputs" not in examples[0]:
        raise SystemExit(
            "the loader did not preprocess qwen_inputs — set framework.qwenvl.preprocess_in_dataloader: true "
            "in the arm YAML (the training path relies on it too)"
        )


def build_policy(config_yaml, out_dir, device):
    """Return ``(model, loader)`` for an arm YAML, with the epoch-0 sample stream fixed."""
    from omegaconf import OmegaConf

    from starVLA.dataloader import build_dataloader
    from starVLA.model.framework.base_framework import build_framework
    from starVLA.model.framework.share_tools import apply_config_compat

    cfg = apply_config_compat(OmegaConf.load(config_yaml))
    cfg.output_dir = str(out_dir)  # build_dataloader writes dataset statistics here
    model = build_framework(cfg)
    model.to(device)
    model.eval()

    loader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)
    dataset = getattr(loader, "dataset", None)
    if dataset is not None and callable(getattr(dataset, "set_epoch", None)):
        dataset.set_epoch(0)
    return model, loader
