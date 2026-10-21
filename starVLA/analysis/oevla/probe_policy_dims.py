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
"""Offline per-dimension error probe for trained gate-B arms (zero training).

Replays a fixed, deterministic sample stream through ``predict_action`` — the
same call that drives closed-loop evaluation — and decomposes the action error
per dimension and per chunk step. Used to attribute a closed-loop collapse
(SR = 0) to a specific action channel.

The sample stream is deterministic because ``LeRobotMixtureDataset`` seeds each
index independently (``safe_hash((epoch, index, seed))``), so every checkpoint
loaded here sees exactly the same frames, instructions and targets.

Instrument check: the recomputed L1 on the first batch is reported next to the model's
own ``pure_action_loss`` on that same batch — the trainer logs that exact number, so
the two must agree to float precision.

Usage (cluster, GPU) — must run in the **StarVLA training environment** (torch,
transformers, accelerate, omegaconf, pyav); it never touches robosuite, so it does
not belong in the LIBERO eval environment, which has no accelerate:
    python -m starVLA.analysis.oevla.probe_policy_dims \
        --config-yaml examples/simBenchmarks/LIBERO/train_files/train_oeB_baseline.yaml \
        --ckpt baseline=.../steps_10000_pytorch_model.pt \
        --ckpt trace=.../steps_10000_pytorch_model.pt \
        --out-dir bar/probe_dims
"""

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

GRIPPER_IDX = 6
CLOSE_THRESHOLD = 0.5  # deployment threshold on the binary gripper channel


def _pearson(a, b):
    """Pearson correlation of two 1-D arrays; 0.0 when either side is constant."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def summarize(pred, target):
    """Decompose normalized action error per dimension and per chunk step.

    pred, target: [N, chunk, dim] arrays in the normalized action space used by
    the training loss and by ``unnormalize_actions`` at deployment.
    """
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if pred.shape != target.shape:
        raise ValueError(f"pred/target shape mismatch: {pred.shape} != {target.shape}")
    n, chunk, dim = pred.shape
    if dim <= GRIPPER_IDX:
        raise ValueError(f"expected at least {GRIPPER_IDX + 1} action dims, got {dim}")

    err = pred - target
    abs_err = np.abs(err)
    grip_pred = pred[..., GRIPPER_IDX]
    grip_tgt = target[..., GRIPPER_IDX]
    eef = slice(0, GRIPPER_IDX)
    sq_mass = np.square(err).sum(axis=(0, 1))

    result = {
        "n_samples": int(n),
        "chunk_len": int(chunk),
        "action_dim": int(dim),
        "action_loss": float(abs_err.mean()),
        "per_dim_l1": abs_err.mean(axis=(0, 1)).tolist(),
        "per_step_l1": abs_err.mean(axis=(0, 2)).tolist(),
        "per_dim_step_l1": abs_err.mean(axis=0).tolist(),
        "eef_l1": float(abs_err[..., eef].mean()),
        "gripper_l1": float(abs_err[..., GRIPPER_IDX].mean()),
        "gripper_error_share": float(sq_mass[GRIPPER_IDX] / max(sq_mass.sum(), 1e-12)),
        "gripper": {
            "close_rate_pred": (grip_pred >= CLOSE_THRESHOLD).mean(axis=0).tolist(),
            "close_rate_target": (grip_tgt >= CLOSE_THRESHOLD).mean(axis=0).tolist(),
            "closed_accuracy": float(((grip_pred >= CLOSE_THRESHOLD) == (grip_tgt >= CLOSE_THRESHOLD)).mean()),
            "pred_mean_when_target_closed": float(grip_pred[grip_tgt >= CLOSE_THRESHOLD].mean())
            if (grip_tgt >= CLOSE_THRESHOLD).any()
            else None,
            "pred_mean_when_target_open": float(grip_pred[grip_tgt < CLOSE_THRESHOLD].mean())
            if (grip_tgt < CLOSE_THRESHOLD).any()
            else None,
            "pred_p05_p50_p95": np.percentile(grip_pred, [5, 50, 95]).tolist(),
            "target_p05_p50_p95": np.percentile(grip_tgt, [5, 50, 95]).tolist(),
        },
        "state_blindness": {
            # Across-sample correlation at a fixed chunk step: ~1 means the
            # gripper output tracks the scene, ~0 means it ignores it.
            "gripper_r2_across_samples": [_pearson(grip_pred[:, t], grip_tgt[:, t]) ** 2 for t in range(chunk)],
            "gripper_pred_std": grip_pred.std(axis=0).tolist(),
            "gripper_target_std": grip_tgt.std(axis=0).tolist(),
        },
    }
    return result


def _load_state_dict(path):
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


def _load_into(model, state, label):
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise ValueError(f"{label}: {len(missing)} missing keys, e.g. {list(missing)[:5]}")
    if unexpected:
        raise ValueError(f"{label}: {len(unexpected)} unexpected keys, e.g. {list(unexpected)[:5]}")


def _to_device(obj, device):
    """Recursively move tensors to ``device`` — what accelerate's dataloader does during training.

    ``predict_action``/``forward`` consume ``examples[0]["qwen_inputs"]`` as handed to them, so
    without accelerate (no DataLoaderShard) the nested tensors would stay on the CPU. The
    processor returns a ``BatchFeature`` (a ``UserDict``), which is a Mapping but *not* a dict,
    hence the Mapping branch rather than an isinstance(obj, dict) one.
    """
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, Mapping):
        moved = {k: _to_device(v, device) for k, v in obj.items()}
        if type(obj) is dict:
            return moved
        try:
            return type(obj)(moved)
        except TypeError:
            return moved
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_device(v, device) for v in obj)
    return obj


def _check_preprocessed(examples):
    """The collate must have built qwen_inputs; otherwise predict_action builds them on the CPU."""
    if "qwen_inputs" not in examples[0]:
        raise SystemExit(
            "the loader did not preprocess qwen_inputs — set framework.qwenvl.preprocess_in_dataloader: true "
            "in the arm YAML (the training path relies on it too)"
        )


def collect(model, loader, batches):
    """Run ``predict_action`` on a fixed prefix of the loader; return (pred, target)."""
    horizon = int(model.action_horizon)
    device = next(model.parameters()).device
    preds, targets = [], []
    for index, examples in enumerate(loader):
        if index >= batches:
            break
        _check_preprocessed(examples)
        out = model.predict_action(examples=_to_device(examples, device))
        preds.append(np.asarray(out["normalized_actions"], dtype=np.float64))
        targets.append(np.asarray([np.asarray(ex["action"]) for ex in examples], dtype=np.float64)[:, -horizon:])
    if not preds:
        raise RuntimeError("dataloader produced no batches")
    return np.concatenate(preds), np.concatenate(targets)


def check_instrument(model, examples):
    """Reproduce the trainer's ``pure_action_loss`` on one batch (self-check)."""
    device = next(model.parameters()).device
    batch = _to_device(examples, device)
    with torch.no_grad():
        own = float(model(examples=batch)["pure_action_loss"])
    horizon = int(model.action_horizon)
    pred = np.asarray(model.predict_action(examples=batch)["normalized_actions"], dtype=np.float64)
    target = np.asarray([np.asarray(ex["action"]) for ex in examples], dtype=np.float64)[:, -horizon:]
    return {"pure_action_loss": own, "l1_recomputed": float(np.abs(pred - target).mean())}


def _print_table(results):
    labels = list(results)
    print("\n=== per-dimension error ===")
    header = (
        f"{'arm':<14}{'action_loss':>12}{'eef_l1':>10}{'grip_l1':>10}"
        f"{'grip_err%':>11}{'close_pred':>11}{'close_tgt':>10}"
    )
    print(header)
    for label in labels:
        r = results[label]
        print(
            f"{label:<14}{r['action_loss']:>12.4f}{r['eef_l1']:>10.4f}{r['gripper_l1']:>10.4f}"
            f"{100 * r['gripper_error_share']:>10.1f}%"
            f"{np.mean(r['gripper']['close_rate_pred']):>11.3f}{np.mean(r['gripper']['close_rate_target']):>10.3f}"
        )
    print("\n=== gripper close-rate per chunk step (pred | target) ===")
    for label in labels:
        grip = results[label]["gripper"]
        pred_row = " ".join(f"{v:.2f}" for v in grip["close_rate_pred"])
        tgt_row = " ".join(f"{v:.2f}" for v in grip["close_rate_target"])
        print(f"{label:<14} pred: {pred_row}")
        print(f"{'':<14} tgt : {tgt_row}")
    print("\n=== per-dim L1 (x,y,z,roll,pitch,yaw,grip) ===")
    for label in labels:
        print(f"{label:<14} " + " ".join(f"{v:.3f}" for v in results[label]["per_dim_l1"]))
    print("\n=== gripper state-blindness (R^2 across samples per step) ===")
    for label in labels:
        r2 = results[label]["state_blindness"]["gripper_r2_across_samples"]
        print(f"{label:<14} " + " ".join(f"{v:.2f}" for v in r2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config-yaml", required=True, help="any arm YAML (predict_action ignores framework.oe)")
    parser.add_argument("--ckpt", action="append", required=True, metavar="LABEL=PATH", help="repeatable")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--batches", type=int, default=24, help="microbatches replayed per checkpoint")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    ckpts = []
    for item in args.ckpt:
        if "=" not in item:
            raise SystemExit(f"--ckpt expects LABEL=PATH, got {item!r}")
        label, path = item.split("=", 1)
        if not Path(path).is_file():
            raise SystemExit(f"{label}: checkpoint not found: {path}")
        ckpts.append((label, path))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from omegaconf import OmegaConf

    from starVLA.dataloader import build_dataloader
    from starVLA.model.framework.base_framework import build_framework
    from starVLA.model.framework.share_tools import apply_config_compat

    cfg = apply_config_compat(OmegaConf.load(args.config_yaml))
    cfg.output_dir = str(out_dir)  # build_dataloader writes dataset statistics here

    model = build_framework(cfg)
    model.to(args.device)
    model.eval()

    loader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)
    dataset = getattr(loader, "dataset", None)
    if dataset is not None and callable(getattr(dataset, "set_epoch", None)):
        dataset.set_epoch(0)

    results = {}
    first_batch = next(iter(loader))
    for label, path in ckpts:
        _load_into(model, _load_state_dict(path), label)
        pred, target = collect(model, loader, args.batches)
        results[label] = summarize(pred, target)
        results[label]["checkpoint"] = path
        results[label]["instrument"] = check_instrument(model, first_batch)
        inst = results[label]["instrument"]
        print(
            f"[probe] {label}: {results[label]['n_samples']} samples; "
            f"pure_action_loss={inst['pure_action_loss']:.4f} vs recomputed={inst['l1_recomputed']:.4f}"
        )

    _print_table(results)
    payload = {
        "config_yaml": str(args.config_yaml),
        "batches": args.batches,
        "close_threshold": CLOSE_THRESHOLD,
        "gripper_idx": GRIPPER_IDX,
        "arms": results,
    }
    (out_dir / "probe_dims.json").write_text(json.dumps(payload, indent=2))
    print(f"\n[probe] wrote {out_dir / 'probe_dims.json'}")


if __name__ == "__main__":
    main()
