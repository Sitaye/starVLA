# Copyright 2026 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""RAP-FSD Stage A component 2: pi0 prediction export.

Batch inference over every transition of every manifest episode: exports the
ground-truth normalized action chunk A [T*7 = 56] (8x7 delta qpos normalized), the
model prediction pred_A0 [56] (normalized_actions from the pi0-style action expert)
and the context features C = concat(instruction embedding z_lang, per-view pooled
E_vis, robot state), plus optionally the raw frozen visual entry embeddings E_vis
[B, V, Ht, Wt, D] for holdout episodes with a stride-s subsample.

Model loading mirrors deployment/model_server/server_policy.py: baseframework
.from_pretrained(ckpt_path) -> .to(device).eval(); predictions are read from the
framework's own ``predict_action`` code path so train/test consistency is preserved.
The context extraction mirrors QwenOFT.forward (entry embeddings captured through a
forward pre-hook on the language model, exactly like starVLA/model/framework/VLM4A/
QwenOFT.py:495-512 ``_gather_visual_token_embeddings``).

IMPORTANT: this module must stay importable WITHOUT torch (no GPU / no data on the
dev machine; pytest imports it). All torch / starVLA model imports are lazy, inside
functions. Use --dry-run to self-check manifest + data args without loading anything.

Shard contract (consumed by run_stage_a.py): one npz per episode at
``<output_dir>/<episode_id with "/" -> "__">.npz`` with
  A [T,56] fp32, pred_A0 [T,56] fp32, C [T,D_C] fp32, steps [T] int64, optional evis
  [Te,V,Ht,Wt,D] fp16 and evis_rows [Te] int64 (stride-subsampled rows), and a 0-d
  string ``meta_json`` {episode_id, suite, task_id, c_layout, n_steps}.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .split import suite_from_dataset_name

C_LAYOUT_LANG = "lang"
C_LAYOUT_EVIS_POOLED = "evis_pooled"
C_LAYOUT_STATE = "state"

SEED_EXPORT_DEFAULT = 42


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--ckpt-path", required=True, help="starVLA checkpoint directory")
    parser.add_argument("--manifest", required=True, help="split manifest JSON from split.py")
    parser.add_argument("--data-args", required=True, help="training yaml with datasets.vla_data config")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=1, help="row subsample stride (1 = all transitions)")
    parser.add_argument(
        "--export-evis",
        choices=("none", "holdout"),
        default="none",
        help="also export raw E_vis entry embeddings (holdout episodes only, stride subsampled)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use-bf16", action="store_true")
    parser.add_argument("--limit-episodes", type=int, default=0, help="debug: export only the first N episodes")
    parser.add_argument("--shard-index", type=int, default=0, help="this process's shard (0-based)")
    parser.add_argument("--num-shards", type=int, default=1, help="total export shards, e.g. 8 for 8 GPUs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="self-check inputs, no torch import")
    return parser.parse_args(argv)


def _load_data_cfg(data_args: str):
    """Load datasets.vla_data from the training yaml (lazy OmegaConf import)."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(data_args)
    try:
        return cfg.datasets.vla_data
    except (AttributeError, KeyError) as exc:
        raise RuntimeError(f"{data_args} has no datasets.vla_data section") from exc


def _load_framework(ckpt_path: str, device: str, use_bf16: bool):
    """Load the policy exactly like deployment/model_server/server_policy.py."""
    import torch

    from starVLA.model.framework.base_framework import baseframework

    framework = baseframework.from_pretrained(ckpt_path)
    if use_bf16:
        framework = framework.to(torch.bfloat16)
    framework = framework.to(device).eval()
    torch.manual_seed(SEED_EXPORT_DEFAULT)
    return framework



def _find_qwen_oft(framework):
    """Locate the QwenOFT module inside the framework (the module that owns the
    visual-token gather / instruction pooling helpers)."""
    candidates = []
    for module in framework.modules():
        if hasattr(module, "_gather_visual_token_embeddings") and hasattr(
            module, "_pool_instruction_entry_embeddings"
        ):
            candidates.append(module)
    if not candidates:
        raise RuntimeError(
            "no QwenOFT-like module found (needs _gather_visual_token_embeddings + "
            "_pool_instruction_entry_embeddings); context export requires the QwenOFT framework"
        )
    if len(candidates) > 1:
        raise RuntimeError(f"ambiguous QwenOFT modules: {[type(c).__name__ for c in candidates]}")
    return candidates[0]


def _build_dataset(data_cfg, seed: int):
    """Build the mixture dataset (lazy import; single datasets enumerated directly)."""
    from starVLA.dataloader.lerobot_datasets import get_vla_dataset

    return get_vla_dataset(data_cfg=data_cfg, mode="test", seed=seed)


def _dataset_episodes(dataset) -> dict[int, list[int]]:
    """trajectory_id -> sorted list of dataset row indices (all_steps enumeration)."""
    mapping: dict[int, list[int]] = {}
    for idx, (trajectory_id, base_index) in enumerate(dataset.all_steps):
        mapping.setdefault(int(trajectory_id), []).append((int(base_index), idx))
    return {traj: [idx for _, idx in sorted(pairs)] for traj, pairs in mapping.items()}


def _predict_batch(qwen_oft, examples: list[dict], device: str) -> dict:
    """Single forward pass returning normalized actions AND context features.

    Mirrors QwenOFT.forward / predict_action: entry embeddings are captured with a
    forward pre-hook on the language model (same mechanism as QwenOFT.forward), the
    visual entry embeddings are gathered per view with the framework's own
    ``_gather_visual_token_embeddings`` and pooled per view for the context block.
    """
    import torch

    # Export-only process: nothing here calls .backward(), but without disabling grad the
    # backbone forward retains a full autograd graph (peak GPU memory was ~10x a no_grad
    # forward and OOMed large batches).
    torch.set_grad_enabled(False)

    from deployment.model_server.tools.image_tools import to_pil_preserve
    from starVLA.model.framework.VLM4A.QwenOFT import assemble_prompts
    from starVLA.training.trainer_utils.trainer_tools import resize_images

    batch_images = [to_pil_preserve(example["image"]) for example in examples]
    instructions = assemble_prompts(examples, qwen_oft.chunk_len, qwen_oft.action_token)
    train_obs_image_size = getattr(qwen_oft.config.datasets.vla_data, "obs_image_size", None)
    if train_obs_image_size:
        batch_images = resize_images(batch_images, target_size=train_obs_image_size)
    qwen_inputs = qwen_oft.qwen_vl_interface.build_qwenvl_inputs(
        images=batch_images, instructions=instructions
    )
    entry_embeds = None

    def _stash_entry_embeds(module, args, kwargs):
        nonlocal entry_embeds
        entry_embeds = kwargs["inputs_embeds"]

    hook_handle = qwen_oft.qwen_vl_interface.model.model.language_model.register_forward_pre_hook(
        _stash_entry_embeds, with_kwargs=True
    )
    try:
        with torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
            qwenvl_outputs = qwen_oft.qwen_vl_interface.model.model(
                **qwen_inputs, output_attentions=False, return_dict=True
            )
            last_hidden = qwenvl_outputs.last_hidden_state  # [B, L, H]
    finally:
        hook_handle.remove()

    with torch.autocast(torch.device(device).type, dtype=torch.float32):
        input_ids = qwen_inputs.get("input_ids", None)
        action_queries = qwen_oft._gather_action_token_embeddings(
            last_hidden, input_ids, action_token_id=qwen_oft.action_token_id
        )
        pred_actions = qwen_oft.action_model.predict_action(action_queries)  # [B, T, D]

    num_views = len(examples[0]["image"])
    visual_mask = qwen_inputs["input_ids"] == qwen_oft.qwen_vl_interface.model.config.image_token_id
    evis = qwen_oft._gather_visual_token_embeddings(
        entry_embeds, visual_mask, qwen_inputs["image_grid_thw"], num_views
    )  # [B, V, Ht, Wt, D]
    z_lang = qwen_oft._pool_instruction_entry_embeddings(entry_embeds, qwen_inputs, examples)  # [B, D]

    evis_np = evis.detach().float().cpu().numpy()
    evis_pooled = evis_np.mean(axis=(2, 3))  # [B, V, D]
    state = None
    if "state" in examples[0]:
        state = np.stack(
            [np.asarray(example["state"], dtype=np.float64).reshape(-1) for example in examples]
        )
    return {
        "normalized_actions": np.asarray(pred_actions.detach().float().cpu().numpy()),
        "z_lang": z_lang.detach().float().cpu().numpy(),
        "evis": evis_np,
        "evis_pooled": evis_pooled.reshape(evis_pooled.shape[0], -1),
        "state": state,
        "num_views": num_views,
    }


def _c_layout(d_lang: int, n_views: int, state_dim: int) -> dict:
    layout = {
        C_LAYOUT_LANG: [0, d_lang],
        C_LAYOUT_EVIS_POOLED: [d_lang, n_views * d_lang],
    }
    if state_dim:
        layout[C_LAYOUT_STATE] = [d_lang + n_views * d_lang, state_dim]
    return layout


def _assemble_context(pred_out: dict) -> np.ndarray:
    c_parts = [pred_out["z_lang"], pred_out["evis_pooled"]]
    if pred_out.get("state") is not None:
        c_parts.append(pred_out["state"])
    return np.concatenate(c_parts, axis=1).astype(np.float32)


def _action_chunk(sample: dict, horizon: int) -> np.ndarray:
    action = np.asarray(sample["action"], dtype=np.float64)
    if action.ndim == 1:
        action = action[None, :]
    return action[-horizon:, :].reshape(-1)


def run_export(args: argparse.Namespace) -> None:
    """Full export pass; every torch import happens inside here (lazy)."""
    from .split import load_manifest

    manifest = load_manifest(args.manifest)
    episodes = dict(manifest["episodes"])
    all_ids = sorted(episodes)
    if args.num_shards > 1:
        # 8-GPU sharding: each process exports every num_shards-th episode (sorted order).
        episodes = {ep: episodes[ep] for ep in all_ids[args.shard_index :: args.num_shards]}
        print(f"[export] shard {args.shard_index}/{args.num_shards}: {len(episodes)} episodes")
    if args.limit_episodes:
        keep = sorted(episodes)[: args.limit_episodes]
        episodes = {ep: episodes[ep] for ep in keep}
    wanted = set(episodes)

    data_cfg = _load_data_cfg(args.data_args)
    framework = _load_framework(args.ckpt_path, args.device, args.use_bf16)
    qwen_oft = _find_qwen_oft(framework)
    dataset = _build_dataset(data_cfg, args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    horizon = int(qwen_oft.action_horizon)
    state_rows: dict[str, dict] = {}
    skip_ledger: dict[str, dict] = {}  # episode_id -> {"skipped": int, "errors": {type: count}}
    batch_samples: list[tuple[str, int, dict]] = []  # (episode_id, step, sample)

    def _flush() -> None:
        if not batch_samples:
            return
        t0 = time.time()
        samples = [s for _, _, s in batch_samples]
        pred_out = _predict_batch(qwen_oft, samples, args.device)
        _add_batch_rows(state_rows, batch_samples, pred_out, horizon)
        n = len(batch_samples)
        del batch_samples[:]
        print(f"[export] batch of {n} done in {time.time() - t0:.1f}s")

    for single in dataset.datasets:
        dataset_name = single.dataset_name
        traj_map = _dataset_episodes(single)
        for traj, row_indices in sorted(traj_map.items()):
            episode_id = f"{dataset_name}/{traj:05d}"
            if episode_id not in wanted:
                continue
            suite = suite_from_dataset_name(dataset_name)
            task = episodes[episode_id].get("task", "")
            state_rows[episode_id] = {
                "A": [],
                "pred_A0": [],
                "steps": [],
                "z_lang": [],
                "evis_pooled": [],
                "state": [],
                "evis_rows": [],
                "want_evis": args.export_evis == "holdout"
                and episodes[episode_id]["labels"].get(str(args_holdout_size(manifest))) == "holdout",
                "suite": suite,
                "task": task,
            }
            for local_idx in row_indices[:: max(args.stride, 1)]:
                try:
                    sample = single[local_idx]
                except Exception as exc:
                    # Corrupt/truncated videos surface here (pyav InvalidDataError, fs
                    # OSError): skip the row and record it. Anything else is our bug -
                    # still raise.
                    if not _is_decode_error(exc):
                        raise
                    step = int(single.all_steps[local_idx][1])
                    _record_skip(skip_ledger, episode_id, step, exc)
                    continue
                step = int(single.all_steps[local_idx][1])
                batch_samples.append((episode_id, step, sample))
                if len(batch_samples) >= args.batch_size:
                    _flush()
    _flush()
    # Drop fully-empty episodes (every row unreadable): they carry no data and an all-
    # zero npz would poison the keys join downstream.
    empty_eps = [ep for ep, rec in state_rows.items() if not rec["steps"]]
    for ep in empty_eps:
        del state_rows[ep]
    _write_skip_ledger(out_dir, skip_ledger, empty_eps, args, state_rows)
    _write_shards(out_dir, state_rows, manifest, horizon)



def _add_batch_rows(state_rows: dict, batch_samples: list, pred_out: dict, horizon: int) -> None:
    """Route one batch's outputs into per-episode accumulators."""
    for i, (episode_id, step, sample) in enumerate(batch_samples):
        rec = state_rows[episode_id]
        rec["A"].append(_action_chunk(sample, horizon))
        rec["pred_A0"].append(pred_out["normalized_actions"][i].reshape(-1))
        rec["steps"].append(step)
        rec["z_lang"].append(pred_out["z_lang"][i])
        rec["evis_pooled"].append(pred_out["evis_pooled"][i])
        if pred_out["state"] is not None:
            rec["state"].append(pred_out["state"][i])
        if rec["want_evis"]:
            rec["evis_rows"].append((step, pred_out["evis"][i]))


def args_holdout_size(manifest: dict) -> str:
    return str(manifest.get("primary_holdout_size", 2500))


def _is_decode_error(exc: BaseException) -> bool:
    """True for unreadable-media failures (pyav decode/open errors, fs OSError).

    Dataset video copies can contain corrupt packets (av.error.InvalidDataError et
    al.); those rows are skipped and recorded. Everything else - i.e. our own bugs -
    must propagate.
    """
    return exc.__class__.__module__.startswith("av.") or isinstance(exc, OSError)


def _record_skip(ledger: dict, episode_id: str, step: int, exc: BaseException) -> None:
    rec = ledger.setdefault(episode_id, {"skipped": 0, "first_steps": [], "errors": {}})
    rec["skipped"] += 1
    if len(rec["first_steps"]) < 20:
        rec["first_steps"].append(step)  # for cross-checking against the video file
    name = f"{exc.__class__.__module__}.{exc.__class__.__name__}"
    rec["errors"][name] = rec["errors"].get(name, 0) + 1
    if rec["skipped"] <= 3:
        print(f"[export] skip {episode_id} step {step}: {name}: {exc}")


def _write_skip_ledger(
    out_dir: Path,
    ledger: dict,
    empty_eps: list[str],
    args: argparse.Namespace,
    state_rows: dict,
) -> None:
    """Record skipped/unreadable rows per episode (one file per shard, no collision)."""
    if not ledger and not empty_eps:
        return
    payload = {
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "skipped_rows_total": int(sum(rec["skipped"] for rec in ledger.values())),
        "episodes": {
            ep: {
                "skipped_rows": rec["skipped"],
                "first_skipped_steps": rec["first_steps"],
                "errors": rec["errors"],
                "exported_rows": len(state_rows.get(ep, {}).get("steps", [])),
                "fully_excluded": ep in empty_eps,
            }
            for ep, rec in sorted(ledger.items())
        },
    }
    path = out_dir / f"excluded_shard_{args.shard_index}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    n_eps = len(payload["episodes"])
    n_full = sum(1 for rec in payload["episodes"].values() if rec["fully_excluded"])
    print(f"[export] shard {args.shard_index}: skipped {payload['skipped_rows_total']} rows in "
          f"{n_eps} episodes ({n_full} fully excluded) -> {path}")


def _write_shards(
    out_dir: Path,
    state_rows: dict,
    manifest: dict,
    horizon: int,
) -> None:
    for episode_id, rec in state_rows.items():
        a_arr = np.stack(rec["A"]) if rec["A"] else np.zeros((0, horizon * 7))
        pred_arr = np.stack(rec["pred_A0"]) if rec["pred_A0"] else np.zeros((0, horizon * 7))
        steps = np.asarray(rec["steps"], dtype=np.int64)
        c_arr = _assemble_context(
            {
                "z_lang": np.stack(rec["z_lang"]) if rec["z_lang"] else np.zeros((0, 0)),
                "evis_pooled": np.stack(rec["evis_pooled"]) if rec["evis_pooled"] else np.zeros((0, 0)),
                "state": np.stack(rec["state"]) if rec["state"] else None,
            }
        )
        payload = {
            "A": a_arr.astype(np.float32),
            "pred_A0": pred_arr.astype(np.float32),
            "C": c_arr,
            "steps": steps,
        }
        if rec["evis_rows"]:
            rows = np.asarray([r[0] for r in rec["evis_rows"]], dtype=np.int64)
            evis = np.stack([r[1] for r in rec["evis_rows"]]).astype(np.float16)
            payload["evis"] = evis
            payload["evis_rows"] = rows
        meta = {
            "episode_id": episode_id,
            "suite": rec["suite"],
            "task_id": rec["task"],
            "c_layout": _c_layout_from_record(rec),
            "n_steps": _safe_int(steps.shape[0], "step count"),
            "horizon": horizon,
        }
        payload["meta_json"] = np.array(json.dumps(meta))
        shard = out_dir / (episode_id.replace("/", "__") + ".npz")
        try:
            np.savez_compressed(shard, **payload)
        except OSError as exc:
            raise RuntimeError(f"failed to write shard {shard}: {exc}") from exc
        print(f"[export] wrote {shard} ({payload['A'].shape[0]} steps)")


def _safe_int(value, what: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{what}: not an integer ({value!r})") from exc


def _c_layout_from_record(rec: dict) -> dict:
    d_lang = _safe_int(np.stack(rec["z_lang"]).shape[-1], "lang dim") if rec["z_lang"] else 0
    n_views = _safe_int(np.stack(rec["evis_pooled"]).shape[-1] // d_lang, "view count") if d_lang else 0
    state_dim = _safe_int(np.stack(rec["state"]).shape[-1], "state dim") if rec["state"] else 0
    return _c_layout(d_lang, n_views, state_dim)


def dry_run(args: argparse.Namespace) -> None:
    """Self-check without torch: manifest + data args + output dir."""
    from .split import load_manifest

    manifest = load_manifest(args.manifest)
    episodes = manifest["episodes"]
    counts = {}
    for rec in episodes.values():
        for lab in rec["labels"].values():
            counts[lab] = counts.get(lab, 0) + 1
    print(f"[dry-run] manifest protocol: {manifest.get('protocol')}")
    print(f"[dry-run] episodes: {len(episodes)}; label counts: {counts}")
    print(f"[dry-run] data args: {args.data_args}")
    print(f"[dry-run] export_evis: {args.export_evis}; stride: {args.stride}; batch_size: {args.batch_size}")
    print("[dry-run] OK (torch/model loading deferred to a real run on a GPU node)")


def main() -> None:
    args = parse_args()
    if args.dry_run:
        dry_run(args)
        return
    if args.batch_size < 1 or args.stride < 1:
        raise RuntimeError("--batch-size and --stride must be >= 1")
    if not (0 <= args.shard_index < args.num_shards):
        raise RuntimeError("--shard-index must be in [0, --num-shards)")
    run_export(args)


if __name__ == "__main__":
    main()
