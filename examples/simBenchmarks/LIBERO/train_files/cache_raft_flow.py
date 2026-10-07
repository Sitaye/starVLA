"""
Offline RAFT flow-target cache builder for the motion-compatibility system.

For every (trajectory, step) of every suite in the data mixture, reads the current
frame and the future frame (base + delta_t) for each video view, runs frozen RAFT-Large
on the same resized PIL images the VLA sees (Vision Resolution Contract, docs/new_idea.md #25),
projects the flow onto the Qwen3.5 visual token grid in token-cell units, and stores float16 targets:

    <output_root>/<dataset_name>/flow.npy  [N, V, H_tok, W_tok, 2] float16
    <output_root>/<dataset_name>/keys.npy  [N] int64, key = trajectory_id * 100_000 + step

Only valid anchors (step + delta_t < trajectory length) get RAFT supervision; invalid rows
stay NaN so the training side masks them out instead of reading clamped self-flow (zero
motion) at episode ends. Each directory also gets a meta.json recording the cache contract.
Rows are aligned with ``dataset.all_steps`` order, so the training-side
``FlowTargetCache`` looks samples up by (dataset_name, trajectory_id, step).
Resume: pending NaN rows of valid anchors are recomputed; invalid rows stay NaN.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from omegaconf import OmegaConf
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
from torchvision.transforms.functional import pil_to_tensor

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from starVLA.dataloader.gr00t_lerobot.registry import DATASET_NAMED_MIXTURES
from starVLA.dataloader.gr00t_lerobot.video import get_frames_by_timestamps
from starVLA.dataloader.lerobot_datasets import make_LeRobotSingleDataset
from starVLA.model.modules.motion.head import flow_to_token_grid
from starVLA.model.modules.vlm.QWen3_5 import build_qwen_processor, qwen_apply_processor, qwen_build_messages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/simBenchmarks/LIBERO/train_files/starvla_cotrain_libero.yaml",
    )
    parser.add_argument("--output_root", type=str, default="./playground/RaftFlowCache")
    parser.add_argument("--delta_t", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--data_root_dir", type=str, default=None, help="Override datasets.vla_data.data_root_dir from the yaml"
    )
    parser.add_argument(
        "--base_vlm",
        type=str,
        default=None,
        help="Override framework.qwenvl.base_vlm from the yaml; MUST match the value used for training",
    )
    parser.add_argument(
        "--video_backend",
        type=str,
        default=None,
        help="Override datasets.vla_data.video_backend for this cache run only; "
        "use pyav to avoid the torchvision_av decoder memory leak",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = OmegaConf.load(args.config_yaml)
    data_cfg = cfg.datasets.vla_data
    if args.video_backend is not None:
        data_cfg.video_backend = args.video_backend
    if args.data_root_dir is not None:
        data_cfg.data_root_dir = args.data_root_dir
    data_root_dir = Path(data_cfg.data_root_dir)
    image_size = data_cfg.get("image_size", 224)
    if isinstance(image_size, int):
        image_size = (image_size, image_size)
    # Token grid must match the runtime VLM pipeline exactly (224 can be upscaled by min_pixels):
    # derive it from the real processor.
    if args.base_vlm is not None:
        cfg.framework.qwenvl.base_vlm = args.base_vlm
    processor = build_qwen_processor(cfg.framework.qwenvl.base_vlm)
    ip = processor.image_processor
    probe = qwen_apply_processor(
        processor, qwen_build_messages([[Image.new("RGB", image_size, 0)]], ["probe"])
    )
    grid = probe["image_grid_thw"][0].tolist()  # (t, h, w) in patch units
    patch, merge = int(ip.patch_size), int(ip.merge_size)
    token_hw = (grid[1] // merge, grid[2] // merge)
    print(
        f"[probe] image_processor={type(ip).__name__} resized_grid={(grid[1], grid[2])} "
        f"patch={patch} merge={merge} -> token_grid={token_hw}"
    )

    raft = raft_large(weights=Raft_Large_Weights.DEFAULT).to(device).eval()
    raft_transforms = Raft_Large_Weights.DEFAULT.transforms()

    for data_name, _, robot_type in DATASET_NAMED_MIXTURES[data_cfg.data_mix]:
        dataset = make_LeRobotSingleDataset(data_root_dir, data_name, robot_type, data_cfg=data_cfg)
        out_dir = Path(args.output_root) / dataset.dataset_name
        out_dir.mkdir(parents=True, exist_ok=True)
        all_steps = dataset.all_steps
        video_keys = dataset.modality_keys["video"]
        keys = np.array(
            [trajectory_id * 100_000 + base_index for trajectory_id, base_index in all_steps], dtype=np.int64
        )
        trajectory_length_by_id = dict(zip(dataset._trajectory_ids, dataset.trajectory_lengths, strict=True))
        delta0 = int(dataset.delta_indices[video_keys[0]][0])
        valid_anchor_count = sum(
            1
            for trajectory_id, base_index in all_steps
            if 0 <= base_index + delta0 + args.delta_t < trajectory_length_by_id[trajectory_id]
        )
        flow_path, keys_path = out_dir / "flow.npy", out_dir / "keys.npy"

        todo_mask = np.ones(len(all_steps), dtype=bool)
        mode = "w+"
        if keys_path.exists() and flow_path.exists():
            cached_meta = {}
            try:
                cached_meta = json.loads((out_dir / "meta.json").read_text())
            except (OSError, ValueError):
                pass  # missing/corrupt meta.json -> stale
            meta_stale = "num_valid_anchors" not in cached_meta
            cached = np.load(flow_path, mmap_mode="r")
            shape_ok = cached.shape[1:] == (len(video_keys), token_hw[0], token_hw[1], 2)
            if np.array_equal(np.load(keys_path), keys) and shape_ok and not meta_stale:
                todo_mask = np.isnan(np.asarray(cached[:, 0, 0, 0, 0]))
                mode = "r+"
            else:
                print(f"[{data_name}] stale cache (layout/shape/meta mismatch), rebuilding")
        if not todo_mask.any():
            print(f"[{data_name}] cache complete, skipping")
            continue

        V = len(video_keys)
        flow = np.lib.format.open_memmap(
            flow_path,
            mode=mode,
            dtype=np.float16,
            shape=(len(all_steps), V, token_hw[0], token_hw[1], 2),
        )
        if mode == "w+":
            flow[:] = np.nan
        np.save(keys_path, keys)

        rows_by_trajectory = {}
        for i in np.flatnonzero(todo_mask):
            rows_by_trajectory.setdefault(all_steps[i][0], []).append(i)
        for trajectory_id in sorted(rows_by_trajectory):
            rows = np.array(rows_by_trajectory[trajectory_id], dtype=np.int64)
            dataset.curr_traj_data = dataset.get_trajectory_data(trajectory_id)
            dataset.curr_traj_id = trajectory_id
            trajectory_index = dataset.get_trajectory_index(trajectory_id)
            trajectory_length = int(dataset.trajectory_lengths[trajectory_index])
            for view_index, video_key in enumerate(video_keys):
                stripped_key = video_key.replace("video.", "")
                original_key = dataset.lerobot_modality_meta.video[stripped_key].original_key
                if original_key is None:
                    original_key = stripped_key
                video_path = dataset.get_video_path(trajectory_id, stripped_key)
                timestamp = dataset.curr_traj_data["timestamp"].to_numpy()
                from_ts = 0.0
                if dataset._lerobot_version == "v3.0":
                    from_ts = float(
                        dataset.trajectory_ids_to_metadata.get(trajectory_id, {})
                        .get("videos/from_timestamps", {})
                        .get(original_key, 0.0)
                    )
                valid_rows, cur_ts_list, fut_ts_list = [], [], []
                for i in rows:
                    _, base_index = all_steps[i]
                    fut_index = int(dataset.delta_indices[video_key][0]) + base_index + args.delta_t
                    if fut_index < 0 or fut_index >= trajectory_length:
                        continue  # invalid anchor: row stays NaN, no clamped self-flow supervision
                    valid_rows.append(i)
                    cur_indices = np.minimum(
                        np.maximum(dataset.delta_indices[video_key] + base_index, 0), trajectory_length - 1
                    )
                    cur_ts_list.append(timestamp[cur_indices] + from_ts)
                    fut_ts_list.append(timestamp[np.array([fut_index])] + from_ts)
                if not valid_rows:
                    continue
                cur_ts = np.concatenate(cur_ts_list)
                fut_ts = np.concatenate(fut_ts_list)
                cur_frames = get_frames_by_timestamps(
                    video_path.as_posix(),
                    cur_ts,
                    video_backend=dataset.video_backend,
                    video_backend_kwargs=dataset.video_backend_kwargs,
                )
                fut_frames = get_frames_by_timestamps(
                    video_path.as_posix(),
                    fut_ts,
                    video_backend=dataset.video_backend,
                    video_backend_kwargs=dataset.video_backend_kwargs,
                )
                img1_batch = [Image.fromarray(frame).resize(image_size) for frame in cur_frames]
                img2_batch = [Image.fromarray(frame).resize(image_size) for frame in fut_frames]
                del cur_frames, fut_frames
                flow_pairs = []
                for batch_start in range(0, len(img1_batch), args.batch_size):
                    img1 = torch.stack(
                        [pil_to_tensor(im) for im in img1_batch[batch_start : batch_start + args.batch_size]]
                    )
                    img2 = torch.stack(
                        [pil_to_tensor(im) for im in img2_batch[batch_start : batch_start + args.batch_size]]
                    )
                    img1, img2 = raft_transforms(img1, img2)
                    img1, img2 = img1.to(device), img2.to(device)
                    with torch.inference_mode():
                        flow_full = raft(img1, img2)[-1]  # [B, 2, H, W]
                    flow_pairs.append(flow_to_token_grid(flow_full.float().cpu(), token_hw).numpy())
                flow[np.array(valid_rows, dtype=np.int64), view_index] = np.concatenate(flow_pairs, axis=0).astype(
                    np.float16
                )
        meta = {
            "dataset": dataset.dataset_name,
            "data_root_dir": str(data_root_dir / data_name),
            "canonical_size": [image_size[0], image_size[1]],
            "token_grid": [token_hw[0], token_hw[1]],
            "patch": patch,
            "merge": merge,
            "resized_grid": [grid[1], grid[2]],
            "stride": image_size[1] // token_hw[1],
            "delta_step": args.delta_t,
            "unit": "token-cell displacement (area-pooled RAFT flow divided by token stride)",
            "teacher": "torchvision.raft_large weights=Raft_Large_Weights.DEFAULT",
            "torchvision_version": torchvision.__version__,
            "views": list(video_keys),
            "num_rows": len(all_steps),
            "num_valid_anchors": valid_anchor_count,
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        print(f"[{data_name}] computed {int(todo_mask.sum())}/{len(all_steps)} steps")


if __name__ == "__main__":
    main()
