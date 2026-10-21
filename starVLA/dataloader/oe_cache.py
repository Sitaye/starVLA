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
"""Read-only view over the precomputed outcome-geometry cache (gate B arms).

One directory per dataset under ``root`` (written by
``starVLA.analysis.oevla.precompute_g``):

  ``G.npy``          ([N, d, d] float16 memmap, d = chunk_len * action_dim) raw
                     action-space outcome geometry G_t = J_t^T W J_t at every
                     chunk-complete transition; NaN rows mark invalid
                     transitions (unmatched episode).
  ``lengths.json``   {episode_index: length} per episode.
  ``meta.json``      {"chunk_h": action chunk length} — the row domain is
                     T - chunk_h + 1 transitions per episode (precompute
                     writes it; the cache derives per-episode row counts
                     from it rather than G.shape).
  ``half_range.npy`` ([action_dim] float32) per-dim (max - min) / 2 of the
                     suite's raw actions in training action-key order; the
                     passthrough gripper dim stays 1.0.
  ``task_mean.npy``  ([d, d] float16) valid-row mean of the per-state
                     normalized geometry (fixed task-space arm).
  ``trace_mean.npy`` (scalar float32) valid-row mean of mean(diag(G)).
  ``rank.npy``/``perm.npy`` ([N]/[N_valid] int64, optional) shuffled-arm
                     permutation within the suite (seed 42, precompute).

Rows are indexed by trajectory layout (episode-ordered, contiguous), so no
key file is needed: episode ``tid`` owns rows ``[offset[tid], offset[tid] +
length - chunk_len + 1)``. Invalid transitions return an all-NaN G row with
``valid=False`` — the sample stays in the batch and only drops out of the OE
term, keeping the sample stream identical across all six gate-B arms.
"""

import json
from pathlib import Path

import numpy as np


class OutcomeGeometryCache:
    """Memmap-backed per-sample OE payload reader (flow_cache.py precedent)."""

    def __init__(self, root: str, shuffle: bool = False):
        self.root = root
        self.shuffle = shuffle
        self._entries = {}

    def suites(self):
        return {p.name for p in Path(self.root).iterdir() if p.is_dir()}

    def validate_suites(self, dataset_names):
        """Fail fast at dataset construction (outside the __getitem__ retry
        loop, which would silently resample around a config error)."""
        missing = sorted(set(dataset_names) - self.suites())
        if missing:
            raise ValueError(f"{self.root}: no OE geometry precomputed for {missing}")
        try:
            manifest = json.loads((Path(self.root) / "manifest.json").read_text())
        except (OSError, json.JSONDecodeError) as e:
            raise ValueError(f"{self.root}: unreadable manifest.json — incomplete precompute ({e})") from e
        not_done = sorted(set(dataset_names) - set(manifest.get("suites", {})))
        if not_done:
            raise ValueError(f"{self.root}: precompute manifest lacks {not_done} (incomplete run?)")
        failed = int(manifest.get("failed_episodes", 0))
        if failed:
            print(f"[oe_cache] WARNING: precompute had {failed} failed episodes (NaN rows -> base-loss-only samples)")

    def _load(self, dataset_name: str):
        if dataset_name not in self._entries:
            directory = Path(self.root) / dataset_name
            G = np.load(directory / "G.npy", mmap_mode="r")
            try:
                lengths_raw = json.loads((directory / "lengths.json").read_text())
            except (OSError, json.JSONDecodeError) as e:
                raise ValueError(f"{directory}: unreadable lengths.json ({e})") from e
            lengths = {int(k): int(v) for k, v in lengths_raw.items()}
            try:
                meta = json.loads((directory / "meta.json").read_text())
            except (OSError, json.JSONDecodeError) as e:
                raise ValueError(f"{directory}: unreadable meta.json ({e})") from e
            if "chunk_h" not in meta:
                raise ValueError(f"{directory}: meta.json missing chunk_h (stale cache?)")
            chunk_h = int(meta["chunk_h"])
            if G.shape[1] != chunk_h * int(meta["action_dim"]):
                raise ValueError(
                    f"{directory}: G d-dim {G.shape[1]} != chunk_h {chunk_h} x action_dim {meta['action_dim']}"
                    " (stale cache?)"
                )
            offsets, start = {}, 0
            for tid in sorted(lengths):
                offsets[tid] = start
                start += max(lengths[tid] - chunk_h + 1, 0)
            if start != G.shape[0]:
                # The row domain is defined twice (writer: precompute_g.n_rows, reader: here);
                # any drift silently pairs a state with another state's geometry.
                raise ValueError(f"{directory}: G.npy has {G.shape[0]} rows but lengths.json implies {start}")
            half_range = np.load(directory / "half_range.npy").astype(np.float32)
            task_mean = np.load(directory / "task_mean.npy").astype(np.float32)
            trace_mean = float(np.load(directory / "trace_mean.npy"))
            # e_raw = e_norm * half_range and the per-mode normalizers all divide by a trace,
            # so a non-finite or zero statistic turns the geometry term into silent garbage.
            for stat_name, stat, positive in (
                ("half_range.npy", half_range, True),
                ("task_mean.npy", task_mean, False),
                ("trace_mean.npy", trace_mean, True),
            ):
                if not np.isfinite(stat).all():
                    raise ValueError(f"{directory}: {stat_name} has non-finite entries")
                if positive and not (np.asarray(stat) > 0).all():
                    raise ValueError(f"{directory}: {stat_name} must be strictly positive")
            rank = perm = None
            if self.shuffle:
                rank = np.load(directory / "rank.npy")
                perm = np.load(directory / "perm.npy")
            self._entries[dataset_name] = (G, lengths, offsets, chunk_h, half_range, task_mean, trace_mean, rank, perm)
        return self._entries[dataset_name]

    def get(self, dataset_name: str, trajectory_id: int, step: int) -> dict:
        """Per-sample OE payload: {G, valid, half_range, task_mean, trace_mean}."""
        G, lengths, offsets, chunk_h, half_range, task_mean, trace_mean, rank, perm = self._load(dataset_name)
        payload = {
            "G": np.empty((0,)),  # placeholder, replaced below
            "valid": True,
            "half_range": half_range,
            "task_mean": task_mean,
            "trace_mean": trace_mean,
        }
        row = offsets.get(trajectory_id)
        n_rows = max(lengths.get(trajectory_id, 0) - chunk_h + 1, 0)
        if row is None or step < 0 or step >= n_rows:
            payload["G"] = np.full(G.shape[1:], np.nan, dtype=G.dtype)
            payload["valid"] = False
            return payload
        idx = row + step
        if self.shuffle:
            r = int(rank[idx])
            if r < 0:  # invalid transitions stay invalid under shuffling (never borrow)
                payload["G"] = np.full(G.shape[1:], np.nan, dtype=G.dtype)
                payload["valid"] = False
                return payload
            idx = int(perm[r])
        payload["G"] = np.asarray(G[idx], dtype=G.dtype)
        payload["valid"] = bool(np.isfinite(payload["G"]).all())
        return payload
