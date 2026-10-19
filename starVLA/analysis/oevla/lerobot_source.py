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
"""LeRobot v2.1 LIBERO suite reader for gate A.

Demo source replacement for the raw-hdf5 reader: per-suite meta (info.json,
tasks.jsonl, episodes.jsonl) plus per-episode parquet files give the expert
actions and proprio. LeRobot carries no MuJoCo simulator states, so mid-episode
restore is reconstruction-by-replay (branch.replay_to); episode init states
come from the libero benchmark package (branch), matched per episode against
the recorded frame-0 proprio.
"""

import json
import re
from pathlib import Path

import numpy as np

NO_NOOPS_RE = re.compile(r"_no_noops.*$")


def norm_text(s):
    """Instruction normal form: lowercase, alphanumerics only."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


def suite_benchmark_key(dataset_dir):
    """libero_spatial_no_noops_1.0.0_lerobot -> libero_spatial (benchmark key)."""
    return NO_NOOPS_RE.sub("", Path(dataset_dir).name)


def _read_jsonl(path):
    try:
        rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    except json.JSONDecodeError as e:
        raise ValueError(f"{path}: malformed meta jsonl ({e})") from e
    if not rows:
        raise ValueError(f"{path}: empty jsonl")
    return rows


def _episode_parquet(dataset_dir, info, episode_index):
    chunk = episode_index // int(info["chunks_size"])
    path = dataset_dir / info["data_path"].format(episode_chunk=chunk, episode_index=episode_index)
    if not path.exists():
        raise FileNotFoundError(f"{path}: missing episode parquet")
    return path


class LeRobotSuite:
    """One LeRobot LIBERO suite directory (lazily reads episode parquets)."""

    def __init__(self, dataset_dir):
        self.dataset_dir = Path(dataset_dir)
        meta = self.dataset_dir / "meta"
        try:
            self.info = json.loads((meta / "info.json").read_text())
        except (OSError, json.JSONDecodeError) as e:
            raise ValueError(f"{self.dataset_dir}: unreadable meta/info.json ({e})") from e
        if self.info.get("codebase_version") != "v2.1":
            raise ValueError(f"{self.dataset_dir}: expected LeRobot v2.1, got {self.info.get('codebase_version')}")
        self.tasks = {row["task_index"]: row["task"] for row in _read_jsonl(meta / "tasks.jsonl")}
        self.episodes = sorted(_read_jsonl(meta / "episodes.jsonl"), key=lambda r: int(r["episode_index"]))
        self.benchmark_key = suite_benchmark_key(self.dataset_dir)
        self._actions = {}
        self._proprio = {}

    def episode_rows(self):
        return [
            {"episode_index": int(row["episode_index"]), "task": self.tasks[int(row["task_index"])],
             "task_index": int(row["task_index"]), "length": int(row["length"])}
            for row in self.episodes
        ]

    def actions(self, episode_index):
        if episode_index not in self._actions:
            import pandas as pd

            df = pd.read_parquet(_episode_parquet(self.dataset_dir, self.info, episode_index))
            actions = np.stack(df["action"].to_numpy()).astype(np.float64)
            state = np.stack(df["observation.state"].to_numpy()).astype(np.float64)
            if len(actions) != self.episode_length(episode_index):
                raise ValueError(f"ep{episode_index}: parquet rows != episodes.jsonl length")
            self._actions[episode_index] = actions
            self._proprio[episode_index] = state
        return self._actions[episode_index]

    def proprio(self, episode_index):
        """(T, 8) recorded eef proprio [x, y, z, aa1, aa2, aa3, gripper, gripper]."""
        self.actions(episode_index)  # fills both caches
        return self._proprio[episode_index]

    def episode_length(self, episode_index):
        for row in self.episodes:
            if int(row["episode_index"]) == episode_index:
                return int(row["length"])
        raise KeyError(f"{self.dataset_dir}: no episode {episode_index}")

    def episode_task(self, episode_index):
        for row in self.episodes:
            if int(row["episode_index"]) == episode_index:
                task_index = int(row["task_index"]) if "task_index" in row else None
                return self.tasks[task_index] if task_index is not None else row["tasks"][0]
        raise KeyError(f"{self.dataset_dir}: no episode {episode_index}")

    def frame_rgb(self, episode_index, frame=0):
        """Dataset image frame for init-state matching (channel order as recorded)."""
        import av

        info = self.info
        video_key = next(k for k in info["features"] if k.startswith("observation.images.") and "wrist" not in k)
        chunk = episode_index // int(info["chunks_size"])
        path = self.dataset_dir / info["video_path"].format(
            episode_chunk=chunk, episode_index=episode_index, video_key=video_key
        )
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            for i, frame_ in enumerate(container.decode(stream)):
                if i == frame:
                    return frame_.to_ndarray(format="rgb24")
        raise ValueError(f"{path}: no frame {frame}")
