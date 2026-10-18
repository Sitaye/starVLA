# Copyright 2026 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""RAP-FSD Stage A component 1: deterministic episode-level data split.

Implements the pre-registered Stage A protocol v1.2 (docs/research/ac_audit_rapfsd_math_20261018.md §8):

* episode-level K=2 fold split + an independent holdout that participates in NO fold;
* assignment is a pure deterministic hash of (seed, episode_id) — repeated runs on the
  same inputs are byte-identical, and the label of an episode never depends on which
  other episodes exist (stable under partial cache availability);
* holdout sizes 2000/2500/3000 are all recorded in one manifest (nested subsets:
  holdout_2000 ⊂ holdout_2500 ⊂ holdout_3000); the primary protocol size is 2500.

Episode enumeration sources (either, or both combined):
* RAFT flow cache directories: ``<cache_root>/<dataset_name>/keys.npy`` with
  ``key = trajectory_id * 100_000 + step`` (examples/simBenchmarks/LIBERO/train_files/cache_raft_flow.py:9-10),
  episode = unique ``key // 100_000`` per dataset;
* LeRobot dataset roots (v2.0 ``meta/episodes.jsonl`` / v3.0 ``meta/episodes/*.parquet``),
  which additionally provide the per-episode task string used by the per-suite/per-task
  breakdown in the Stage A report.

Episode id convention: ``<dataset_name>/<trajectory_id:05d>`` — globally unique across
the libero_all mixture (4 suites x 2500 episodes = 10,000 episodes).

Usage:
    python -m starVLA.analysis.rap_fsd.split --cache-root /path/to/flow_cache --output manifest.json
    python -m starVLA.analysis.rap_fsd.split --cache-root ... --data-root ... --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from itertools import pairwise
from pathlib import Path

import numpy as np

# protocol §8 (v1.2): K=2 fold + holdout 2,500 primary (sensitivity 2k/3k), seed=42 frozen
DEFAULT_SEED = 42
DEFAULT_HOLDOUT_SIZES = (2000, 2500, 3000)
PRIMARY_HOLDOUT_SIZE = 2500
FOLD_LABELS = ("fold_0", "fold_1")
HOLDOUT_LABEL = "holdout"

# dataset_name -> suite, e.g. "libero_object_no_noops_1.0.0_lerobot" -> "libero_object"
LIBERO_SUITE_PREFIXES = ("libero_object", "libero_goal", "libero_spatial", "libero_10")


def suite_from_dataset_name(dataset_name: str) -> str:
    for prefix in LIBERO_SUITE_PREFIXES:
        if dataset_name.startswith(prefix):
            return prefix
    raise ValueError(f"cannot infer suite from dataset name {dataset_name!r}")


def episode_key(trajectory_id: int, step: int) -> int:
    """Transition key convention of the RAFT cache (cache_raft_flow.py:9-10)."""
    return int(trajectory_id) * 100_000 + int(step)


def episode_id_of(dataset_name: str, trajectory_id: int) -> str:
    return f"{dataset_name}/{int(trajectory_id):05d}"


def _stable_hash(seed: int, episode_id: str) -> int:
    """Unsigned 64-bit hash of (seed, episode_id); pure function of the inputs."""
    payload = f"rapfsd/stage_a/v1/seed={seed}/{episode_id}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def enumerate_episodes_from_cache(cache_root: str | Path) -> list[dict]:
    """Enumerate episodes from RAFT cache keys.npy files (mmap, never loads flow.npy)."""
    episodes: list[dict] = []
    for dataset_dir in sorted(p for p in Path(cache_root).iterdir() if (p / "keys.npy").exists()):
        keys = np.load(dataset_dir / "keys.npy")
        trajectory_ids = np.unique(keys // 100_000)
        dataset_name = dataset_dir.name
        for trajectory_id in trajectory_ids:
            episodes.append(
                {
                    "episode_id": episode_id_of(dataset_name, int(trajectory_id)),
                    "dataset_name": dataset_name,
                    "trajectory_id": int(trajectory_id),
                    "suite": suite_from_dataset_name(dataset_name),
                    "task": None,
                }
            )
    return episodes


def enumerate_episodes_from_lerobot(data_root: str | Path, data_mix: str) -> list[dict]:
    """Enumerate episodes from LeRobot dataset metadata (v2.0 jsonl / v3.0 parquet).

    Import-light by design: json for v2.0, pandas only for v3.0 parquet (lazy).
    """
    import starVLA.dataloader.gr00t_lerobot.registry as registry

    mixture_spec = registry.DATASET_NAMED_MIXTURES[data_mix]
    episodes: list[dict] = []
    for dataset_name, _weight, _robot_type in mixture_spec:
        dataset_path = Path(data_root) / dataset_name
        suite = suite_from_dataset_name(dataset_name)
        episodes.extend(
            {
                "episode_id": episode_id_of(dataset_name, int(entry["episode_index"])),
                "dataset_name": dataset_name,
                "trajectory_id": int(entry["episode_index"]),
                "suite": suite,
                "task": entry.get("task"),
            }
            for entry in _read_lerobot_episode_meta(dataset_path)
        )
    return episodes


def _read_lerobot_episode_meta(dataset_path: Path) -> list[dict]:
    """Read episode metadata with task strings; v2.0 jsonl first, then v3.0 parquet."""
    jsonl_path = dataset_path / "meta" / "episodes.jsonl"
    if jsonl_path.exists():
        entries = []
        try:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except OSError as exc:
            raise RuntimeError(f"cannot read episode metadata {jsonl_path}") from exc
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"malformed episode metadata line in {jsonl_path}") from exc
            tasks = record.get("tasks") or []
            task = tasks[0] if isinstance(tasks, list) and tasks else None
            entries.append({"episode_index": int(record["episode_index"]), "task": task})
        return entries

    parquet_paths = sorted(dataset_path.glob("meta/episodes/*.parquet"))
    if parquet_paths:
        import pandas as pd

        entries = []
        for parquet_path in parquet_paths:
            frame = pd.read_parquet(parquet_path)
            for _, row in frame.iterrows():
                task = row.get("tasks")
                if isinstance(task, (list, tuple)) and task:
                    task = task[0]
                entries.append({"episode_index": int(row["episode_index"]), "task": task})
        return entries
    raise FileNotFoundError(f"no episode metadata under {dataset_path}/meta (v2.0 jsonl or v3.0 parquet)")


def assign_labels(
    episodes: list[dict],
    seed: int = DEFAULT_SEED,
    holdout_sizes: tuple[int, ...] = DEFAULT_HOLDOUT_SIZES,
) -> dict:
    """Deterministic hash assignment: holdout = lowest hashes, folds alternate by fold rank.

    Pure function of (seed, episode ids): no RNG state, no dependence on input order.
    Holdout subsets for different sizes are nested by construction (same sorted hash
    order truncated at each size).
    """
    holdout_sizes = tuple(sorted(int(h) for h in holdout_sizes))
    primary = max(holdout_sizes) if holdout_sizes else PRIMARY_HOLDOUT_SIZE
    records = sorted(episodes, key=lambda e: e["episode_id"])
    hashed = sorted(
        ((_stable_hash(seed, e["episode_id"]), e["episode_id"]) for e in records),
        key=lambda pair: (pair[0], pair[1]),
    )
    n = len(hashed)
    if primary > n:
        raise ValueError(f"holdout size {primary} exceeds episode count {n}")
    if n - primary < len(FOLD_LABELS):
        raise ValueError("not enough non-holdout episodes to populate both folds")

    labels: dict[str, dict[str, str]] = {}
    for rank, (_hash, episode_id) in enumerate(hashed):
        per_size = {}
        for size in holdout_sizes:
            per_size[str(size)] = HOLDOUT_LABEL if rank < size else FOLD_LABELS[(rank - size) % len(FOLD_LABELS)]
        labels[episode_id] = per_size
    return {"episodes": labels, "holdout_sizes": list(holdout_sizes)}


def build_manifest(
    episodes: list[dict],
    seed: int = DEFAULT_SEED,
    holdout_sizes: tuple[int, ...] = DEFAULT_HOLDOUT_SIZES,
) -> dict:
    """Assemble the manifest JSON structure (sorted everywhere for byte-identical output)."""
    assignment = assign_labels(episodes, seed=seed, holdout_sizes=holdout_sizes)
    episode_records = sorted(episodes, key=lambda e: e["episode_id"])
    episode_ids = [e["episode_id"] for e in episode_records]
    if len(episode_ids) != len(set(episode_ids)):
        raise ValueError("duplicate episode ids in enumeration")
    manifest = {
        "protocol": "rap_fsd/stage_a/v1.2",
        "seed": int(seed),
        "episode_key_convention": "trajectory_id * 100_000 + step",
        "episode_id_convention": "<dataset_name>/<trajectory_id:05d>",
        "fold_count": len(FOLD_LABELS),
        "holdout_sizes": assignment["holdout_sizes"],
        "primary_holdout_size": max(assignment["holdout_sizes"]),
        "episodes": {
            e["episode_id"]: {
                "suite": e["suite"],
                "trajectory_id": int(e["trajectory_id"]),
                "task": e["task"],
                "labels": assignment["episodes"][e["episode_id"]],
            }
            for e in episode_records
        },
    }
    _validate_manifest(manifest)
    return manifest


def _validate_manifest(manifest: dict) -> None:
    """Hard invariants: every episode exactly one label per holdout size; folds and holdout disjoint."""
    sizes = manifest["holdout_sizes"]
    for size in sizes:
        counts: dict[str, int] = {}
        for episode_id, record in manifest["episodes"].items():
            label = record["labels"].get(str(size))
            if label not in (HOLDOUT_LABEL, *FOLD_LABELS):
                raise ValueError(f"episode {episode_id} has invalid label {label!r} at holdout size {size}")
            counts[label] = counts.get(label, 0) + 1
        if counts.get(HOLDOUT_LABEL, 0) != size:
            raise ValueError(f"holdout size {size}: expected {size} episodes, got {counts.get(HOLDOUT_LABEL, 0)}")
        fold_sizes = [counts.get(fold, 0) for fold in FOLD_LABELS]
        if min(fold_sizes) == 0:
            raise ValueError(f"empty fold at holdout size {size}: {counts}")
    # nesting: holdout_2000 ⊂ holdout_2500 ⊂ holdout_3000
    ordered = sorted(sizes)
    for smaller, larger in pairwise(ordered):
        small = {
            episode_id
            for episode_id, record in manifest["episodes"].items()
            if record["labels"][str(smaller)] == HOLDOUT_LABEL
        }
        large = {
            episode_id
            for episode_id, record in manifest["episodes"].items()
            if record["labels"][str(larger)] == HOLDOUT_LABEL
        }
        if not small <= large:
            raise ValueError(f"holdout subsets not nested: {smaller} not subset of {larger}")


def load_manifest(path: str | Path) -> dict:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"cannot read manifest {path}") from exc
    try:
        manifest = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"malformed manifest JSON in {path}") from exc
    _validate_manifest(manifest)
    return manifest


def label_lookup(manifest: dict, holdout_size: int | None = None) -> dict[str, str]:
    """episode_id -> fold/holdout label at the requested holdout size (default primary)."""
    size = int(holdout_size if holdout_size is not None else manifest["primary_holdout_size"])
    key = str(size)
    sample = next(iter(manifest["episodes"].values()), None)
    if sample is not None and key not in sample["labels"]:
        raise ValueError(
            f"holdout size {size} not recorded in manifest "
            f"(tiers: {sorted(manifest.get('holdout_sizes', []))}, "
            f"primary: {manifest.get('primary_holdout_size')})"
        )
    return {episode_id: record["labels"][key] for episode_id, record in manifest["episodes"].items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cache-root", type=str, default=None, help="RAFT flow cache root (per-dataset keys.npy dirs)")
    parser.add_argument("--data-root", type=str, default=None, help="LeRobot data root (adds task strings to manifest)")
    parser.add_argument("--data-mix", type=str, default="libero_all", help="mixture name for --data-root enumeration")
    parser.add_argument("--output", type=str, default="rap_fsd_split_manifest.json")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--holdout-sizes",
        type=str,
        default=",".join(str(s) for s in DEFAULT_HOLDOUT_SIZES),
        help="comma-separated nested holdout sizes; largest is primary (protocol: 2000,2500,3000)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="enumerate + assign + validate, print counts, no file writes"
    )
    args = parser.parse_args(argv)
    if args.cache_root is None and args.data_root is None:
        parser.error("--cache-root and/or --data-root is required (they may be combined)")

    episodes = []
    if args.cache_root is not None:
        episodes.extend(enumerate_episodes_from_cache(args.cache_root))
    if args.data_root is not None:
        by_id = {e["episode_id"]: e for e in episodes}
        for entry in enumerate_episodes_from_lerobot(args.data_root, args.data_mix):
            if entry["episode_id"] in by_id:
                by_id[entry["episode_id"]]["task"] = entry["task"]
            else:
                episodes.append(entry)
    if not episodes:
        raise SystemExit("no episodes enumerated — check --cache-root / --data-root")

    manifest = build_manifest(episodes, seed=args.seed, holdout_sizes=tuple(_parse_sizes(args.holdout_sizes)))
    counts: dict[int, dict[str, int]] = {}
    for size in manifest["holdout_sizes"]:
        per_label: dict[str, int] = {}
        for record in manifest["episodes"].values():
            label = record["labels"][str(size)]
            per_label[label] = per_label.get(label, 0) + 1
        counts[size] = per_label
    print(f"episodes={len(manifest['episodes'])} seed={manifest['seed']} counts={counts}")
    per_suite: dict[str, dict[str, int]] = {}
    primary = str(manifest["primary_holdout_size"])
    for record in manifest["episodes"].values():
        suite_counts = per_suite.setdefault(record["suite"], {})
        suite_counts[record["labels"][primary]] = suite_counts.get(record["labels"][primary], 0) + 1
    print(f"per-suite (primary holdout={primary}): {per_suite}")
    if args.dry_run:
        print("dry-run: manifest validated, not written")
        return 0
    output = Path(args.output)
    payload = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    output.write_text(payload, encoding="utf-8")
    print(f"wrote {output}")
    return 0


def _parse_sizes(text: str) -> list[int]:
    sizes = [int(part) for part in text.split(",") if part.strip()]
    if not sizes:
        raise ValueError("empty --holdout-sizes")
    return sorted(sizes)


if __name__ == "__main__":
    sys.exit(main())
