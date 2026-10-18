"""Tests for the parallel-eval shard aggregator.

Covers the P0 fixes from the eval-pipeline review:
- per-suite restriction (--suites) so concurrent suites sharing an eval root
  never race each other's shard files;
- --expected-total hard-fails when a shard is missing or empty (the original
  silent-wrong-success-rate failure mode);
- --write-overall gates the overall file (only the final pass may write it);
- non-parsable / malformed shard JSONs are hard errors naming the file;
- writes are atomic (no .tmp residue) and suite task_results.json carries
  success_rate entries.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

AGG = Path(__file__).resolve().parent.parent / (
    "examples/simBenchmarks/LIBERO-plus/eval_files/parallel_eval/aggregate_results.py"
)


def _shard(suite_dir: Path, name: str, entries: dict) -> None:
    with open(suite_dir / name, "w", encoding="utf-8") as f:
        json.dump(entries, f)


def _make_root(base: Path, suites: dict) -> Path:
    root = base / "eval_root"
    for suite, shards in suites.items():
        suite_dir = root / "logs" / suite
        suite_dir.mkdir(parents=True, exist_ok=True)
        for name, entries in shards.items():
            _shard(suite_dir, name, entries)
    return root


def run_agg(root: Path, *extra: str) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(AGG), "--root_path", str(root), *extra]
    return subprocess.run(cmd, capture_output=True, text=True)


LIBERO_SHARD = {"0": {"total_count": 250, "success_count": 240}, "1": {"total_count": 250, "success_count": 245}}
PLUS_SHARD = {"Camera": {"total_count": 30, "success_count": 25}, "Light": {"total_count": 20, "success_count": 18}}


def test_per_suite_aggregation_and_success_rate(tmp_path):
    root = _make_root(
        tmp_path,
        {
            "libero_goal": {"0_5.json": LIBERO_SHARD},
            "libero_spatial": {"0_5.json": LIBERO_SHARD},
        },
    )
    proc = run_agg(root, "--suites", "libero_goal,libero_spatial")
    assert proc.returncode == 0, proc.stderr
    merged = json.loads((root / "logs" / "libero_goal" / "task_results.json").read_text())
    assert merged["0"] == {"total_count": 250, "success_count": 240, "success_rate": pytest.approx(0.96)}
    # --write-overall absent -> no overall file
    assert not (root / "overall_results.json").exists()


def test_missing_shard_fails_expected_total(tmp_path):
    # planned 500 episodes but only a half-size shard (one 250-episode task) present
    root = _make_root(tmp_path, {"libero_goal": {"0_5.json": {"0": {"total_count": 250, "success_count": 240}}}})
    proc = run_agg(root, "--suites", "libero_goal", "--expected-total", "500")
    assert proc.returncode != 0
    assert "expected=500" in proc.stdout


def test_matching_expected_total_passes(tmp_path):
    shards = {"0_5.json": LIBERO_SHARD, "5_10.json": LIBERO_SHARD}
    root = _make_root(tmp_path, {"libero_goal": shards})
    proc = run_agg(root, "--suites", "libero_goal", "--expected-total", "1000")
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_write_overall_and_atomic_no_residue(tmp_path):
    root = _make_root(tmp_path, {"libero_10": {"0_2.json": LIBERO_SHARD}})
    proc = run_agg(root, "--suites", "libero_10", "--expected-total", "500", "--write-overall")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    overall = json.loads((root / "overall_results.json").read_text())
    assert overall["overall"]["total_count"] == 500
    assert overall["overall"]["success_count"] == 485
    hidden = [p for p in (root / "logs" / "libero_10").iterdir() if p.name.startswith(".tmp_")]
    assert not hidden  # os.replace consumed the tmp file


def test_corrupt_json_hard_error_names_file(tmp_path):
    suite_dir = tmp_path / "eval_root" / "logs" / "libero_goal"
    suite_dir.mkdir(parents=True)
    (suite_dir / "0_5.json").write_text('{"0": {"total_count": 250, "success')
    proc = run_agg(tmp_path / "eval_root", "--suites", "libero_goal")
    assert proc.returncode != 0
    assert "0_5.json" in proc.stdout


def test_missing_suite_dir_hard_error(tmp_path):
    proc = run_agg(tmp_path / "eval_root", "--suites", "libero_goal")
    assert proc.returncode != 0
    assert "Suite dir missing" in proc.stdout


def test_empty_suite_hard_error(tmp_path):
    root = _make_root(tmp_path, {"libero_object": {"0_5.json": {}}})
    proc = run_agg(root, "--suites", "libero_object")
    assert proc.returncode != 0
    assert "total_count is 0" in proc.stdout


def test_stale_task_results_and_tmp_ignored(tmp_path):
    suite_dir = tmp_path / "eval_root" / "logs" / "libero_spatial"
    suite_dir.mkdir(parents=True)
    _shard(suite_dir, "0_5.json", LIBERO_SHARD)
    (suite_dir / "task_results.json").write_text('{"999": {"total_count": 1, "success_count": 9}}')
    (suite_dir / ".tmp_task_results.json_xyz").write_text("{}")
    proc = run_agg(tmp_path / "eval_root", "--suites", "libero_spatial", "--write-overall")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    merged = json.loads((suite_dir / "task_results.json").read_text())
    assert "999" not in merged  # stale aggregate was not re-ingested


def test_plus_category_keys_aggregate(tmp_path):
    root = _make_root(
        tmp_path,
        {
            "libero_10": {"0_to_50.json": PLUS_SHARD, "50_to_100.json": PLUS_SHARD},
        },
    )
    proc = run_agg(root, "--suites", "libero_10", "--expected-total", "100", "--write-overall")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    merged = json.loads((root / "logs" / "libero_10" / "task_results.json").read_text())
    assert merged["Camera"]["total_count"] == 60
    assert merged["Camera"]["success_count"] == 50
