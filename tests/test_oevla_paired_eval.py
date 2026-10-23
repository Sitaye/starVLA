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
"""Tests for the gate's paired decision statistic (numpy only; no torch, no GPU)."""

import json

import pytest

from starVLA.analysis.oevla import paired_eval as PE


def _write(root, suite, name, payload):
    suite_dir = root / "logs" / suite
    suite_dir.mkdir(parents=True, exist_ok=True)
    (suite_dir / name).write_text(json.dumps(payload))


def _arm(rows, suite="libero_goal"):
    """rows: [(key, episodes, successes)] on one suite."""
    return {(suite, key): (n, s) for key, n, s in rows}


# --------------------------------------------------------------------------- loading


def test_load_arm_sums_disjoint_shards_and_ignores_the_per_suite_file(tmp_path):
    _write(tmp_path, "libero_goal", "0_4.json", {"0": {"total_count": 5, "success_count": 4}})
    _write(tmp_path, "libero_goal", "4_7.json", {"1": {"total_count": 3, "success_count": 1}})
    _write(tmp_path, "libero_goal", PE.PER_SUITE_FILE, {"0": {"total_count": 999, "success_count": 999}})
    strata = PE.load_arm(tmp_path)
    assert strata == {
        ("libero_goal", "0"): (5, 4),
        ("libero_goal", "1"): (3, 1),
    }


def test_load_arm_sums_a_key_written_by_two_overlapping_shards(tmp_path):
    _write(tmp_path, "libero_goal", "0_4.json", {"0": {"total_count": 5, "success_count": 4}})
    _write(tmp_path, "libero_goal", "2_4.json", {"0": {"total_count": 5, "success_count": 5}})
    assert PE.load_arm(tmp_path)[("libero_goal", "0")] == (10, 9)


def test_load_arm_rejects_a_missing_suite(tmp_path):
    _write(tmp_path, "libero_goal", "0_4.json", {"0": {"total_count": 5, "success_count": 4}})
    with pytest.raises(SystemExit, match="missing suite"):
        PE.load_arm(tmp_path, ["libero_goal", "libero_10"])


def test_load_arm_rejects_a_malformed_entry(tmp_path):
    _write(tmp_path, "libero_goal", "0_4.json", {"0": {"total_count": 5}})
    with pytest.raises(SystemExit, match="malformed entry"):
        PE.load_arm(tmp_path)


def test_load_arm_skips_a_flat_per_suite_summary(tmp_path, capsys):
    """`overall_sr.json` carries the same count names at the top level; summing it double-counts."""
    _write(tmp_path, "libero_10", "0_4.json", {"0": {"total_count": 50, "success_count": 44}})
    _write(tmp_path, "libero_10", "overall_sr.json", {"total_count": 50, "success_count": 46, "sr": 0.92})
    assert PE.load_arm(tmp_path) == {("libero_10", "0"): (50, 44)}
    assert "overall_sr.json" in capsys.readouterr().err


def test_load_arm_rejects_more_successes_than_episodes(tmp_path):
    _write(tmp_path, "libero_goal", "0_4.json", {"0": {"total_count": 5, "success_count": 6}})
    with pytest.raises(SystemExit, match="outside"):
        PE.load_arm(tmp_path)


# --------------------------------------------------------------------------- alignment


def test_align_rejects_a_key_only_one_arm_scored():
    a = _arm([("0", 5, 4)])
    b = _arm([("1", 5, 4)])
    with pytest.raises(SystemExit, match="do not share"):
        PE.align(a, b)


def test_align_rejects_unequal_episode_counts():
    a = _arm([("0", 5, 4)])
    b = _arm([("0", 4, 4)])
    with pytest.raises(SystemExit, match="different episode counts"):
        PE.align(a, b)


# --------------------------------------------------------------------------- statistics


def test_point_estimate_weights_trials_not_keys():
    # 10 episodes vs 990: the pooled difference must not be the mean of 80.0pp and 0.0pp
    a = _arm([("big", 990, 495), ("small", 10, 10)])
    b = _arm([("big", 990, 495), ("small", 10, 2)])
    record = PE.compare(a, b, "a", "b", draws=200, seed=42)
    assert record["episodes"] == 1000
    assert record["delta_pp"] == pytest.approx(0.8, abs=1e-9)
    assert record["per_suite"]["libero_goal"]["delta_pp"] == pytest.approx(0.8, abs=1e-9)


def test_identical_arms_give_zero_delta_and_a_degenerate_interval():
    a = _arm([(str(k), 100, 90) for k in range(6)])
    record = PE.compare(a, a, "a", "a", draws=200, seed=42)
    assert record["delta_pp"] == 0.0
    assert record["ci_pp"] == [0.0, 0.0]
    assert record["p_permutation"] == 1.0


def test_permutation_and_bootstrap_flag_a_large_gap():
    keys = [(suite, str(k), 200) for suite in ("libero_goal", "libero_10") for k in range(8)]
    a = {("libero_goal", k): (n, 180) for _, k, n in keys if _ == "libero_goal"}
    a.update({("libero_10", k): (n, 180) for _, k, n in keys if _ == "libero_10"})
    b = {key: (n, 140) for key, (n, _) in a.items()}
    record = PE.compare(a, b, "a", "b", draws=2000, seed=42)
    assert record["delta_pp"] == pytest.approx(20.0)
    assert record["ci_pp"][0] > 15.0
    assert record["p_permutation"] < 1e-3


def test_statistics_are_reproducible_for_a_fixed_seed():
    a = _arm([(str(k), 50, 40 + k % 3) for k in range(10)])
    b = _arm([(str(k), 50, 35 + k % 4) for k in range(10)])
    first = PE.compare(a, b, "a", "b", draws=500, seed=7)
    second = PE.compare(a, b, "a", "b", draws=500, seed=7)
    assert first["ci_pp"] == second["ci_pp"]
    assert first["p_permutation"] == second["p_permutation"]


# --------------------------------------------------------------------------- gate


def test_judge_requires_the_margin_and_a_positive_ci_low():
    passing = {"delta_pp": 4.0, "ci_pp": [1.5, 6.5]}
    too_small = {"delta_pp": 2.9, "ci_pp": [0.5, 5.3]}
    straddling = {"delta_pp": 3.5, "ci_pp": [-0.2, 7.2]}
    assert PE.judge(passing, 3.0)["pass"]
    assert not PE.judge(too_small, 3.0)["pass"]
    assert not PE.judge(straddling, 3.0)["pass"]


def test_main_gates_the_primary_against_any_reference(tmp_path, monkeypatch, capsys):
    """The gate must work when the primary is not the first `--run`; that lookup used to raise."""
    roots = {}
    for name, successes in (("base", 40), ("full", 48), ("taskfixed", 44)):
        root = tmp_path / name
        shard = {str(k): {"total_count": 50, "success_count": successes} for k in range(8)}
        _write(root, "libero_goal", "0_4.json", shard)
        roots[name] = root
    out_dir = tmp_path / "out"
    argv = ["paired_eval", "--draws", "500", "--min-delta", "3.0", "--out-dir", str(out_dir)]
    for name, root in roots.items():
        argv += ["--run", f"{name}={root}"]
    argv += ["--primary", "full", "--reference", "taskfixed"]
    monkeypatch.setattr("sys.argv", argv)
    PE.main()
    payload = json.loads((out_dir / "paired_eval.json").read_text())
    assert payload["gate"]["verdict"] == "GO"
    assert payload["gate"]["references"]["taskfixed"]["pass"]
    assert payload["pairs"]["full|taskfixed"]["delta_pp"] == pytest.approx(8.0)
    assert "GO" in capsys.readouterr().out


def test_load_arm_can_select_one_endpoint(tmp_path):
    """LIBERO and LIBERO-Plus share logs/<suite>/ but disagree on the key universe."""
    _write(tmp_path, "libero_spatial", "0_4.json", {"0": {"total_count": 50, "success_count": 48}})
    _write(tmp_path, "libero_spatial", "0_to_100.json", {"camera": {"total_count": 90, "success_count": 70}})
    mixed = PE.load_arm(tmp_path)
    assert sorted(mixed) == [("libero_spatial", "0"), ("libero_spatial", "camera")]
    assert sum(n for n, _ in mixed.values()) == 140
    only_libero = PE.load_arm(tmp_path, skip_globs=("*_to_*",))
    assert only_libero == {("libero_spatial", "0"): (50, 48)}
    only_plus = PE.load_arm(tmp_path, shard_glob="*_to_*.json")
    assert only_plus == {("libero_spatial", "camera"): (90, 70)}


def test_load_arm_hard_fails_on_an_implausible_per_suite_total(tmp_path):
    _write(tmp_path, "libero_spatial", "0_4.json", {"0": {"total_count": 50, "success_count": 48}})
    _write(tmp_path, "libero_spatial", "0_to_100.json", {"camera": {"total_count": 50, "success_count": 40}})
    with pytest.raises(SystemExit, match="differ from the expected 500"):
        PE.load_arm(tmp_path, skip_globs=(), expected_per_suite=500)
    assert PE.load_arm(tmp_path, skip_globs=("*_to_*",), expected_per_suite=50)
