"""Aggregate per-part LIBERO / LIBERO-Plus shard JSONs into suite and overall results.

Design contract (see docs/research/p0_eval_fixes_impl_20261019.md):

- Each eval part writes ``logs/<suite>/<start>_<end>.json`` (LIBERO, keyed by
  task id) or ``logs/<suite>/<start>_to_<end>.json`` (LIBERO-Plus, keyed by
  disturbance category). Parts are aggregated by key.
- Per-suite aggregation is invoked by ``eval_libero_in_one.sh`` **after all of
  that suite's parts have exited 0**, restricted to ``--suites <suite>`` so
  concurrent sibling suites sharing the same ``--root_path`` are never read
  mid-write.
- ``overall_results.json`` is written only with ``--write-overall``: the
  auto-eval driver passes it once, after every suite job of a checkpoint has
  finished, so a partial or stale overall file can never survive a crash.
- ``--expected-total`` (comma list aligned with ``--suites``) hard-fails when
  a suite's aggregated episode count does not match the planned launch size:
  a silently missing shard must abort loudly instead of producing a
  plausible-looking success rate.
- All writes are atomic (tmp file + ``os.replace``) and all JSON errors are
  reported with the offending filename.
"""

import argparse
import json
import os
import sys
import tempfile

DEFAULT_SUITES = "libero_10,libero_goal,libero_object,libero_spatial"
PER_SUITE_FILE = "task_results.json"


def atomic_json_dump(path: str, obj) -> None:
    """Write JSON atomically: tmp file in the same dir, then os.replace."""
    tmp_fd, tmp_path = tempfile.mkstemp(
        prefix=".tmp_" + os.path.basename(path) + "_", dir=os.path.dirname(path)
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(obj, f)
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="aggregate parallel-eval shard results")
    parser.add_argument("--root_path", required=True, help="per-checkpoint eval root (contains logs/)")
    parser.add_argument(
        "--suites",
        default=DEFAULT_SUITES,
        help=f"comma-separated suites to aggregate (default: {DEFAULT_SUITES})",
    )
    parser.add_argument(
        "--expected-total",
        default=None,
        help="comma-separated planned episode count per suite (same order as --suites); mismatch hard-fails",
    )
    parser.add_argument(
        "--write-overall",
        action="store_true",
        help="write overall_results.json over the given suites (auto-eval final pass only)",
    )
    args = parser.parse_args()

    suites = [s.strip() for s in args.suites.split(",") if s.strip()]
    expected = None
    if args.expected_total is not None:
        try:
            expected = [int(x) for x in args.expected_total.split(",")]
        except ValueError as exc:
            print(f"[ERROR] --expected-total must be integers: {exc}")
            return 1
        if len(expected) != len(suites):
            print(f"[ERROR] --expected-total has {len(expected)} entries for {len(suites)} suites")
            return 1

    results = {"overall": {"total_count": 0, "success_count": 0}}
    for suite_idx, suite in enumerate(suites):
        cur_root = os.path.join(args.root_path, "logs", suite)
        if not os.path.isdir(cur_root):
            print(f"[ERROR] Suite dir missing: {cur_root}")
            return 1
        try:
            names = os.listdir(cur_root)
        except OSError as exc:
            print(f"[ERROR] Cannot list {cur_root}: {exc}")
            return 1
        json_files = sorted(
            os.path.join(cur_root, name)
            for name in names
            if name.endswith(".json") and name != PER_SUITE_FILE and not name.startswith(".tmp_")
        )
        if not json_files:
            print(f"[ERROR] No shard JSONs in {cur_root}")
            return 1
        suite_results = {}
        for file in json_files:
            try:
                with open(file, encoding="utf-8") as f:
                    shard = json.load(f)
            except (json.JSONDecodeError, OSError) as exc:
                print(f"[ERROR] Cannot read shard JSON {file}: {exc}")
                return 1
            for item, entry in shard.items():
                try:
                    total_count = entry["total_count"]
                    success_count = entry["success_count"]
                except (TypeError, KeyError) as exc:
                    print(f"[ERROR] Malformed entry in {file} (key={item!r}): {exc}")
                    return 1
                if item in suite_results:
                    suite_results[item]["total_count"] += total_count
                    suite_results[item]["success_count"] += success_count
                else:
                    suite_results[item] = {"total_count": total_count, "success_count": success_count}
        suite_total = sum(e["total_count"] for e in suite_results.values())
        if expected is not None and suite_total != expected[suite_idx]:
            print(
                f"[ERROR] Suite {suite}: aggregated total_count={suite_total} "
                f"but expected={expected[suite_idx]} (missing or failed shard?)"
            )
            return 1
        if suite_total == 0:
            print(f"[ERROR] Suite {suite}: aggregated total_count is 0")
            return 1
        for item, entry in suite_results.items():
            entry["success_rate"] = float(entry["success_count"]) / float(entry["total_count"])
            results["overall"]["total_count"] += entry["total_count"]
            results["overall"]["success_count"] += entry["success_count"]
            if item not in results:
                results[item] = {"total_count": entry["total_count"], "success_count": entry["success_count"]}
            else:
                results[item]["total_count"] += entry["total_count"]
                results[item]["success_count"] += entry["success_count"]
        atomic_json_dump(os.path.join(cur_root, PER_SUITE_FILE), suite_results)
        print(f"[OK] Suite {suite}: {suite_total} episodes -> {os.path.join(cur_root, PER_SUITE_FILE)}")

    for item in results:
        entry = results[item]
        if entry["total_count"] > 0:
            entry["success_rate"] = float(entry["success_count"]) / float(entry["total_count"])
    if args.write_overall:
        atomic_json_dump(os.path.join(args.root_path, "overall_results.json"), results)
        print(f"[OK] Overall ({args.suites}) -> {os.path.join(args.root_path, 'overall_results.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
