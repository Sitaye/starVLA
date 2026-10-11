import argparse
import glob
import json
import os

parser = argparse.ArgumentParser(description="aggregate results")

parser.add_argument("--root_path", help="")
args = parser.parse_args()
log_dir = args.root_path

task_suites = ["libero_10", "libero_goal", "libero_object", "libero_spatial"]
per_suite_file = "task_results.json"
overall_results = {"overall": {"total_count": 0, "success_count": 0}}
for task_suite in task_suites:
    cur_root = os.path.join(log_dir, "logs", task_suite)
    json_files = glob.glob(os.path.join(cur_root, "*.json"), recursive=True)
    # multi-side (per-part) results for each task, merged by task id within this suite
    suite_results = {}
    for file in json_files:
        if os.path.basename(file) == per_suite_file:
            continue
        with open(file) as f:
            results = json.load(f)
        for item in results:
            total_count = results[item]["total_count"]
            success_count = results[item]["success_count"]
            if item in suite_results:
                suite_results[item]["total_count"] += total_count
                suite_results[item]["success_count"] += success_count
            else:
                suite_results[item] = {"total_count": total_count, "success_count": success_count}
            overall_results["overall"]["total_count"] += total_count
            overall_results["overall"]["success_count"] += success_count
            if item not in overall_results:
                overall_results[item] = {"total_count": total_count, "success_count": success_count}
            else:
                overall_results[item]["total_count"] += total_count
                overall_results[item]["success_count"] += success_count
    if suite_results:
        for item in suite_results:
            suite_results[item]["success_rate"] = float(suite_results[item]["success_count"]) / float(
                suite_results[item]["total_count"]
            )
        with open(os.path.join(cur_root, per_suite_file), "w", encoding="utf-8") as f:
            json.dump(suite_results, f)

for category in overall_results:
    overall_results[category]["success_rate"] = float(overall_results[category]["success_count"]) / float(
        overall_results[category]["total_count"]
    )

with open(os.path.join(log_dir, "overall_results.json"), "w", encoding="utf-8") as f:
    json.dump(overall_results, f)
