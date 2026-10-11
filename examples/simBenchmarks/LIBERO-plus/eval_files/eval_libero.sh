#!/usr/bin/env bash
set -euo pipefail

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../../.." && pwd)}"
LIBERO_HOME="${LIBERO_HOME:-}"
LIBERO_Python="${LIBERO_Python:-python}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
host="${host:-127.0.0.1}"
your_ckpt="${your_ckpt:-/path/to/checkpoint.pt}"
output_dir="${output_dir:-${STARVLA_DIR}/results/libero_plus_eval}"
gpu_id="${gpu_id:-0}"

if [[ -z "${LIBERO_HOME}" ]]; then
  echo "LIBERO_HOME is required."
  exit 1
fi

cd "${STARVLA_DIR}"
export LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero"
export PYTHONPATH="${PYTHONPATH:-}:${LIBERO_HOME}:${STARVLA_DIR}"

ckpt_dir=$(dirname "$your_ckpt")
eval_name=$(basename "$your_ckpt")
eval_name="${eval_name%.*}"
eval_root="${ckpt_dir}/${eval_name}"
mkdir -p "${eval_root}"

base_port=9883
num_trials_per_task=1
pids=()

for task_suite_name in libero_goal libero_spatial libero_object libero_10; do
    video_out_path="${eval_root}/videos/${task_suite_name}"
    log_path="${eval_root}/logs/${task_suite_name}"
    log_file="${log_path}/${task_suite_name}.log"
    mkdir -p "${video_out_path}" "${log_path}"

    CUDA_VISIBLE_DEVICES="${gpu_id}" "${LIBERO_Python}" ./examples/simBenchmarks/LIBERO-plus/eval_files/eval_libero.py \
        --args.pretrained-path "${your_ckpt}" \
        --args.host "$host" \
        --args.port ${base_port} \
        --args.task-suite-name "${task_suite_name}" \
        --args.num-trials-per-task ${num_trials_per_task} \
        --args.video-out-path "${video_out_path}" \
        --args.log-path "${log_path}" \
        2>&1 | tee "${log_file}" &
    pids+=("$!")
done

echo "Waiting for all evaluation tasks to finish..."
fail=0
for suite_pid in "${pids[@]}"; do
    wait "$suite_pid" || fail=1
done

echo "All tasks completed. Aggregating results..."
"${LIBERO_Python}" ./examples/simBenchmarks/LIBERO-plus/eval_files/parallel_eval/aggregate_results.py --root_path "${eval_root}" || fail=1
echo "Results at ${eval_root}/overall_results.json"
exit $fail
