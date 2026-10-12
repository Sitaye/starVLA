#!/usr/bin/env bash
set -euo pipefail

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../../../.." && pwd)}"
LIBERO_PLUS_HOME="${LIBERO_PLUS_HOME:-}"
LIBERO_PLUS_PYTHON="${LIBERO_PLUS_PYTHON:-python}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
tasks_per_gpu="${tasks_per_gpu:-3}"
your_ckpt="${your_ckpt:-/path/to/checkpoint.pt}"
output_dir="${output_dir:-${STARVLA_DIR}/results/libero_plus_parallel_eval}"

if [[ -z "${LIBERO_PLUS_HOME}" ]]; then
  echo "LIBERO_PLUS_HOME is required."
  exit 1
fi

cd "${STARVLA_DIR}"
# user-facing vars above; package-facing aliases for the libero codebase below
export LIBERO_PLUS_HOME
export LIBERO_HOME="${LIBERO_PLUS_HOME}"
export LIBERO_CONFIG_PATH="${LIBERO_PLUS_HOME}/libero"
export PYTHONPATH="${PYTHONPATH:-}:${LIBERO_PLUS_HOME}:${STARVLA_DIR}"

task_suite_name=$1
start_idx=$2
end_idx=$3
num_trials_per_task=1
# torchrun --nproc_per_node=1 ./examples/simBenchmarks/LIBERO-plus/eval_files/eval_nebula/eval_libero_model.py \

total=$((end_idx - start_idx))
chunk_size=$((total / tasks_per_gpu))
remainder=$((total % tasks_per_gpu))
current_start=$start_idx

for ((i=0; i<tasks_per_gpu; i++)); do

    if [ $i -lt $remainder ]; then
        current_end=$((current_start + chunk_size + 1))
    else
        current_end=$((current_start + chunk_size))
    fi


    if [ $current_end -gt $end_idx ]; then
        current_end=$end_idx
    fi

    if [ $current_start -ge $current_end ]; then
        break
    fi

    echo "Part $((i)): start=$current_start, end=$current_end ([$current_start, $current_end))"
    # torchrun --nproc_per_node=$gpu_per_pod --nnodes=$WORLD_SIZE --node_rank=$RANK --master_addr=$((MASTER_ADDR+i)) --master_port=$MASTER_PORT 
    "${LIBERO_PLUS_PYTHON}" ./examples/simBenchmarks/LIBERO-plus/eval_files/parallel_eval/eval_libero_model.py \
    --pretrained_path $your_ckpt \
    --task_suite_name $task_suite_name \
    --num_trials_per_task $num_trials_per_task \
    --output_dir $output_dir \
    --start_idx $current_start \
    --end_idx $current_end &

    current_start=$current_end

    if [ $current_start -ge $end_idx ]; then
        break
    fi
done


fail=0
for part_pid in $(jobs -p); do
    wait "$part_pid" || fail=1
done
exit $fail

# # =============== Aggregate results ===============
# echo "All tasks completed. Aggregating results..."
# export LOG_DIR="${LOG_DIR}"
# python ./examples/simBenchmarks/LIBERO-plus/eval_files/aggregate_results.py
