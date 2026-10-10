#!/usr/bin/env bash
set -euo pipefail

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../../../.." && pwd)}"
LIBERO_HOME="${LIBERO_HOME:-}"
LIBERO_PYTHON="${LIBERO_PYTHON:-python}"
STARVLA_PYTHON="${STARVLA_PYTHON:-python}"

if [[ -z "${LIBERO_HOME}" ]]; then
  echo "LIBERO_HOME is required."
  exit 1
fi

cd "${STARVLA_DIR}"
export LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero"
export PYTHONPATH="${PYTHONPATH:-}:${LIBERO_HOME}:${STARVLA_DIR}"

##### === variables for which evaluation to setup ===
your_ckpt=$1        # e.g. results/Checkpoints/.../steps_20000_pytorch_model.pt
task_suite_name=$2  # align with your model | libero_goal
gpu_id=$3           # GPU id to use (e.g. 0, 1, 2, ...)
base_port=$4        # unique port for this eval instance
##### === variables for which evaluation to setup ===

num_trials_per_task=50
num_tasks="${NUM_TASKS:-10}"
tasks_per_gpu="${TASKS_PER_GPU:-3}"
host="127.0.0.1"

CUDA_VISIBLE_DEVICES=$gpu_id ${STARVLA_PYTHON} deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${base_port} \
    --use_bf16 &

# Get the server PID
server_pid=$!

# Put logs/videos/aggregate under the checkpoint's own directory
ckpt_dir=$(dirname "$your_ckpt")
folder_name=$(echo "$your_ckpt" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')

video_out_path="${ckpt_dir}/videos/${task_suite_name}/${folder_name}"
log_path="${ckpt_dir}/logs/${task_suite_name}"
mkdir -p "$video_out_path"
mkdir -p "$log_path"

start_idx=0
end_idx_tasks=$((num_tasks))
total=$((end_idx_tasks))
chunk_size=$((total / tasks_per_gpu))
remainder=$((total % tasks_per_gpu))
pids=()

for ((i=0; i<tasks_per_gpu; i++)); do
    if [ $i -lt $remainder ]; then
        current_end=$((start_idx + chunk_size + 1))
    else
        current_end=$((start_idx + chunk_size))
    fi
    if [ $current_end -gt $total ]; then
        current_end=$total
    fi

    p_log="${log_path}/${folder_name}_part${i}.log"
    echo "Part ${i}: tasks [${start_idx}, ${current_end}) -> ${p_log}"

    eval_start=$start_idx
    eval_end=$current_end
    if [ $eval_start -ge $eval_end ]; then
        break
    fi

    "${LIBERO_PYTHON}" ./examples/simBenchmarks/LIBERO/eval_files/eval_libero.py \
        --args.pretrained-path "${your_ckpt}" \
        --args.host "$host" \
        --args.port "$base_port" \
        --args.task-suite-name "$task_suite_name" \
        --args.num-trials-per-task "$num_trials_per_task" \
        --args.video-out-path "$video_out_path" \
        --args.log-path "$log_path" \
        --args.start-task-idx "$eval_start" \
        --args.end-task-idx "$eval_end" \
        2>&1 | tee "$p_log" &
    pids+=($!)

    start_idx=$current_end
    if [ $start_idx -ge $total ]; then
        break
    fi
done

wait "${pids[@]}"

"${STARVLA_PYTHON}" ./examples/simBenchmarks/LIBERO-plus/eval_files/parallel_eval/aggregate_results.py \
    --root_path "${ckpt_dir}"

echo "Evaluation completed. Videos in ${video_out_path}, logs in ${log_path}, aggregate at ${ckpt_dir}/overall_results.json"

if [ -n "$server_pid" ]; then
    echo "Killing server process with PID: $server_pid"
    kill $server_pid
else
    echo "No server process found to kill."
fi
