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
save_video="${SAVE_VIDEO:-false}"
host="127.0.0.1"
# tyro bool (default True) is a flag switch, not a valued option:
if [[ "${save_video}" == "true" ]]; then video_args=(--args.save-video); else video_args=(--args.no-save-video); fi

# Block until the policy server is actually listening, or fail fast when it
# crashed at startup (bad ckpt path, CUDA OOM, port in use). Without this gate,
# all shards would hammer a dead port and burn the whole GPU-time slice.
wait_for_server() {
    local pid="$1" port="$2"
    local timeout="${SERVER_START_TIMEOUT:-600}" waited=0
    while kill -0 "${pid}" 2>/dev/null; do
        if timeout 2 bash -c "echo > '/dev/tcp/127.0.0.1/${port}'" 2>/dev/null; then
            echo "[server] listening on port ${port} (pid ${pid})"
            return 0
        fi
        if (( waited >= timeout )); then
            echo "[ERROR] Server (pid ${pid}) did not open port ${port} within ${timeout}s."
            return 1
        fi
        sleep 2
        waited=$((waited + 2))
    done
    echo "[ERROR] Server process (pid ${pid}) exited before opening port ${port}."
    return 1
}

# Client-side evals run on CPU (Mesa EGL swrast only — no NVIDIA EGL on this
# host): no CUDA_VISIBLE_DEVICES scoping and no MUJOCO_EGL_DEVICE_ID pinning. Cap
# llvmpipe threads so 24 concurrent CPU-rendering clients do not oversubscribe.
export LP_NUM_THREADS="${LP_NUM_THREADS:-4}"
CUDA_VISIBLE_DEVICES="${gpu_id}" ${STARVLA_PYTHON} deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${base_port} \
    --use_bf16 &

# Get the server PID
server_pid=$!

if ! wait_for_server "${server_pid}" "${base_port}"; then
    kill "${server_pid}" 2>/dev/null || true
    exit 1
fi

# Eval artifacts: <ckpt_dir>/<ckpt basename without ext>/eval/<endpoint>/
#   {logs,videos,overall_results.json}
# The endpoint is a path component, not a filename suffix: LIBERO and LIBERO-Plus share
# every suite name and each driver wipes logs/<suite>/*.json at start, so a shared root
# means whichever endpoint runs second deletes the other one's shards.
ckpt_dir=$(dirname "$your_ckpt")
eval_name=$(basename "$your_ckpt")
eval_name="${eval_name%.*}"
ENDPOINT="libero"
eval_root="${ckpt_dir}/${eval_name}/eval/${ENDPOINT}"

video_out_path="${eval_root}/videos/${task_suite_name}"
log_path="${eval_root}/logs/${task_suite_name}"
mkdir -p "$video_out_path"
mkdir -p "$log_path"

# This job owns <eval_root>/logs/<suite>: clear stale shard/aggregate JSONs from
# any earlier attempt of the same checkpoint before launching new parts, so the
# aggregator can never mix old shards with new ones.
rm -f "${log_path}"/*.json
rm -f "${eval_root}/overall_results.json"

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

    p_log="${log_path}/${eval_name}_part${i}.log"
    echo "Part ${i}: tasks [${start_idx}, ${current_end}) -> ${p_log}"

    eval_start=$start_idx
    eval_end=$current_end
    if [ $eval_start -ge $eval_end ]; then
        break
    fi

    # pipefail inside a subshell: tee must not swallow the python exit code,
    # otherwise a crashed shard would silently count as success.
    ( set -o pipefail; \
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
        "${video_args[@]}" \
        2>&1 | tee "$p_log" ) &
    pids+=($!)

    start_idx=$current_end
    if [ $start_idx -ge $total ]; then
        break
    fi
done

fail=0
for part_pid in "${pids[@]}"; do
    wait "$part_pid" || fail=1
done

if (( fail != 0 )); then
    echo "[ERROR] One or more eval shards failed (see part logs in ${log_path}); skipping aggregation."
    if [ -n "$server_pid" ]; then
        kill "$server_pid" 2>/dev/null || true
    fi
    exit 1
fi

# Aggregate only THIS suite (siblings sharing eval_root run concurrently;
# overall_results.json is written once by the auto-eval final pass).
"${STARVLA_PYTHON}" ./examples/simBenchmarks/LIBERO-plus/eval_files/parallel_eval/aggregate_results.py \
    --root_path "${eval_root}" \
    --suites "${task_suite_name}" \
    --expected-total "$((num_tasks * num_trials_per_task))" \
    || fail=1

echo "Evaluation completed. Videos in ${video_out_path}, logs in ${log_path}, per-suite results in ${log_path}/task_results.json"

if [ -n "$server_pid" ]; then
    echo "Killing server process with PID: $server_pid"
    kill "$server_pid" 2>/dev/null || true
else
    echo "No server process found to kill."
fi

exit "$fail"
