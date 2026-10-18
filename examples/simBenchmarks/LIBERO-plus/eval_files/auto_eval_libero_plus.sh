#!/usr/bin/env bash
set -euo pipefail

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../../../.." && pwd)}"
cd "${STARVLA_DIR}"
SCRIPT_PATH="./examples/simBenchmarks/LIBERO-plus/eval_files/eval_libero_in_one.sh"
STARVLA_PYTHON="${STARVLA_PYTHON:-python}"

###############################################################################
# ============ USER CONFIG: modify this section ============
###############################################################################

# --- Checkpoint directory (all .pt files inside will be evaluated) ---
CKPT_DIR="results/Checkpoints/<run_name>/checkpoints"

# --- Or specify an explicit list (overrides CKPT_DIR when non-empty) ---
CKPT_LIST=(
    # "results/Checkpoints/<run_name>/checkpoints/steps_30000_pytorch_model.pt"
)

# --- Task suites to evaluate (n_tasks probed automatically per suite) ---
TASK_SUITES=(libero_10 libero_goal libero_object libero_spatial)

# --- Available GPUs (least-loaded selection, capped by MAX_JOBS_PER_GPU) ---
GPU_LIST=(0 1 2 3 4 5 6 7)

# --- Base port (each job gets base_port + job_index; env-overridable so two
# --- concurrent eval runs on the same node never collide) ---
BASE_PORT="${BASE_PORT:-6450}"

# --- Seconds to wait between launching jobs on the SAME GPU ---
SLEEP_BETWEEN=20

###############################################################################
# ============ END USER CONFIG ============
###############################################################################

# Build checkpoint list
if [ ${#CKPT_LIST[@]} -eq 0 ]; then
    # Auto-discover from CKPT_DIR
    mapfile -t CKPT_LIST < <(ls -1 "${CKPT_DIR}"/*.pt 2>/dev/null | sort -t_ -k2 -n)
    if [ ${#CKPT_LIST[@]} -eq 0 ]; then
        echo "[ERROR] No .pt files found in ${CKPT_DIR}"
        exit 1
    fi
fi

num_gpus=${#GPU_LIST[@]}
job_index=0
pids=()
gpu_job_count=()  # active jobs per GPU (for slot throttling)
gpu_job_total=()  # cumulative jobs dispatched per GPU
MAX_JOBS_PER_GPU="${MAX_JOBS_PER_GPU:-1}"  # per-GPU concurrency cap (VRAM-safe: server bf16 + 3 clients ≈ 6-8GB per job)
declare -A PID_GPU=()  # pid -> gpu_idx
declare -A JOB_LABEL=() # pid -> "ckpt:task"

# Initialize GPU job counts
for ((i=0; i<num_gpus; i++)); do
    gpu_job_count[$i]=0
    gpu_job_total[$i]=0
done

echo "=========================================="
echo " Auto Eval LIBERO-Plus"
echo "=========================================="
echo " Checkpoints : ${CKPT_LIST[*]}"
echo " Task suites : ${TASK_SUITES[*]}"
echo " GPU list    : ${GPU_LIST[*]}"
echo "=========================================="

# Free GPU slots of finished jobs.
reap_jobs() {
    local pid gpu
    for pid in "${!PID_GPU[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            gpu=${PID_GPU[$pid]}
            gpu_job_count[$gpu]=$(( ${gpu_job_count[$gpu]} - 1 ))
            unset "PID_GPU[$pid]"
        fi
    done
}

for ckpt in "${CKPT_LIST[@]}"; do
    for task in "${TASK_SUITES[@]}"; do
        # Least-loaded GPU selection
        gpu_idx=0
        for ((i=1; i<num_gpus; i++)); do
            if (( ${gpu_job_count[$i]} < ${gpu_job_count[$gpu_idx]} )); then
                gpu_idx=$i
            fi
        done
        # Throttle: wait for a free slot on the target GPU
        while (( ${gpu_job_count[$gpu_idx]} >= MAX_JOBS_PER_GPU )); do
            reap_jobs
            sleep 5
        done
        gpu_id=${GPU_LIST[$gpu_idx]}
        port=$((BASE_PORT + job_index))

        ckpt_name=$(basename "$ckpt" .pt)
        echo "[Job ${job_index}] GPU=${gpu_id}  port=${port}  ckpt=${ckpt_name}  task=${task}"

        bash "$SCRIPT_PATH" "$ckpt" "$task" "$gpu_id" "$port" &
        pids+=($!)
        PID_GPU[$!]=$gpu_idx
        JOB_LABEL[$!]="${ckpt_name}:${task}"

        gpu_job_count[$gpu_idx]=$(( ${gpu_job_count[$gpu_idx]} + 1 ))
        gpu_job_total[$gpu_idx]=$(( ${gpu_job_total[$gpu_idx]} + 1 ))
        job_index=$((job_index + 1))

        sleep "$SLEEP_BETWEEN"
    done
done

# Wait for all jobs, tracking failures (no silent swallowing).
fail_count=0
failed_jobs=()
echo "--- All jobs launched (${job_index} total). Waiting for completion... ---"
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        echo "[WARN] Eval job FAILED: ${JOB_LABEL[$pid]:-unknown} (pid=${pid}; see its part logs)."
        failed_jobs+=("${JOB_LABEL[$pid]:-unknown}")
        fail_count=$((fail_count + 1))
    fi
done

# Final pass per checkpoint: rebuild suite aggregates + overall_results.json.
# Per-suite task_results.json (expected-total-checked) was written by each
# in_one job; this pass adds overall over all suites. Checkpoints with any
# failed job are skipped: a partial overall_results.json must never be written.
AGGREGATOR="./examples/simBenchmarks/LIBERO-plus/eval_files/parallel_eval/aggregate_results.py"
for ckpt in "${CKPT_LIST[@]}"; do
    ckpt_dir=$(dirname "$ckpt")
    eval_name=$(basename "$ckpt" .pt)
    if [[ " ${failed_jobs[*]:-} " == *" ${eval_name}:"* ]]; then
        echo "[SKIP] Final aggregation skipped for ${eval_name} (had failed eval jobs)"
        continue
    fi
    eval_root="${ckpt_dir}/${eval_name}"
    suites_csv=""
    for suite in "${TASK_SUITES[@]}"; do
        if [[ -n "$suites_csv" ]]; then suites_csv="${suites_csv},"; fi
        suites_csv="${suites_csv}${suite}"
    done
    if ! "${STARVLA_PYTHON}" "$AGGREGATOR" \
        --root_path "${eval_root}" \
        --suites "${suites_csv}" \
        --write-overall; then
        echo "[WARN] Final aggregation failed for ${eval_root}"
        fail_count=$((fail_count + 1))
    fi
done

echo "=========================================="
echo " All evaluations completed!"
echo " GPU job distribution:"
for ((i=0; i<num_gpus; i++)); do
    echo "   GPU ${GPU_LIST[$i]}: ${gpu_job_total[$i]} jobs"
done
echo "=========================================="
echo " Failed jobs: ${fail_count}"
if (( fail_count > 0 )); then
    printf '   - %s\n' "${failed_jobs[@]}"
fi

exit "$(( fail_count > 0 ? 1 : 0 ))"
