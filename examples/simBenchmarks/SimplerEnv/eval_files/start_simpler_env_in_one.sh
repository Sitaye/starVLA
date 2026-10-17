#!/usr/bin/env bash
set -euo pipefail

echo "$(which python)"

########### USER CONFIG ###########
star_vla_python="${star_vla_python:-python}"
sim_python="${sim_python:-python}"
SimplerEnv_PATH="${SimplerEnv_PATH:-}"
SIMPLER_ENV_LIB_DIR="${SIMPLER_ENV_LIB_DIR:-}"
port="${port:-6678}"
gpu_id="${gpu_id:-0}"
seed="${seed:-7}"
USE_BF16="${USE_BF16:-1}"
your_ckpt="${your_ckpt:-./results/Checkpoints/0418_oxe_bridge_rt_1_QwenGR00T/checkpoints/steps_10000_pytorch_model.pt}"
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-600}"
SERVER_READY_POLL="${SERVER_READY_POLL:-5}"
########### USER CONFIG ###########

MODEL_PATH=${1:-"${your_ckpt}"}
export your_ckpt="${MODEL_PATH}"
export star_vla_python sim_python SimplerEnv_PATH SIMPLER_ENV_LIB_DIR port seed USE_BF16
export SERVER_READY_TIMEOUT SERVER_READY_POLL

# evaluator tree also needs a GPU scope (its diagnostics assume CUDA_VISIBLE_DEVICES set);
# evaluator-side MuJoCo/SAPIEN rendering then lands on the same card as the server
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${gpu_id}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_PID=""

cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "⏹ Stopping policy server pipeline (pid ${SERVER_PID})"
    pkill -P "${SERVER_PID}" 2>/dev/null || true   # python server + tee children
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

# ── 0) pre-flight: the port must be free. The readiness probe below only checks
#    "something listens on ${port}" — a stale server (e.g. another checkpoint)
#    would silently be evaluated against instead of ours
if (exec 3<>/dev/tcp/127.0.0.1/${port}) 2>/dev/null; then
  echo "❌ port ${port} is already occupied — free it (or choose another port) and restart"
  exit 1
fi

# ── 1) policy server in background (reuses run_policy_server.sh semantics:
#    bf16 flag + per-command CUDA_VISIBLE_DEVICES gpu_id + tee'd server log)
echo "▶️ Starting policy server on GPU ${gpu_id}, port ${port}"
bash "${SCRIPT_DIR}/run_policy_server.sh" &
SERVER_PID=$!

# ── 2) readiness probe: port accepts a connection (no blind sleep)
t0=${SECONDS}
while true; do
  # kill -0 stays true for a zombie; ps stat exposes it, so reap-and-fail fast
  server_stat="$(ps -o stat= -p "${SERVER_PID}" 2>/dev/null || true)"
  if ! kill -0 "${SERVER_PID}" 2>/dev/null || [[ "${server_stat:0:1}" == "Z" ]]; then
    wait "${SERVER_PID}" 2>/dev/null || true
    echo "❌ policy server exited before ready; log → <ckpt_dir>/output_server/<ckpt_name>_policy_server_${port}.log"
    exit 1
  fi
  if (exec 3<>/dev/tcp/127.0.0.1/${port}) 2>/dev/null; then
    echo "🎛 policy server ready after $((SECONDS - t0))s"
    break
  fi
  if (( SECONDS - t0 >= SERVER_READY_TIMEOUT )); then
    echo "❌ timed out after ${SERVER_READY_TIMEOUT}s waiting for port ${port}"
    exit 1
  fi
  sleep "${SERVER_READY_POLL}"
done

# ── 3) parallel evaluator run (4 widowx tasks co-run + aggregate inside)
bash "${SCRIPT_DIR}/start_simpler_env.sh" "${MODEL_PATH}"

echo "✅ all-in-one finished: summary → <ckpt_dir>/output_eval/<ckpt_name>_summary.json"
