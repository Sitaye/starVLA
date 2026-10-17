#!/usr/bin/env bash
set -euo pipefail

echo "$(which python)"

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../../.." && pwd)}"
sim_python="${sim_python:-python}"
SimplerEnv_PATH="${SimplerEnv_PATH:-}"
SIMPLER_ENV_LIB_DIR="${SIMPLER_ENV_LIB_DIR:-}"
port="${port:-6678}"
gpu_id="${gpu_id:-0}"
seed="${seed:-0}"
your_ckpt="${your_ckpt:-./results/Checkpoints/0418_oxe_bridge_rt_1_QwenGR00T/checkpoints/steps_10000_pytorch_model.pt}"

MODEL_PATH=${1:-"${your_ckpt}"}
port=${2:-"${port}"}

cd "${STARVLA_DIR}"
export PYTHONPATH="${STARVLA_DIR}:${PYTHONPATH:-}"
if [[ -n "${SIMPLER_ENV_LIB_DIR}" ]]; then
  export LD_LIBRARY_PATH="${SIMPLER_ENV_LIB_DIR}:${LD_LIBRARY_PATH:-}"
fi

#### build output directory #####
ckpt_path=${MODEL_PATH}
ckpt_dir=$(dirname "${ckpt_path}")
ckpt_base=$(basename "${ckpt_path}")
ckpt_name="${ckpt_base%.*}"

# Create output directories
output_server_dir="${ckpt_dir}/output_server"
output_eval_dir="${ckpt_dir}/output_eval"
mkdir -p "${output_server_dir}"
mkdir -p "${output_eval_dir}"
# keep the aggregation merge exact for reruns: purge this checkpoint's stale result jsons
rm -f "${output_eval_dir}/${ckpt_name}_"*_run*.json "${output_eval_dir}/${ckpt_name}_summary.json"
#### build output directory #####

TSET_NUM=1
# export DEBUG=1

IFS=',' read -r -a CUDA_DEVICES <<< "${CUDA_VISIBLE_DEVICES:-0}"
NUM_GPUS=${#CUDA_DEVICES[@]} 

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "CUDA_DEVICES: ${CUDA_DEVICES[*]}"
echo "NUM_GPUS: $NUM_GPUS"

scene_name=bridge_table_1_v1
robot=widowx
rgb_overlay_path=${SimplerEnv_PATH}/ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png
robot_init_x=0.147
robot_init_y=0.028

declare -a ENV_NAMES=(
  StackGreenCubeOnYellowCubeBakedTexInScene-v0
  PutCarrotOnPlateInScene-v0
  PutSpoonOnTableClothInScene-v0
)

declare -a task_pids=()
for i in "${!ENV_NAMES[@]}"; do
  env="${ENV_NAMES[i]}"
  for ((run_idx=1; run_idx<=TSET_NUM; run_idx++)); do
  # Path for log file
    task_log="${output_eval_dir}/${ckpt_name}_${env}_run${run_idx}.log"
    echo "▶️ Launching task [${env}] run#${run_idx}, log → ${task_log}"

    ${sim_python} examples/simBenchmarks/SimplerEnv/eval_files/start_simpler_env.py \
      --ckpt-path ${ckpt_path} \
      --port ${port} \
      --seed ${seed} \
      --results-file "${output_eval_dir}/${ckpt_name}_${env}_run${run_idx}.json" \
      --robot ${robot} \
      --policy-setup widowx_bridge \
      --control-freq 5 \
      --sim-freq 500 \
      --max-episode-steps 120 \
      --env-name "${env}" \
      --scene-name ${scene_name} \
      --rgb-overlay-path ${rgb_overlay_path} \
      --robot-init-x ${robot_init_x} ${robot_init_x} 1 \
      --robot-init-y ${robot_init_y} ${robot_init_y} 1 \
      --obj-variation-mode episode \
      --obj-episode-range 0 24 \
      --robot-init-rot-quat-center 0 0 0 1 \
      --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
      > "${task_log}" 2>&1 &
    task_pids+=( "$!" )

    sleep 6

  done
done

declare -a ENV_NAMES_V2=(
  PutEggplantInBasketScene-v0
)

scene_name=bridge_table_1_v2
robot=widowx_sink_camera_setup
rgb_overlay_path=${SimplerEnv_PATH}/ManiSkill2_real2sim/data/real_inpainting/bridge_sink.png
robot_init_x=0.127
robot_init_y=0.06

for i in "${!ENV_NAMES_V2[@]}"; do
  env="${ENV_NAMES_V2[i]}"
  for ((run_idx=1; run_idx<=TSET_NUM; run_idx++)); do
  # Path for log file
    task_log="${output_eval_dir}/${ckpt_name}_${env}_run${run_idx}.log"
    echo "▶️ Launching V2 task [${env}] run#${run_idx}, log → ${task_log}"

    ${sim_python} examples/simBenchmarks/SimplerEnv/eval_files/start_simpler_env.py\
      --ckpt-path ${ckpt_path} \
      --port ${port} \
      --seed ${seed} \
      --results-file "${output_eval_dir}/${ckpt_name}_${env}_run${run_idx}.json" \
      --robot ${robot} \
      --policy-setup widowx_bridge \
      --control-freq 5 \
      --sim-freq 500 \
      --max-episode-steps 120 \
      --env-name "${env}" \
      --scene-name ${scene_name} \
      --rgb-overlay-path ${rgb_overlay_path} \
      --robot-init-x ${robot_init_x} ${robot_init_x} 1 \
      --robot-init-y ${robot_init_y} ${robot_init_y} 1 \
      --obj-variation-mode episode \
      --obj-episode-range 0 24 \
      --robot-init-rot-quat-center 0 0 0 1 \
      --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
      > "${task_log}" 2>&1 &
    task_pids+=( "$!" )

    sleep 6
  done
done

# wait each evaluator individually: no-operand `wait` always returns 0 and would
# hide a crashed task behind a silently partial summary
echo "⏳ Waiting for ${#task_pids[@]} evaluator processes..."
eval_failed=0
for pid in "${task_pids[@]}"; do
  if ! wait "${pid}"; then
    echo "❌ Evaluator pid ${pid} failed"
    eval_failed=1
  fi
done
if (( eval_failed != 0 )); then
  echo "❌ Aborting aggregation: one or more evaluators failed (see task logs)"
  exit 1
fi

# aggregate per-task JSON summaries into one summary file (report layer only)
${sim_python} - "${output_eval_dir}" "${ckpt_name}" <<'PY'
import glob, json, os, sys

eval_dir, ckpt_name = sys.argv[1], sys.argv[2]
summary = {"ckpt_name": ckpt_name, "tasks": {}, "num_successes": 0, "num_episodes": 0}
files = sorted(glob.glob(os.path.join(eval_dir, f"{ckpt_name}_*_run*.json")))
if not files:
    raise SystemExit(f"no eval result json found in {eval_dir} for ckpt {ckpt_name}")
for path in files:
    result = json.load(open(path))
    n_ok, n_tot = result["num_successes"], result["num_episodes"]
    task = result["environment"]
    summary["tasks"][task] = {"num_successes": summary["tasks"].get(task, {}).get("num_successes", 0) + n_ok,
                              "num_episodes": summary["tasks"].get(task, {}).get("num_episodes", 0) + n_tot}
    summary["num_successes"] += n_ok
    summary["num_episodes"] += n_tot
    print(f"{task}: {summary['tasks'][task]['num_successes']}/{summary['tasks'][task]['num_episodes']}")
summary["success_rate"] = summary["num_successes"] / summary["num_episodes"]
print(f"OVERALL: {summary['num_successes']}/{summary['num_episodes']} = {summary['success_rate']:.4f}")
with open(os.path.join(eval_dir, f"{ckpt_name}_summary.json"), "w") as f:
    json.dump(summary, f, indent=2, sort_keys=True)
PY

echo "✅ Finished: summary → ${output_eval_dir}/${ckpt_name}_summary.json"
