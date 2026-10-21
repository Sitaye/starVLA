export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

###########################################################################################
# === Please modify the following paths according to your environment ===
Framework_name=QwenOFT
freeze_module_list=''
base_vlm=/defaultShare/archive/linpenghan/models/Qwen3.5-0.8B/
ARM=${1:-}
LAMBDA=${2:-}
if [ -z "${ARM}" ]; then
  echo "usage: bash run_oeB_train.sh <arm> [lambda]  # arms: baseline taskfixed trace trace_eef diag full shuffled" >&2
  exit 2
fi
case ${ARM} in
  baseline|taskfixed|trace|trace_eef|diag|full|shuffled) ;;
  *) echo "unknown arm ${ARM}; see train_oeB_${ARM}.yaml" >&2; exit 2 ;;
esac
config_yaml=./examples/simBenchmarks/LIBERO/train_files/train_oeB_${ARM}.yaml
# lambda_OE is calibrated per arm by calibrate_oe_lambda.py and already written into the arm
# YAML, which is the single source of truth; the optional second argument only overrides it.
# baseline must never receive framework.oe: an override under a missing key would create the
# section, and the model now refuses to build one without an explicit mode and weight.
oe_args=()
if [ -n "${LAMBDA}" ]; then
  if [ "${ARM}" = baseline ]; then
    echo "baseline takes no lambda (it has no framework.oe section)" >&2
    exit 2
  fi
  oe_args+=(--framework.oe.loss_weight "${LAMBDA}")
fi
libero_data_root=/defaultShare/pubdata/OXE_LEROBOT_DATASET/libero_no_noops_1.0.0_lerobot/
data_mix=libero_all
seed=42
num_workers=8
run_root_dir=/defaultShare/archive/linpenghan/checkpoints/cvpr27/
run_id=$(date +%Y%m%d_%H%M%S)_oeB_${ARM}
# === End of environment variable configuration ===
###########################################################################################


output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
# mv this script to the output dir
cp $0 ${output_dir}/


num_processes=${NUM_PROCESSES:-$(nvidia-smi -L | wc -l)}

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes ${num_processes} \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --datasets.vla_data.data_root_dir ${libero_data_root}\
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 16 \
  --datasets.vla_data.video_backend pyav \
  --datasets.vla_data.num_workers ${num_workers} \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 10000 \
  --trainer.save_interval 5000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 100 \
  --seed ${seed} \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --swanlab_project cvpr27 \
  "${oe_args[@]}"
  # --is_debug True
