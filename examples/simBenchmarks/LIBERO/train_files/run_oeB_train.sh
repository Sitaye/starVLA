export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

###########################################################################################
# === Please modify the following paths according to your environment ===
Framework_name=QwenOFT
freeze_module_list=''
base_vlm=/defaultShare/archive/linpenghan/models/Qwen3.5-0.8B/
ARM=${1:?usage: bash run_oeB_train.sh <baseline|taskfixed|trace|diag|full|shuffled>}
config_yaml=./examples/simBenchmarks/LIBERO/train_files/train_oeB_${ARM}.yaml
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
  # --is_debug True
