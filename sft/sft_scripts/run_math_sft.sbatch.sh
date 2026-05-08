#!/usr/bin/env bash
# Paper SFT entrypoint for vanilla SFT, RPSFT, IW, DFT, and LoRA.
#
# Examples:
#   sbatch --export=ALL,MODEL_TYPE=qwen,METHOD=sft sft/sft_scripts/run_math_sft.sbatch.sh
#   sbatch --export=ALL,MODEL_TYPE=qwen,METHOD=rpsft,SVD_REG_TOPK=768 sft/sft_scripts/run_math_sft.sbatch.sh
#   sbatch --export=ALL,MODEL_TYPE=qwen-3B,METHOD=iw sft/sft_scripts/run_math_sft.sbatch.sh
#   sbatch --export=ALL,MODEL_TYPE=llama,METHOD=lora sft/sft_scripts/run_math_sft.sbatch.sh

#SBATCH --export=ALL
#SBATCH --job-name=rpsft-sft
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=500000M
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=h200:8
#SBATCH --output=sft_%j.out
#SBATCH --error=sft_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
export REPO_ROOT

COMMON_ENV="${COMMON_ENV:-${REPO_ROOT}/sft/eval/common_env.sh}"
if [[ ! -f "${COMMON_ENV}" ]]; then
  echo "[ERROR] common env not found: ${COMMON_ENV}" >&2
  exit 1
fi
# shellcheck source=/dev/null
source "${COMMON_ENV}"

MODEL_TYPE="${MODEL_TYPE:-qwen}"       # llama | qwen | qwen-3B
METHOD="${METHOD:-rpsft}"              # sft | rpsft | iw | dft | lora
METHOD="$(echo "${METHOD}" | tr '[:upper:]' '[:lower:]')"

case "${MODEL_TYPE}" in
  llama)
    MODEL_NAME="${MODEL_NAME:-${SCRATCH}/data/Meta-Llama-3.1-8B-Instruct}"
    TRAIN_MODULE="training.train"
    DEFAULT_EPOCHS=20
    ;;
  qwen)
    MODEL_NAME="${MODEL_NAME:-${SCRATCH}/data/Qwen2.5-7B-Instruct}"
    TRAIN_MODULE="training.train_qwen"
    DEFAULT_EPOCHS=12
    ;;
  qwen-3B)
    MODEL_NAME="${MODEL_NAME:-${SCRATCH}/data/Qwen2.5-3B-Instruct}"
    TRAIN_MODULE="training.train_qwen"
    DEFAULT_EPOCHS=12
    ;;
  *)
    echo "[ERROR] MODEL_TYPE must be one of: llama, qwen, qwen-3B" >&2
    exit 1
    ;;
esac

case "${METHOD}" in
  sft)
    METHOD_DIR="sft"
    SFT_METHOD_ARG="base"
    SVD_REG_COEFF="${SVD_REG_COEFF:-0}"
    LORA_ENABLE="${LORA_ENABLE:-False}"
    ;;
  rpsft|svd)
    METHOD="rpsft"
    SVD_REG_TOPK="${SVD_REG_TOPK:-768}"
    METHOD_DIR="sft_svd_${SVD_REG_TOPK}"
    SFT_METHOD_ARG="base"
    SVD_REG_COEFF="${SVD_REG_COEFF:-1}"
    LORA_ENABLE="${LORA_ENABLE:-False}"
    ;;
  iw)
    METHOD_DIR="sft_iw"
    SFT_METHOD_ARG="iw"
    SVD_REG_COEFF="${SVD_REG_COEFF:-0}"
    LORA_ENABLE="${LORA_ENABLE:-False}"
    ;;
  dft)
    METHOD_DIR="sft_dft"
    SFT_METHOD_ARG="dft"
    SVD_REG_COEFF="${SVD_REG_COEFF:-0}"
    LORA_ENABLE="${LORA_ENABLE:-False}"
    ;;
  lora)
    METHOD_DIR="lora_sft"
    SFT_METHOD_ARG="base"
    SVD_REG_COEFF="${SVD_REG_COEFF:-0}"
    LORA_ENABLE="${LORA_ENABLE:-True}"
    ;;
  *)
    echo "[ERROR] METHOD must be one of: sft, rpsft, iw, dft, lora" >&2
    exit 1
    ;;
esac

SVD_REG_TOPK="${SVD_REG_TOPK:-768}"
LR="${LR:-1e-6}"
EPOCHS="${EPOCHS:-${DEFAULT_EPOCHS}}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-2}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
NUM_GPUS="${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-8}}"
NUM_NODES="${SLURM_JOB_NUM_NODES:-1}"
NODE_RANK="${SLURM_NODEID:-${SLURM_PROCID:-0}}"

DATA_JSON="${DATA_JSON:-${SCRATCH}/data/SFTvsRL_Data/SFT_Data/math-l/train_openr1.sft.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-.}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRATCH}/data/train_ckpt/sft_reg}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${MODEL_TYPE}/${METHOD_DIR}}"
LOG_FILE="${LOG_FILE:-train-${MODEL_TYPE}-${METHOD}.log}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${REPO_ROOT}/sft/sft_scripts/zero2_offload.json}"
SAVE_STEPS="${SAVE_STEPS:-3200}"

if [[ ! -f "${DATA_JSON}" ]]; then
  echo "[ERROR] SFT data not found: ${DATA_JSON}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"

echo "[INFO] model=${MODEL_TYPE} method=${METHOD}"
echo "[INFO] model path=${MODEL_NAME}"
echo "[INFO] data=${DATA_JSON}"
echo "[INFO] output=${OUTPUT_DIR}"
echo "[INFO] svd coeff=${SVD_REG_COEFF} topk=${SVD_REG_TOPK}"

python -m accelerate.commands.launch \
  --num_processes "${NUM_GPUS}" \
  --num_machines "${NUM_NODES}" \
  --machine_rank "${NODE_RANK}" \
  --mixed_precision bf16 \
  --main_process_port "${MASTER_PORT}" \
  --main_process_ip "${MASTER_ADDR}" \
  --use_deepspeed \
  --deepspeed_config_file "${DEEPSPEED_CONFIG}" \
  --deepspeed_multinode_launcher pdsh \
  --deepspeed_hostfile "${ACCELERATE_HOST_FILE}" \
  -m "${TRAIN_MODULE}" \
    --model_id "${MODEL_NAME}" \
    --ref_model_id "${MODEL_NAME}" \
    --data_path "${DATA_JSON}" \
    --image_folder "${IMAGE_FOLDER}" \
    --disable_flash_attn2 True \
    --lora_enable "${LORA_ENABLE}" \
    --freeze_llm False \
    --bf16 True \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs "${EPOCHS}" \
    --per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}" \
    --learning_rate "${LR}" \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --tf32 False \
    --gradient_checkpointing True \
    --report_to none \
    --lazy_preprocess True \
    --save_strategy steps \
    --save_steps "${SAVE_STEPS}" \
    --save_total_limit 100 \
    --dataloader_num_workers 4 \
    --svd_reg_coeff "${SVD_REG_COEFF}" \
    --svd_reg_topk "${SVD_REG_TOPK}" \
    --use_cot True \
    --sft_method "${SFT_METHOD_ARG}" \
    --iw_temperature "${IW_TEMPERATURE:-1.0}" \
    --iw_max_scale "${IW_MAX_SCALE:-16.0}" \
    --lora_rank "${LORA_RANK:-32}" \
    --lora_alpha "${LORA_ALPHA:-64}" \
    --lora_dropout "${LORA_DROPOUT:-0.0}" \
    --save_only_model "${SAVE_ONLY_MODEL:-False}" 2>&1 | tee "${LOG_FILE}"
