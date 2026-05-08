#!/usr/bin/env bash
# Generic multi-GPU SFT/SVD eval script.
# Job name should be set by the caller via: sbatch --job-name=...
# Arguments:
#   1: STEP (or 0000 for base model)
#   2: DATA_NAME (e.g., aime-24)
#   3: TOPK suffix or method (e.g., _768, _0000, _base, sft_dft, sft_iw)
#   4: model_type (e.g., llama, qwen, deepseek_math)
#   5: distribution (id or ood)
#   6: PASS_AT_K
#   7: base model name (only used if STEP/ck == 0000)

#SBATCH --job-name=eval-sft
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --mem=500000M

set -euo pipefail

STEP="${1:-11200}"
DATA_NAME="${2:-}"
TOPK="${3:-}"
model_type="${4:-qwen-4B}"
distribution="${5:-id}"
PASS_AT_K="${6:-1}"
ck="${STEP}"

# Under sbatch, BASH_SOURCE still points at this script. Derive the extracted
# project root from it unless REPO_ROOT is explicitly provided.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
export REPO_ROOT

export SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-$(hostname)}
export SLURM_NODELIST=${SLURM_NODELIST:-$(hostname)}

COMMON_ENV="${COMMON_ENV:-${REPO_ROOT}/sft/eval/common_env.sh}"
if [[ ! -f "${COMMON_ENV}" ]]; then
  echo "[ERROR] common_env.sh not found at ${COMMON_ENV}" >&2
  exit 1
fi
export CONDA_ENV="${CONDA_ENV:-RPSFT}"
source "${COMMON_ENV}"

resolve_base_model_path() {
  local model="$1"
  case "${model}" in
    llama) echo "${BASE_MODEL:-${SCRATCH}/data/Meta-Llama-3.1-8B-Instruct}" ;;
    qwen) echo "${BASE_MODEL:-${SCRATCH}/data/Qwen2.5-7B-Instruct}" ;;
    qwen-3B) echo "${BASE_MODEL:-${SCRATCH}/data/Qwen2.5-3B-Instruct}" ;;
    deepseek_math) echo "${BASE_MODEL:-${SCRATCH}/data/deepseek-math-7b-instruct}" ;;
    *)
      echo "[ERROR] No default base model configured for model_type=${model}. Set BASE_MODEL to merge LoRA checkpoints." >&2
      exit 1
      ;;
  esac
}

is_adapter_only_checkpoint() {
  local model_dir="$1"
  [[ -f "${model_dir}/adapter_config.json" ]] && [[ ! -f "${model_dir}/config.json" ]]
}

ensure_vllm_ready_model_dir() {
  local model_dir="$1"
  local base_model merged_dir lock_dir tmp_dir

  if ! is_adapter_only_checkpoint "${model_dir}"; then
    printf '%s\n' "${model_dir}"
    return 0
  fi

  base_model="$(resolve_base_model_path "${model_type}")"
  if [[ ! -d "${base_model}" ]]; then
    echo "[ERROR] Base model directory not found for LoRA merge: ${base_model}" >&2
    exit 1
  fi

  merged_dir="${MERGED_MODEL_DIR:-${model_dir}-merged}"
  lock_dir="${merged_dir}.lock"

  if [[ -f "${merged_dir}/config.json" ]]; then
    echo "[INFO] Reusing merged LoRA checkpoint: ${merged_dir}" >&2
    printf '%s\n' "${merged_dir}"
    return 0
  fi

  if mkdir "${lock_dir}" 2>/dev/null; then
    tmp_dir="${merged_dir}.tmp.${SLURM_JOB_ID:-$$}"
    trap 'rm -rf "'"${lock_dir}"'" "'"${tmp_dir}"'"' EXIT
    if [[ ! -f "${merged_dir}/config.json" ]]; then
      rm -rf "${tmp_dir}"
      mkdir -p "${tmp_dir}"
      echo "[INFO] Merging LoRA adapter into a reusable checkpoint" >&2
      echo "       Adapter    : ${model_dir}" >&2
      echo "       Base model : ${base_model}" >&2
      echo "       Merged dir : ${merged_dir}" >&2
      "${PYTHON_BIN}" - <<PY >&2
import os
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

model_path = r"""${model_dir}"""
model_base = r"""${base_model}"""
save_model_path = r"""${tmp_dir}"""

print(f"[INFO] Loading base model from {model_base}")
model = AutoModelForCausalLM.from_pretrained(
    model_base,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    trust_remote_code=True,
)

non_lora_path = os.path.join(model_path, "non_lora_state_dict.bin")
if os.path.exists(non_lora_path):
    print(f"[INFO] Loading non-LoRA weights from {non_lora_path}")
    non_lora_trainables = torch.load(non_lora_path, map_location="cpu")
    non_lora_trainables = {
        (k[11:] if k.startswith("base_model.") else k): v
        for k, v in non_lora_trainables.items()
    }
    if any(k.startswith("model.model.") for k in non_lora_trainables):
        non_lora_trainables = {
            (k[6:] if k.startswith("model.") else k): v
            for k, v in non_lora_trainables.items()
        }
    model.load_state_dict(non_lora_trainables, strict=False)

print(f"[INFO] Loading LoRA adapter from {model_path}")
model = PeftModel.from_pretrained(model, model_path)
print("[INFO] Merging LoRA weights")
model = model.merge_and_unload()
model.save_pretrained(save_model_path, safe_serialization=True)

try:
    processor = AutoProcessor.from_pretrained(model_base, trust_remote_code=True)
    processor.save_pretrained(save_model_path)
except Exception:
    tokenizer = AutoTokenizer.from_pretrained(model_base, trust_remote_code=True, use_fast=True)
    tokenizer.save_pretrained(save_model_path)
PY
      if [[ ! -f "${tmp_dir}/config.json" ]]; then
        echo "[ERROR] LoRA merge did not produce config.json in ${tmp_dir}" >&2
        exit 1
      fi
      mv "${tmp_dir}" "${merged_dir}"
    fi
    rm -rf "${lock_dir}"
    trap - EXIT
  else
    echo "[INFO] Waiting for another job to finish merging ${model_dir}" >&2
    while [[ -d "${lock_dir}" ]]; do
      sleep 10
    done
    if [[ ! -f "${merged_dir}/config.json" ]]; then
      echo "[ERROR] Expected merged checkpoint missing after waiting: ${merged_dir}" >&2
      exit 1
    fi
  fi

  printf '%s\n' "${merged_dir}"
}


# If TOPK is "_0000" (or empty), use the non-SVD SFT directory.
# Keep the existing SVD behavior for suffixes like "_768", and additionally
# allow named SFT method directories for DFT/IW evals.
if [[ -z "${TOPK}" || "${TOPK}" == "_0000" ]]; then
  MODEL_ROOT="${MODEL_ROOT:-${SCRATCH}/data/train_ckpt/sft_reg/${model_type}/sft}"
elif [[ "${TOPK}" == "sft_dft" || "${TOPK}" == "dft" ]]; then
  MODEL_ROOT="${MODEL_ROOT:-${SCRATCH}/data/train_ckpt/sft_reg/${model_type}/sft_dft}"
elif [[ "${TOPK}" == "sft_iw" || "${TOPK}" == "iw" ]]; then
  MODEL_ROOT="${MODEL_ROOT:-${SCRATCH}/data/train_ckpt/sft_reg/${model_type}/sft_iw}"
else
  MODEL_ROOT="${MODEL_ROOT:-${SCRATCH}/data/train_ckpt/sft_reg/${model_type}/sft_svd${TOPK}}"
fi

if [[ "${ck}" == "0000" ]]; then
  CHECKPOINT_DIR="${BASE_MODEL:-${7:-$(resolve_base_model_path "${model_type}")}}"
else
  CHECKPOINT_DIR="${MODEL_ROOT}/checkpoint-${STEP}"
fi
if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
  echo "[ERROR] Checkpoint directory not found: ${CHECKPOINT_DIR}" >&2
  echo "        Make sure training has saved checkpoint-${STEP}." >&2
  exit 1
fi


TEST_FILE="${TEST_FILE:-${REPO_ROOT}/evaluation/data/${distribution}/${DATA_NAME}.jsonl}"
if [[ ! -f "${TEST_FILE}" ]]; then
  echo "[ERROR] Evaluation data file not found: ${TEST_FILE}" >&2
  exit 1
fi

NUM_GPUS="${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-8}}"
if ! [[ "${NUM_GPUS}" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] NUM_GPUS must be an integer, got ${NUM_GPUS}" >&2
  exit 1
fi
if (( NUM_GPUS < 1 )); then
  echo "[ERROR] NUM_GPUS must be at least 1" >&2
  exit 1
fi
if [[ -n "${SLURM_GPUS_ON_NODE:-}" ]] && (( NUM_GPUS > SLURM_GPUS_ON_NODE )); then
  echo "[ERROR] Requested NUM_GPUS=${NUM_GPUS} but only ${SLURM_GPUS_ON_NODE} GPUs allocated" >&2
  exit 1
fi

TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-10240}"
DATA_ID="${DATA_ID:-${DATA_NAME}}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EVAL_SEED="${EVAL_SEED:-}"
LOG_TAG="${LOG_TAG:-}"
EVAL_BACKEND="${EVAL_BACKEND:-hf}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-}"
VLLM_PIPELINE_PARALLEL_SIZE="${VLLM_PIPELINE_PARALLEL_SIZE:-}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-}"
VLLM_DTYPE="${VLLM_DTYPE:-}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-}"
AUTO_VISIBLE_GPUS="${AUTO_VISIBLE_GPUS:-0}"

auto_select_visible_gpus() {
  if [[ "${AUTO_VISIBLE_GPUS}" != "1" || "${EVAL_BACKEND}" != "vllm" ]]; then
    return 0
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[WARN] AUTO_VISIBLE_GPUS=1 but nvidia-smi is not available; keeping current GPU visibility." >&2
    return 0
  fi

  local -a healthy=()
  local idx ecc
  while IFS=, read -r idx ecc; do
    idx="${idx//[[:space:]]/}"
    ecc="${ecc//[[:space:]]/}"
    if [[ "${idx}" =~ ^[0-9]+$ && "${ecc}" =~ ^[0-9]+$ ]]; then
      if (( ecc == 0 )); then
        healthy+=("${idx}")
      else
        echo "[WARN] Excluding GPU ${idx}: volatile uncorrectable ECC=${ecc}" >&2
      fi
    fi
  done < <(nvidia-smi --query-gpu=index,ecc.errors.uncorrected.volatile.total --format=csv,noheader,nounits 2>/dev/null || true)

  if (( ${#healthy[@]} >= 8 )); then
    local selected8="${healthy[0]},${healthy[1]},${healthy[2]},${healthy[3]},${healthy[4]},${healthy[5]},${healthy[6]},${healthy[7]}"
    export CUDA_VISIBLE_DEVICES="${selected8}"
    export NUM_GPUS=8
    echo "[INFO] AUTO_VISIBLE_GPUS selected 8 GPUs: ${CUDA_VISIBLE_DEVICES}"
    return 0
  fi

  if (( ${#healthy[@]} >= 4 )); then
    local selected4="${healthy[0]},${healthy[1]},${healthy[2]},${healthy[3]}"
    export CUDA_VISIBLE_DEVICES="${selected4}"
    export NUM_GPUS=4
    export VLLM_TENSOR_PARALLEL_SIZE=4
    unset VLLM_PIPELINE_PARALLEL_SIZE
    VLLM_PIPELINE_PARALLEL_SIZE=""
    echo "[INFO] AUTO_VISIBLE_GPUS selected 4 GPUs: ${CUDA_VISIBLE_DEVICES}"
    echo "[INFO] Adjusted vLLM topology to TP=4, PP=1"
    return 0
  fi

  echo "[ERROR] AUTO_VISIBLE_GPUS found only ${#healthy[@]} healthy GPUs; need at least 4 for this eval." >&2
  exit 1
}

auto_select_visible_gpus

OUT_DIR="${OUT_DIR:-${CHECKPOINT_DIR}/eval/${DATA_ID}}"
mkdir -p "${OUT_DIR}"

echo "[INFO] Evaluating SVD checkpoint-${STEP}"
echo "       Model dir : ${CHECKPOINT_DIR}"
echo "       Eval data : ${TEST_FILE}"
echo "       Output dir: ${OUT_DIR}"
echo "       Data ID   : ${DATA_ID}"
echo "       Num GPUs  : ${NUM_GPUS}"
echo "       Batch size: ${BATCH_SIZE}"
echo "       MAX_NEW_TOKENS: ${MAX_NEW_TOKENS}"
echo "       CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
  else
    PYTHON_BIN="$(command -v python)"
  fi
fi

if [[ -z "${PYTHON_BIN}" ]]; then
  echo "[ERROR] Could not locate a python executable (checked CONDA_PREFIX and PATH)" >&2
  exit 1
fi

EVAL_MODEL_DIR="${CHECKPOINT_DIR}"
if [[ "${EVAL_BACKEND}" == "vllm" ]]; then
  EVAL_MODEL_DIR="$(ensure_vllm_ready_model_dir "${CHECKPOINT_DIR}")"
fi

SCRIPT_PATH="${REPO_ROOT}/evaluation/eval_local.py"
if [[ ! -f "${SCRIPT_PATH}" ]]; then
  echo "[ERROR] Evaluation script not found at ${SCRIPT_PATH}" >&2
  exit 1
fi

BASE_ARGS=(
  --model "${EVAL_MODEL_DIR}"
  --test_file "${TEST_FILE}"
  --out "${OUT_DIR}"
  --temperature "${TEMPERATURE}"
  --top_p "${TOP_P}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --data_id "${DATA_ID}"
  --batch_size "${BATCH_SIZE}"
  --k "${PASS_AT_K}"
)
if [[ -n "${EVAL_SEED}" ]]; then
  BASE_ARGS+=(--seed "${EVAL_SEED}")
fi
if [[ -n "${LOG_TAG}" ]]; then
  BASE_ARGS+=(--log_tag "${LOG_TAG}")
fi
if [[ "${EVAL_BACKEND}" == "vllm" ]]; then
  BASE_ARGS+=(--backend vllm)
  if [[ -n "${VLLM_PIPELINE_PARALLEL_SIZE}" ]]; then
    if ! [[ "${VLLM_PIPELINE_PARALLEL_SIZE}" =~ ^[0-9]+$ ]] || (( VLLM_PIPELINE_PARALLEL_SIZE < 1 )); then
      echo "[ERROR] VLLM_PIPELINE_PARALLEL_SIZE must be a positive integer" >&2
      exit 1
    fi
  fi
  if [[ -z "${VLLM_TENSOR_PARALLEL_SIZE}" ]]; then
    if [[ -n "${VLLM_PIPELINE_PARALLEL_SIZE}" ]]; then
      if (( NUM_GPUS % VLLM_PIPELINE_PARALLEL_SIZE != 0 )); then
        echo "[ERROR] NUM_GPUS=${NUM_GPUS} not divisible by VLLM_PIPELINE_PARALLEL_SIZE=${VLLM_PIPELINE_PARALLEL_SIZE}" >&2
        exit 1
      fi
      VLLM_TENSOR_PARALLEL_SIZE="$(( NUM_GPUS / VLLM_PIPELINE_PARALLEL_SIZE ))"
    else
      VLLM_TENSOR_PARALLEL_SIZE="${NUM_GPUS}"
    fi
  fi
  BASE_ARGS+=(--vllm_tensor_parallel_size "${VLLM_TENSOR_PARALLEL_SIZE}")
  if [[ -n "${VLLM_PIPELINE_PARALLEL_SIZE}" ]]; then
    BASE_ARGS+=(--vllm_pipeline_parallel_size "${VLLM_PIPELINE_PARALLEL_SIZE}")
  fi
  if [[ -n "${VLLM_GPU_MEMORY_UTILIZATION}" ]]; then
    BASE_ARGS+=(--vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}")
  fi
  if [[ -n "${VLLM_DTYPE}" ]]; then
    BASE_ARGS+=(--vllm_dtype "${VLLM_DTYPE}")
  fi
  if [[ -n "${VLLM_MAX_MODEL_LEN}" ]]; then
    BASE_ARGS+=(--vllm_max_model_len "${VLLM_MAX_MODEL_LEN}")
  fi
  if [[ -n "${VLLM_MAX_NUM_SEQS}" ]]; then
    BASE_ARGS+=(--vllm_max_num_seqs "${VLLM_MAX_NUM_SEQS}")
  fi
  if [[ -n "${VLLM_MAX_NUM_BATCHED_TOKENS}" ]]; then
    BASE_ARGS+=(--vllm_max_num_batched_tokens "${VLLM_MAX_NUM_BATCHED_TOKENS}")
  fi
fi

if [[ "${EVAL_BACKEND}" == "vllm" ]]; then
  LAUNCHER=("${PYTHON_BIN}")
else
  if (( NUM_GPUS > 1 )); then
    LAUNCHER=(
      "${PYTHON_BIN}"
      -m torch.distributed.run
      --standalone
      --nnodes=1
      --nproc_per_node="${NUM_GPUS}"
    )
  else
    LAUNCHER=("${PYTHON_BIN}")
  fi
fi

today=$(date '+%Y-%m-%d')
log_tag_suffix=""
if [[ -n "${LOG_TAG}" ]]; then
  log_tag_suffix="-${LOG_TAG}"
fi
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/sft/eval/final}"
log_dir_suffix="sft"
log_name_suffix="${TOPK}"
if [[ "${TOPK}" == "sft_dft" || "${TOPK}" == "dft" ]]; then
  log_dir_suffix="sft_dft"
  log_name_suffix="sft_dft"
elif [[ "${TOPK}" == "sft_iw" || "${TOPK}" == "iw" ]]; then
  log_dir_suffix="sft_iw"
  log_name_suffix="sft_iw"
elif [[ -n "${TOPK}" && "${TOPK}" != "_0000" ]]; then
  log_dir_suffix="sft_svd_${TOPK#_}"
fi
if [[ -z "${LOG_DIR:-}" ]]; then
  LOG_DIR="${LOG_ROOT}/${model_type}/${log_dir_suffix}"
fi
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/eval-sft-${log_name_suffix}-${today}-ck-${STEP}-data-${DATA_ID}-k-${PASS_AT_K}${log_tag_suffix}.log"
#LOG_FILE="${model_type}/eval-sft-${today}-ck-${STEP}-data-${DATA_ID}.log"
echo "[INFO] Launching evaluation via: ${LAUNCHER[*]} ${SCRIPT_PATH} ${BASE_ARGS[*]}"
echo "[INFO] Logging to ${LOG_FILE} (model=${EVAL_MODEL_DIR}, output=${OUT_DIR})"
"${LAUNCHER[@]}" "${SCRIPT_PATH}" "${BASE_ARGS[@]}" 2>&1 | tee "${LOG_FILE}"
