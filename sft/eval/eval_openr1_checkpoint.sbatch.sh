#!/usr/bin/env bash
# Evaluate an arbitrary checkpoint directory locally (no remote server).
# Uses evaluation/eval_local.py with optional vLLM backend.
#
# Usage:
#   sbatch --job-name=eval-openr1-<tag> eval_openr1_checkpoint.sbatch.sh \
#       /path/to/checkpoint-dir aime-25 id 16 openr1_baseline
#
# Args:
#   1: CHECKPOINT_DIR (absolute path)
#   2: DATA_NAME      (e.g., aime-25)
#   3: DISTRIBUTION   (id|ood)  [default: id]
#   4: PASS_AT_K      [default: 16]
#   5: METHOD_TAG     (string used in logs)

#SBATCH --job-name=eval-openr1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=3:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --mem=500000M

set -euo pipefail

CHECKPOINT_DIR="${1:?CHECKPOINT_DIR required}"
DATA_NAME="${2:?DATA_NAME required}"
DISTRIBUTION="${3:-id}"
PASS_AT_K="${4:-16}"
METHOD_TAG="${5:-method}"
TRAINING_DATASET="${6:-openr1_qwen_grpo}"
MODEL_TYPE="${MODEL_TYPE:-qwen}"
SFT_TYPE="${SFT_TYPE:-sft}"

# Derive the extracted project root unless REPO_ROOT is explicitly provided.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
export REPO_ROOT

# Use vLLM engine directly (no HTTP). Enable with: VLLM=1 (default).
VLLM="${VLLM:-1}"
if [[ "${VLLM}" == "1" || "${VLLM}" == "true" ]]; then
  CONDA_ENV="${CONDA_ENV:-RPSFT}"
fi

COMMON_ENV="${COMMON_ENV:-${REPO_ROOT}/sft/eval/common_env.sh}"
if [[ ! -f "${COMMON_ENV}" ]]; then
  echo "[ERROR] common_env.sh not found at ${COMMON_ENV}" >&2
  exit 1
fi
source "${COMMON_ENV}"

if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
  echo "[ERROR] Checkpoint directory not found: ${CHECKPOINT_DIR}" >&2
  exit 1
fi

TEST_FILE="${TEST_FILE:-${REPO_ROOT}/evaluation/data/${DISTRIBUTION}/${DATA_NAME}.jsonl}"
if [[ ! -f "${TEST_FILE}" ]]; then
  echo "[ERROR] Evaluation data file not found: ${TEST_FILE}" >&2
  exit 1
fi

# vLLM tensor-parallel size (should match allocated GPUs on this node)
TP_SIZE="${TP_SIZE:-${SLURM_GPUS_ON_NODE:-1}}"
PP_SIZE="${PP_SIZE:-1}"
AUTO_VISIBLE_GPUS="${AUTO_VISIBLE_GPUS:-0}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
MAX_TOKENS="${MAX_TOKENS:-16384}"
EVAL_SEED="${EVAL_SEED:-}"

auto_select_visible_gpus() {
  if [[ "${AUTO_VISIBLE_GPUS}" != "1" || ! ( "${VLLM}" == "1" || "${VLLM}" == "true" ) ]]; then
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
    export CUDA_VISIBLE_DEVICES="${healthy[0]},${healthy[1]},${healthy[2]},${healthy[3]},${healthy[4]},${healthy[5]},${healthy[6]},${healthy[7]}"
    TP_SIZE="${TP_SIZE:-4}"
    PP_SIZE="${PP_SIZE:-2}"
    echo "[INFO] AUTO_VISIBLE_GPUS selected 8 GPUs: ${CUDA_VISIBLE_DEVICES}"
    return 0
  fi

  if (( ${#healthy[@]} >= 4 )); then
    export CUDA_VISIBLE_DEVICES="${healthy[0]},${healthy[1]},${healthy[2]},${healthy[3]}"
    TP_SIZE=4
    PP_SIZE=1
    echo "[INFO] AUTO_VISIBLE_GPUS selected 4 GPUs: ${CUDA_VISIBLE_DEVICES}"
    echo "[INFO] Adjusted vLLM topology to TP=4, PP=1"
    return 0
  fi

  echo "[ERROR] AUTO_VISIBLE_GPUS found only ${#healthy[@]} healthy GPUs; need at least 4." >&2
  exit 1
}

auto_select_visible_gpus

OUT_DIR="${OUT_DIR:-${CHECKPOINT_DIR}/eval/${DATA_NAME}}"
mkdir -p "${OUT_DIR}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
  else
    PYTHON_BIN="$(command -v python)"
  fi
fi
if [[ -z "${PYTHON_BIN}" ]]; then
  echo "[ERROR] Could not locate python" >&2
  exit 1
fi

EVAL_SCRIPT_PATH="${REPO_ROOT}/evaluation/eval_local.py"
if [[ ! -f "${EVAL_SCRIPT_PATH}" ]]; then
  echo "[ERROR] evaluation script not found at ${EVAL_SCRIPT_PATH}" >&2
  exit 1
fi

BATCH_SIZE="${BATCH_SIZE:-32}"

if [[ "${VLLM}" == "1" || "${VLLM}" == "true" ]]; then
  if ! "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import importlib.util
import sys
sys.exit(0 if importlib.util.find_spec("vllm") else 1)
PY
  then
    echo "[WARN] vllm not available in ${PYTHON_BIN}. Falling back to HF backend." >&2
    echo "[WARN] Install vllm in the active env or set VLLM=0 to silence this warning." >&2
    VLLM=0
  fi
fi

BASE_ARGS=(
  --model "${CHECKPOINT_DIR}"
  --test_file "${TEST_FILE}"
  --out "${OUT_DIR}"
  --temperature "${TEMPERATURE}"
  --top_p "${TOP_P}"
  --max_new_tokens "${MAX_TOKENS}"
  --data_id "${DATA_NAME}"
  --batch_size "${BATCH_SIZE}"
  --k "${PASS_AT_K}"
)
if [[ -n "${EVAL_SEED}" ]]; then
  BASE_ARGS+=(--seed "${EVAL_SEED}")
fi

if [[ "${VLLM}" == "1" || "${VLLM}" == "true" ]]; then
  BASE_ARGS+=(--backend vllm --vllm_tensor_parallel_size "${TP_SIZE}")
  if [[ "${PP_SIZE}" =~ ^[1-9][0-9]*$ ]] && (( PP_SIZE > 1 )); then
    BASE_ARGS+=(--vllm_pipeline_parallel_size "${PP_SIZE}")
  fi
fi

today=$(date '+%Y-%m-%d')
LOG_DIR="${REPO_ROOT}/sft/eval/final/${TRAINING_DATASET}/${MODEL_TYPE}/${SFT_TYPE}"
mkdir -p "${LOG_DIR}"

# Make checkpoint path safe-ish for filename
ck_leaf=$(basename "${CHECKPOINT_DIR}")
LOG_FILE="${LOG_DIR}/${METHOD_TAG}-${today}-${ck_leaf}-data-${DATA_NAME}-k-${PASS_AT_K}.log"

echo "[INFO] Method      : ${METHOD_TAG}"
echo "[INFO] Model dir   : ${CHECKPOINT_DIR}"
echo "[INFO] Eval data   : ${TEST_FILE}"
echo "[INFO] Output dir  : ${OUT_DIR}"
echo "[INFO] TP_SIZE     : ${TP_SIZE}"
echo "[INFO] PP_SIZE     : ${PP_SIZE}"
echo "[INFO] CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "[INFO] pass@k      : ${PASS_AT_K}"
echo "[INFO] Logging to  : ${LOG_FILE}"

echo "[INFO] Launch eval: ${PYTHON_BIN} ${EVAL_SCRIPT_PATH} ${BASE_ARGS[*]}"
echo "[INFO] Logging to  : ${LOG_FILE}"
"${PYTHON_BIN}" "${EVAL_SCRIPT_PATH}" "${BASE_ARGS[@]}" 2>&1 | tee "${LOG_FILE}"
