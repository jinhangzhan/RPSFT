#!/usr/bin/env bash
# Paper RLFT entrypoint: DAPO/GRPO over a supervised checkpoint using VERL.

#SBATCH --job-name=rpsft-grpo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=500000M
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/grpo_%j.out
#SBATCH --error=slurm_logs/grpo_%j.err
#SBATCH --gpus-per-node=h200:8
#SBATCH --cpus-per-task=32

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
export REPO_ROOT

if [[ ! -f "${REPO_ROOT}/verl/recipe/psft/run_grpo.sh" ]]; then
  echo "[ERROR] Missing ${REPO_ROOT}/verl/recipe/psft/run_grpo.sh" >&2
  exit 1
fi

cd "${REPO_ROOT}"
mkdir -p slurm_logs

: "${SCRATCH:?Please export SCRATCH to a writable data/checkpoint root.}"

if command -v module >/dev/null 2>&1; then
  module load cuda/12.2 || true
  module load cudnn || true
fi
if [[ -z "${CUDA_HOME:-}" ]] && command -v nvcc >/dev/null 2>&1; then
  CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
  export CUDA_HOME
fi

CONDA_DIR="${CONDA_DIR:-$HOME/miniconda3}"
if [[ ! -f "${CONDA_DIR}/etc/profile.d/conda.sh" ]]; then
  echo "[ERROR] Conda activation script not found: ${CONDA_DIR}/etc/profile.d/conda.sh" >&2
  exit 1
fi
# shellcheck source=/dev/null
source "${CONDA_DIR}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-RPSFT}"

PYTHON_BIN="${PYTHON_BIN:-${CONDA_PREFIX}/bin/python}"
RAY_PY_CMD="${PYTHON_BIN} -m ray.scripts.scripts"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[ERROR] Python not found: ${PYTHON_BIN}" >&2
  exit 1
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export WANDB_DIR="${WANDB_DIR:-${SCRATCH}/wandb_logs}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DISABLE_SERVICE="${WANDB_DISABLE_SERVICE:-true}"
export MKL_THREADING_LAYER=GNU
unset MKL_SERVICE_FORCE_INTEL

export NNODES="${SLURM_JOB_NUM_NODES:-1}"
export NGPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-8}"
export GEN_TP="${GEN_TP:-1}"
export GEN_PP="${GEN_PP:-1}"
export GEN_DP="${GEN_DP:-$((NNODES * NGPUS_PER_NODE))}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"

# Data and checkpoint selection.
DATA_ROOT="${DATA_ROOT:-${SCRATCH}/data/SFTvsRL_Data/SFT_Data/math-l}"
TRAIN_DATASET_NAME="${TRAIN_DATASET_NAME:-dapo-math-17k.nosnappy.parquet}"
DATA_PARQUET="${DATA_ROOT}/${TRAIN_DATASET_NAME}"
if [[ ! -f "${DATA_PARQUET}" ]]; then
  echo "[ERROR] RL training parquet not found: ${DATA_PARQUET}" >&2
  exit 1
fi

DATASET_TAG_RAW="${TRAIN_DATASET_NAME%.parquet}"
DATASET_TAG_RAW="${DATASET_TAG_RAW%.nosnappy}"
DATASET_TAG="$(echo "${DATASET_TAG_RAW}" | sed -E 's/[^A-Za-z0-9._-]+/-/g')"
export TRAIN_DATASET_TAG="${DATASET_TAG}"

MODEL_TYPE="${MODEL_TYPE:-qwen}"
SFT_TYPE="${SFT_TYPE:-sft_svd_768}"
CKPT_STEP="${CKPT_STEP:-19200}"
CKPT_NAME="${CKPT_NAME:-${SCRATCH}/data/train_ckpt/sft_reg/${MODEL_TYPE}/${SFT_TYPE}/checkpoint-${CKPT_STEP}}"
if [[ ! -d "${CKPT_NAME}" ]]; then
  echo "[ERROR] Supervised checkpoint not found: ${CKPT_NAME}" >&2
  exit 1
fi

RUN_TAG="${RUN_TAG:-${MODEL_TYPE}-${SFT_TYPE}-ckpt${CKPT_STEP}-${DATASET_TAG}}"
export CKPTS_DIR="${CKPTS_DIR:-${SCRATCH}/data/train_ckpt/grpo/${MODEL_TYPE}/${SFT_TYPE}/${DATASET_TAG}/${CKPT_STEP}}"
export EXP_NAME="${EXP_NAME:-${MODEL_TYPE}-${SFT_TYPE}-ckpt${CKPT_STEP}-GRPO-${DATASET_TAG}}"
export PROJECT_NAME="${PROJECT_NAME:-GRPO}"
export MODEL_PATH="${CKPT_NAME}"
export TRAIN_FILE="${DATA_PARQUET}"
export TEST_FILE="${TEST_FILE:-${REPO_ROOT}/evaluation/data/aime24-eval.rewardfix.nosnappy.parquet}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-100}"
export SAVE_FREQ="${SAVE_FREQ:-100}"

if [[ ! -f "${TEST_FILE}" ]]; then
  echo "[ERROR] RL validation parquet not found: ${TEST_FILE}" >&2
  exit 1
fi

echo "[INFO] REPO_ROOT=${REPO_ROOT}"
echo "[INFO] MODEL_PATH=${MODEL_PATH}"
echo "[INFO] TRAIN_FILE=${TRAIN_FILE}"
echo "[INFO] CKPTS_DIR=${CKPTS_DIR}"
echo "[INFO] EXP_NAME=${EXP_NAME}"

unset PYTHONPATH
unset LD_LIBRARY_PATH
if [[ -n "${CUDA_HOME:-}" ]]; then
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/extras/CUPTI/lib64"
fi
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:${CONDA_PREFIX}/lib"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/verl"

export RAY_memory_monitor_refresh_ms=0
export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1
export RAY_DEDUP_LOGS=0
export RAY_LOG_TO_STDERR=0
export RAY_LOG_LEVEL=ERROR
export RAY_BACKEND_LOG_LEVEL=ERROR
export VLLM_LOGGING_LEVEL=ERROR
export VLLM_LOG_LEVEL=ERROR
export VLLM_ENGINE_ITERATION_TIMEOUT_S="${VLLM_ENGINE_ITERATION_TIMEOUT_S:-300}"
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-1500}"
export VLLM_RPC_TIMEOUT="${VLLM_RPC_TIMEOUT:-60000}"
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=7200
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=7200
unset PYTORCH_CUDA_ALLOC_CONF
export PYTORCH_ALLOC_CONF="expandable_segments:False"

RAY_TMPDIR_BASE="${SLURM_TMPDIR:-/tmp}/ray/${SLURM_JOB_ID:-local}"
mkdir -p "${RAY_TMPDIR_BASE}"
export RAY_TMPDIR="${RAY_TMPDIR_BASE}"
export TMPDIR="${RAY_TMPDIR_BASE}"
ulimit -n 65536

if [[ -n "${SLURM_NODELIST:-}" ]] && command -v scontrol >/dev/null 2>&1; then
  mapfile -t nodes < <(scontrol show hostnames "${SLURM_NODELIST}")
else
  nodes=("$(hostname -s)")
fi
head_node="${nodes[0]}"
if command -v srun >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then
  head_ip="$(srun --nodes=1 --ntasks=1 -w "${head_node}" hostname --ip-address)"
else
  head_ip="$(hostname --ip-address 2>/dev/null || hostname -s)"
fi
if [[ "${head_ip}" == *" "* ]]; then
  read -r first second _ <<< "${head_ip}"
  head_ip="${first}"
  if [[ ${#head_ip} -gt 16 && -n "${second:-}" ]]; then
    head_ip="${second}"
  fi
fi
ray_port="${RAY_PORT:-6379}"
export RAY_ADDRESS="${head_ip}:${ray_port}"
export RAY_NODE_IP_ADDRESS="${head_ip}"

RAY_CPUS_PER_NODE="${SLURM_CPUS_ON_NODE:-${SLURM_CPUS_PER_TASK:-8}}"
RAY_GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-${NGPUS_PER_NODE:-8}}"

echo "[INFO] Starting Ray head on ${head_node} (${RAY_ADDRESS})"
if command -v srun >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then
  srun --overlap --nodes=1 --ntasks=1 -w "${head_node}" bash -lc \
    "export RAY_TMPDIR='${RAY_TMPDIR_BASE}'; export TMPDIR=\"\$RAY_TMPDIR\"; \
     ${RAY_PY_CMD} stop -f >/dev/null 2>&1 || true; \
     ${RAY_PY_CMD} start --head --node-ip-address=${head_ip} --port=${ray_port} \
       --dashboard-host=0.0.0.0 --num-cpus=${RAY_CPUS_PER_NODE} \
       --num-gpus=${RAY_GPUS_PER_NODE} --block" &
else
  ${RAY_PY_CMD} stop -f >/dev/null 2>&1 || true
  ${RAY_PY_CMD} start --head --node-ip-address="${head_ip}" --port="${ray_port}" \
    --dashboard-host=0.0.0.0 --num-cpus="${RAY_CPUS_PER_NODE}" \
    --num-gpus="${RAY_GPUS_PER_NODE}" --block &
fi
sleep 10

for node in "${nodes[@]:1}"; do
  echo "[INFO] Starting Ray worker on ${node}"
  srun --overlap --nodes=1 --ntasks=1 -w "${node}" bash -lc \
    "export RAY_TMPDIR='${RAY_TMPDIR_BASE}'; export TMPDIR=\"\$RAY_TMPDIR\"; \
     node_ip=\$(hostname --ip-address | awk '{print \$1}'); \
     ${RAY_PY_CMD} stop -f >/dev/null 2>&1 || true; \
     ${RAY_PY_CMD} start --address=${RAY_ADDRESS} --node-ip-address=\${node_ip} \
       --num-cpus=${RAY_CPUS_PER_NODE} --num-gpus=${RAY_GPUS_PER_NODE} --block" &
  sleep 5
done

for i in $(seq 1 30); do
  if ${RAY_PY_CMD} status --address="${RAY_ADDRESS}" >/dev/null 2>&1; then
    break
  fi
  if [[ "${i}" -eq 30 ]]; then
    echo "[ERROR] Ray GCS is not reachable at ${RAY_ADDRESS}" >&2
    exit 1
  fi
  sleep 2
done

if command -v srun >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then
  srun --overlap --nodes=1 --ntasks=1 -w "${head_node}" bash -lc \
    "cd '${REPO_ROOT}'; export ROOT='${REPO_ROOT}'; \
     export RAY_NODE_IP_ADDRESS='${head_ip}'; \
     export RAY_TMPDIR='${RAY_TMPDIR_BASE}'; export TMPDIR=\"\$RAY_TMPDIR\"; \
     bash verl/recipe/psft/run_grpo.sh"
else
  export ROOT="${REPO_ROOT}"
  bash verl/recipe/psft/run_grpo.sh
fi
