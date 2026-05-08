#!/usr/bin/env bash
set -euo pipefail

# Shared cluster/runtime setup for SFT and evaluation jobs.
# Override paths from the sbatch command line when your cluster differs.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
export REPO_ROOT

if command -v module >/dev/null 2>&1; then
  module load cuda/12.2 || true
  module load cudnn || true
fi

if [[ -z "${CUDA_HOME:-}" ]] && command -v nvcc >/dev/null 2>&1; then
  CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
  export CUDA_HOME
fi

CONDA_DIR="${CONDA_DIR:-$HOME/miniconda3}"
if [[ -f "${CONDA_DIR}/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "${CONDA_DIR}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV:-RPSFT}"
fi

if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export PATH="${CONDA_PREFIX}/bin:${PATH}"
fi

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/sft/src:${REPO_ROOT}/verl:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"

unset NCCL_DEBUG
unset NCCL_DEBUG_SUBSYS
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

if [[ -n "${SLURM_JOB_NODELIST:-}" ]] && command -v scontrol >/dev/null 2>&1; then
  export MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)}"
else
  export MASTER_ADDR="${MASTER_ADDR:-$(hostname -s)}"
fi
export MASTER_PORT="${MASTER_PORT:-$((10000 + RANDOM % 20000))}"

GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-${NUM_GPUS:-8}}"
HOSTFILE="${HOSTFILE:-${REPO_ROOT}/sft/eval/hostfile_${SLURM_JOB_ID:-local}}"
mkdir -p "$(dirname "${HOSTFILE}")"
if [[ -n "${SLURM_NODELIST:-}" ]] && command -v scontrol >/dev/null 2>&1; then
  scontrol show hostnames "${SLURM_NODELIST}" | awk -v slots="${GPUS_PER_NODE}" '{print $1" slots="slots}' > "${HOSTFILE}"
else
  printf "%s slots=%s\n" "$(hostname -s)" "${GPUS_PER_NODE}" > "${HOSTFILE}"
fi
export ACCELERATE_HOST_FILE="${HOSTFILE}"

if [[ -n "${CUDA_HOME:-}" ]]; then
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/extras/CUPTI/lib64:${LD_LIBRARY_PATH:-}"
fi
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:${CONDA_PREFIX}/lib"
fi

: "${SCRATCH:?Please export SCRATCH to a writable data/checkpoint root.}"
export HF_HOME="${HF_HOME:-${SCRATCH}/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export WANDB_DIR="${WANDB_DIR:-${SCRATCH}/wandb}"

echo "REPO_ROOT=${REPO_ROOT}"
echo "MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
echo "ACCELERATE_HOST_FILE=${ACCELERATE_HOST_FILE}"
echo "Python: $(command -v python || true)"
