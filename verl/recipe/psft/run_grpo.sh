#!/usr/bin/env bash
set -xeuo pipefail
export WANDB_MODE=${WANDB_MODE:-offline}
ROOT=${ROOT:-"$(pwd)"}
export WANDB_DIR=${WANDB_DIR:-${SCRATCH:-${ROOT}}/wandb_logs}
export VLLM_LOGGING_LEVEL=ERROR
export VLLM_LOG_LEVEL=ERROR
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
#export PYTHONUNBUFFERED=1
export VERL_STEP_LOG_EVERY=${VERL_STEP_LOG_EVERY:-1}

project_name=${PROJECT_NAME:-'GRPO'}
exp_name_base=${EXP_NAME_BASE:-'GRPO'}
if [ -n "${EXP_NAME:-}" ]; then
    exp_name="${EXP_NAME}"
elif [ -n "${TRAIN_DATASET_TAG:-}" ]; then
    exp_name="${exp_name_base}-${TRAIN_DATASET_TAG}"
else
    exp_name="${exp_name_base}"
fi

adv_estimator=grpo
norm_adv_by_std_in_grpo=True

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0
kl_loss_type=low_var_kl

clip_ratio_low=0.2
clip_ratio_high=0.28

max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 10))
#enable_overlong_buffer=False
enable_overlong_buffer=True
overlong_buffer_len=$((1024 * 2))
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"

train_prompt_bsz=256
#gen_prompt_bsz=train_prompt_bsz
train_prompt_mini_bsz=32
n_resp_per_prompt=8

# Ray
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

# Paths
RAY_DATA_HOME=${RAY_DATA_HOME:-"./"}
MODEL_PATH=${MODEL_PATH:-"checkpoint-19200"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"dapo-math-17k.nosnappy.parquet"}
TEST_FILE=${TEST_FILE:-"${TRAIN_FILE}"}

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=0.7

# Performance Related Parameter
sp_size=1
use_dynamic_bsz=True
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 3))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 3))
offload=False
gen_tp=${GEN_TP:-1}
gen_pp=${GEN_PP:-1}
gen_dp=${GEN_DP:-8}
# Must exceed the largest single parameter tensor size during rollout weight sync.
# Qwen embedding can be >2GB in fp32, so 2048MB may assert.

TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-100}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-30}
SAVE_FREQ=${SAVE_FREQ:-100}
PYTHON_BIN=${PYTHON_BIN:-python3}

"${PYTHON_BIN}" -m verl.trainer.main_ppo \
    --config-path "${ROOT}/verl/recipe/psft/config" \
    --config-name "psft_trainer" \
    "hydra.searchpath=[file://${ROOT}/verl/verl/trainer/config]" \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=8 \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.gen_batch_size=128 \
    data.train_batch_size=${train_prompt_bsz} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.norm_adv_by_std_in_grpo=${norm_adv_by_std_in_grpo} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.kl_loss_type=${kl_loss_type} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.rollout.max_num_seqs=128 \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=2560 \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.checkpoint.save_contents="['model','extra']" \
    actor_rollout_ref.actor.checkpoint.load_contents="['model','extra']" \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k="${top_k}" \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.nccl_timeout=7200 \
    reward_model.reward_manager=dapo \
    reward_model.overlong_buffer.enable=${enable_overlong_buffer} \
    reward_model.overlong_buffer.len=${overlong_buffer_len} \
    reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=${NGPUS_PER_NODE} \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS} \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto  | tee ${exp_name}.log
