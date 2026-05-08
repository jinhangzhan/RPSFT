# RPSFT

Minimal release artifact for the paper experiments:

- SFT-stage training for vanilla SFT, RPSFT, LoRA, IW-SFT, and DFT.
- RLFT with DAPO/GRPO using the vendored VERL framework.
- SFT and RL checkpoint evaluation on the paper ID/OOD benchmark split.

The project intentionally excludes plotting, analysis notebooks, generated logs,
checkpoint files, private paths, and cluster-specific credentials.

## Layout

```text
RPSFT/
  sft/src/                  # SFT trainers and RPSFT/IW/DFT implementations
  sft/sft_scripts/          # SFT entrypoints and DeepSpeed configs
  sft/eval/                 # SFT/RL eval sbatch entrypoints
  evaluation/               # Local HF/vLLM evaluator and ID/OOD data files
  verl/                     # VERL framework subset used for DAPO/GRPO
  slurm/                    # RLFT sbatch entrypoint
  scripts/data/             # Dataset preparation helpers
```

## Environment

Create a Python environment with CUDA-capable PyTorch, then install the project
dependencies. VERL has its own dependency file, so install both sets.

```bash
conda create -n RPSFT python=3.10 -y
conda activate RPSFT
pip install -r requirements.txt
pip install -r verl/requirements.txt
pip install -e verl
```

Set these environment variables before running cluster jobs:

```bash
export SCRATCH=/path/to/scratch
export REPO_ROOT=/path/to/RPSFT
```

If you use gated Hugging Face models, authenticate outside the scripts, for
example with `huggingface-cli login` or `HF_TOKEN`.

## Data

Prepare SFT JSON:

```bash
python scripts/data/prepare_openr1_sft_json.py \
  --dataset wh-zhu/train_openr1_4k \
  --output "$SCRATCH/data/SFTvsRL_Data/SFT_Data/math-l/train_openr1.sft.json"
```

Prepare RL parquet:

```bash
python scripts/data/prepare_dapo_rl_parquet.py \
  --dataset wh-zhu/dapo \
  --output "$SCRATCH/data/SFTvsRL_Data/SFT_Data/math-l/dapo-math-17k.nosnappy.parquet"
```

The evaluator includes the paper ID/OOD JSONL files under `evaluation/data/`.

## SFT Training

Use one entrypoint and select the method/model with environment variables.

```bash
# Vanilla SFT
sbatch --export=ALL,MODEL_TYPE=qwen,METHOD=sft sft/sft_scripts/run_math_sft.sbatch.sh

# RPSFT, default k=768 and lambda=1
sbatch --export=ALL,MODEL_TYPE=qwen,METHOD=rpsft,SVD_REG_TOPK=768 sft/sft_scripts/run_math_sft.sbatch.sh

# IW and DFT
sbatch --export=ALL,MODEL_TYPE=qwen,METHOD=iw sft/sft_scripts/run_math_sft.sbatch.sh
sbatch --export=ALL,MODEL_TYPE=qwen,METHOD=dft sft/sft_scripts/run_math_sft.sbatch.sh

# LoRA baseline
sbatch --export=ALL,MODEL_TYPE=llama,METHOD=lora sft/sft_scripts/run_math_sft.sbatch.sh
```

Supported `MODEL_TYPE` values are `llama`, `qwen`, and `qwen-3B`. Override
`MODEL_NAME`, `DATA_JSON`, or `OUTPUT_DIR` if your files are not under the
default `$SCRATCH/data/...` layout.

## RLFT

Run DAPO/GRPO from an SFT checkpoint:

```bash
sbatch --export=ALL,MODEL_TYPE=qwen,SFT_TYPE=sft_svd_768,CKPT_STEP=19200 \
  slurm/run_grpo_psft_qwen_svd.sbatch
```

The RL script defaults to `dapo-math-17k.nosnappy.parquet`, 8 responses per
prompt, batch size 256, and 100 total training steps, matching the paper setup.

## Evaluation

Evaluate an SFT-stage checkpoint:

```bash
sbatch sft/eval/eval_sft_checkpoint_multigpu.sbatch.sh \
  19200 aime-25 _768 qwen id 16
```

Evaluate an RL-stage checkpoint:

```bash
sbatch sft/eval/eval_openr1_checkpoint.sbatch.sh \
  "$SCRATCH/data/train_ckpt/grpo/qwen/sft_svd_768/dapo-math-17k/19200/global_step_100/actor/hf_merged" \
  aime-25 id 16 grpo_rpsft
```

For OOD evaluation, pass `ood` and one of the OOD data names, e.g. `gpqa_test`,
`mmlu_pro`, `ifeval_loose_test`, `safety_benchmark_test`, or
`truthful_qa_mc_test`.
