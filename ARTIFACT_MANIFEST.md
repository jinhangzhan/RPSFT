# Artifact Manifest

Included:

- `sft/src/`: SFT training code, including vanilla SFT, RPSFT SVD regularization, LoRA save/merge support, IW-SFT, and DFT trainer paths.
- `sft/sft_scripts/`: paper SFT training entrypoint and DeepSpeed configs.
- `slurm/run_grpo_psft_qwen_svd.sbatch`: RLFT entrypoint.
- `verl/`: VERL source needed by the RLFT entrypoint and PSFT recipe config.
- `sft/eval/`: SFT and RL checkpoint evaluation entrypoints.
- `evaluation/`: local HF/vLLM evaluator, answer checkers, and paper ID/OOD benchmark JSONL files.
- `scripts/data/`: SFT and RL data preparation helpers.

Excluded:

- Analysis notebooks/scripts, plotting code, generated figures, generated logs, WandB runs, caches, checkpoints, private cluster paths, and tokens.
- The old `PSFT/` project root. Only the VERL framework and the paper RL entrypoint were carried forward.
