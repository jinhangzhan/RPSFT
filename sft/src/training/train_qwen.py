import json
import os
import re
import time
import torch
import transformers
from peft import LoraConfig, get_peft_model
import ast
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM, AutoConfig
from training.trainer import (
    QwenTrainer,
    QwenLoraTrainer,
    QwenKLSFTTrainer,
    QwenKLLoraTrainer,
    QwenDFTTrainer,
    QwenDFTLoraTrainer,
    QwenIwSFTTrainer,
    QwenIwSFTLoraTrainer,
)
from training.data_qwen import make_supervised_data_module
from training.params import DataArguments, ModelArguments, TrainingArguments
from training.train_utils import (
    get_peft_state_maybe_zero_3,
    get_peft_state_non_lora_maybe_zero_3,
    safe_save_model_for_hf_trainer,
)
from torch.nn import Linear
import pathlib
from typing import Dict, List, Optional
import torch.distributed as dist
from tqdm import tqdm
local_rank = None
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from torch.nn import Embedding, Linear
try:
    from transformers.models.qwen2.modeling_qwen2 import (
        Qwen2ColumnParallelLinear as ColLinear,
        Qwen2RowParallelLinear as RowLinear,
    )
except ImportError:
    ColLinear = RowLinear = Linear

def rank0_print(*args):
    if local_rank == 0 or local_rank == "0" or local_rank is None:
        print(*args)

def find_target_linear_names(
    model, num_lora_modules=-1, lora_namespan_exclude=[], verbose=True
):
    linear_cls = torch.nn.modules.Linear
    embedding_cls = torch.nn.modules.Embedding
    lora_module_names = []

    for name, module in model.named_modules():
        if any(ex_keyword in name for ex_keyword in lora_namespan_exclude):
            continue
        if isinstance(module, (linear_cls, embedding_cls)):
            lora_module_names.append(name)

    if num_lora_modules > 0:
        lora_module_names = lora_module_names[-num_lora_modules:]
    if verbose:
        rank0_print(f"Found {len(lora_module_names)} lora modules: {lora_module_names}")
    return lora_module_names


def set_requires_grad(parameters, requires_grad):
    for p in parameters:
        p.requires_grad = requires_grad


def configure_llm(model, training_args):
    llm_params = model.parameters()
    set_requires_grad(llm_params, not training_args.freeze_llm)

def _gather_full_param(param):
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            return None
        with zero.GatheredParameters([param], modifier_rank=None):
            return param.detach().cpu().clone()
    return param.detach().cpu().clone()

keep_kw = ("q_proj","k_proj","v_proj", "down_proj" 
           "language_model.model.embed_tokens",
           "language_model.lm_head")

def _infer_resume_step(resume_path: Optional[str]) -> int:
    if not resume_path:
        return 0

    resume_dir = pathlib.Path(resume_path)
    if not resume_dir.exists():
        return 0

    trainer_state = resume_dir / "trainer_state.json"
    if trainer_state.is_file():
        try:
            with open(trainer_state, "r", encoding="utf-8") as f:
                state = json.load(f)
            step = int(state.get("global_step", 0))
            if step:
                return step
        except (OSError, ValueError, TypeError):
            pass

    match = re.search(r"checkpoint-(\d+)", resume_dir.name)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return 0
    return 0

def _select_resume_checkpoint(output_dir: str, resume_from: Optional[str]) -> Optional[str]:
    candidate = resume_from

    if isinstance(candidate, bool):
        if not candidate:
            return None
        candidate = None

    if isinstance(candidate, str):
        lowered = candidate.strip().lower()
        if lowered in {"true", "t", "yes", "y"}:
            candidate = None
        elif lowered in {"false", "f", "no", "n"}:
            return None

    if candidate:
        return candidate

    output_path = pathlib.Path(output_dir)
    if not output_path.exists():
        return None

    best_path = None
    best_step = -1
    for path in output_path.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        step = _infer_resume_step(str(path))
        if step >= best_step:
            best_path = path
            best_step = step

    return str(best_path) if best_path else None


def _is_world_process_zero() -> bool:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0

    rank = os.getenv("RANK")
    if rank is not None:
        try:
            return int(rank) == 0
        except ValueError:
            pass
    return True


class TrainBenchmarkCallback(transformers.TrainerCallback):
    def __init__(self, method_name: str):
        self.method_name = method_name
        self.started = False
        self.start_time: Optional[float] = None
        self.start_step = 0
        self.start_allocated_gib = 0.0
        self.start_reserved_gib = 0.0

    @staticmethod
    def _bytes_to_gib(num_bytes: float) -> float:
        return float(num_bytes) / (1024 ** 3)

    def on_step_begin(self, args, state, control, **kwargs):
        if self.started:
            return control

        self.started = True
        self.start_step = int(state.global_step)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.start_allocated_gib = self._bytes_to_gib(torch.cuda.memory_allocated())
            self.start_reserved_gib = self._bytes_to_gib(torch.cuda.memory_reserved())
            torch.cuda.reset_peak_memory_stats()

        self.start_time = time.perf_counter()
        rank0_print(
            "[TRAIN_BENCHMARK] "
            f"method={self.method_name} "
            f"measurement_started_at_step={self.start_step + 1} "
            f"start_allocated_gib={self.start_allocated_gib:.3f} "
            f"start_reserved_gib={self.start_reserved_gib:.3f}"
        )
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if not self.started or self.start_time is None:
            return control

        elapsed_sec = time.perf_counter() - self.start_time
        peak_allocated_gib = self.start_allocated_gib
        peak_reserved_gib = self.start_reserved_gib

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_allocated_gib = self._bytes_to_gib(torch.cuda.max_memory_allocated())
            peak_reserved_gib = self._bytes_to_gib(torch.cuda.max_memory_reserved())

        measured_steps = max(int(state.global_step) - self.start_step, 0)
        summary = {
            "method": self.method_name,
            "measured_steps": measured_steps,
            "elapsed_sec": float(elapsed_sec),
            "start_allocated_gib": float(self.start_allocated_gib),
            "start_reserved_gib": float(self.start_reserved_gib),
            "peak_allocated_gib": float(peak_allocated_gib),
            "peak_reserved_gib": float(peak_reserved_gib),
            "peak_allocated_delta_gib": float(peak_allocated_gib - self.start_allocated_gib),
            "peak_reserved_delta_gib": float(peak_reserved_gib - self.start_reserved_gib),
        }

        if dist.is_available() and dist.is_initialized():
            device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
            metrics = torch.tensor(
                [
                    summary["elapsed_sec"],
                    summary["start_allocated_gib"],
                    summary["start_reserved_gib"],
                    summary["peak_allocated_gib"],
                    summary["peak_reserved_gib"],
                ],
                device=device,
                dtype=torch.float64,
            )
            dist.all_reduce(metrics, op=dist.ReduceOp.MAX)
            summary["elapsed_sec"] = float(metrics[0].item())
            summary["start_allocated_gib"] = float(metrics[1].item())
            summary["start_reserved_gib"] = float(metrics[2].item())
            summary["peak_allocated_gib"] = float(metrics[3].item())
            summary["peak_reserved_gib"] = float(metrics[4].item())
            summary["peak_allocated_delta_gib"] = (
                summary["peak_allocated_gib"] - summary["start_allocated_gib"]
            )
            summary["peak_reserved_delta_gib"] = (
                summary["peak_reserved_gib"] - summary["start_reserved_gib"]
            )

        if not _is_world_process_zero():
            return control

        os.makedirs(args.output_dir, exist_ok=True)
        summary_path = os.path.join(args.output_dir, "train_benchmark_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
            f.write("\n")

        rank0_print(
            "[TRAIN_BENCHMARK] "
            f"method={summary['method']} "
            f"measured_steps={summary['measured_steps']} "
            f"elapsed_sec={summary['elapsed_sec']:.3f} "
            f"start_allocated_gib={summary['start_allocated_gib']:.3f} "
            f"peak_allocated_gib={summary['peak_allocated_gib']:.3f} "
            f"peak_allocated_delta_gib={summary['peak_allocated_delta_gib']:.3f} "
            f"peak_reserved_gib={summary['peak_reserved_gib']:.3f} "
            f"peak_reserved_delta_gib={summary['peak_reserved_delta_gib']:.3f} "
            f"summary_path={summary_path}"
        )
        return control

def train():
    global local_rank
    import wandb

    os.environ["WANDB_PROJECT"] = "sft"
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    assert not (
        training_args.lora_enable and training_args.gradient_checkpointing_kwargs
    ), "When using LoRA, the LLM should not be frozen. If you want to freeze the LLM, please disable LoRA."

    if not training_args.lora_enable:
        assert not training_args.vision_lora, (
            "Error: training_args.lora_enable is not enabled, but training_args.vision_lora is enabled."
        )
    else:
        if training_args.lora_namespan_exclude is not None:
            training_args.lora_namespan_exclude = ast.literal_eval(
                training_args.lora_namespan_exclude
            )
        else:
            training_args.lora_namespan_exclude = []

    local_rank = training_args.local_rank
    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )
    sft_method = getattr(training_args, "sft_method", "base").lower()

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        bnb_model_from_pretrained_args.update(
            dict(
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=training_args.bits == 4,
                    load_in_8bit=training_args.bits == 8,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    # bnb_4bit_use_double_quant=training_args.double_quant,
                    bnb_4bit_quant_type=training_args.quant_type,
                )
            )
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_id,
        torch_dtype=compute_dtype,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
        # attn_implementation="flash_attention_2"
        # if not training_args.disable_flash_attn2
        # else "sdpa",
        **bnb_model_from_pretrained_args,
    )

    try:
        from transformers.models.llama.modeling_llama import (
            ColumnParallelLinear, RowParallelLinear
        )
    except ImportError:
        ColumnParallelLinear = RowParallelLinear = Linear

    baseline_svd = {}  # maps key -> (U0, V0) kept on GPU
    baseline_block = {}  # maps key -> S_ref = U0^T W0 V0
    baseline_params = {}  # low-rank W0 on CPU; full-rank W0 on GPU for fair memory accounting

    topk = training_args.svd_reg_topk
    device = torch.cuda.current_device()
    baseline_dtype = torch.float16

    modules_dict = dict(model.named_modules())
    if training_args.svd_reg_coeff and training_args.svd_reg_coeff != 0.0:
        for name, p in tqdm(model.named_parameters(), desc="baseline"):
            if 'vision' in name or 'cross' in name or 'multi_modal' in name: continue
            if p.ndim != 2: continue
            # if not any(k in name for k in keep_kw): continue

            W = _gather_full_param(p)  # <- your helper; must return FULL tensor under ZeRO-3
            if W is None or W.numel() == 0: continue

            if W.ndim == 1:
                mod = modules_dict[name.rsplit(".", 1)[0]]
                if hasattr(mod, "out_features") and hasattr(mod, "in_features"):
                    W = W.view(mod.out_features, mod.in_features)
                else:
                    continue
            if W.ndim != 2: continue

            min_dim = min(W.shape)
            k = min(topk, min_dim)
            key = "module." + name  # or choose a single normalized scheme and stick to it
            if k == min_dim:
                # Full-rank case: keep W0 resident on GPU, like U0/V0/S_ref in low-rank mode.
                baseline_params[key] = W.detach().to(
                    device=device,
                    dtype=baseline_dtype,
                    non_blocking=True,
                ).contiguous()
                del W
                continue

            baseline_params[key] = W.detach().cpu()  # optional fallback for low-rank mode

            Wf = W.to(dtype=torch.float32, device=device, non_blocking=True)
            U, S, Vh = torch.linalg.svd(Wf, full_matrices=False)
            U_k = U[:, :k]
            V_k = Vh[:k, :].T
            Sref = (U_k.T @ (Wf @ V_k)).detach()
            U0 = U_k.detach().to(device=device, dtype=baseline_dtype, non_blocking=True).contiguous()
            V0 = V_k.detach().to(device=device, dtype=baseline_dtype, non_blocking=True).contiguous()
            S_ref = Sref.to(device=device, dtype=baseline_dtype, non_blocking=True).contiguous()

            baseline_svd[key] = (U0, V0)
            baseline_block[key] = S_ref

            del W, Wf, U, S, Vh, U0, V0, S_ref
            torch.cuda.empty_cache()

        torch.cuda.synchronize()
        torch.cuda.ipc_collect()
    print(f"[baseline] collected {baseline_svd.keys()} matrices")

    # baseline_svd = collect_baseline_svd(model, topk=training_args.svd_reg_topk)
    # import gc, ctypes
    torch.cuda.ipc_collect()
    from torch.cuda import memory
    # memory._free_cached_blocks()
    memory._free_mutex()
    torch.cuda.empty_cache()

    ref_model = None
    if sft_method in ("kl", "iw"):
        ref_model_id = model_args.ref_model_id or model_args.model_id
        ref_model = AutoModelForCausalLM.from_pretrained(
            ref_model_id,
            torch_dtype=compute_dtype,
            cache_dir=training_args.cache_dir,
            trust_remote_code=True,
            **bnb_model_from_pretrained_args,
        )
        if hasattr(ref_model.config, "use_cache"):
            ref_model.config.use_cache = False
    # I set a hidden size for temporary use. This is to use the deepspeed.
    # I will find a proper way later.
    # model.config.hidden_size = model.config.text_config.hidden_size
    # model.config.text_config.use_cache = False

    if training_args.bits in [4, 8]:
        model.config.torch_dtype = (
            torch.float32
            if training_args.fp16
            else (torch.bfloat16 if training_args.bf16 else torch.float32)
        )
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=training_args.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    if training_args.lora_enable:
        peft_config = LoraConfig(
            r=training_args.lora_rank,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_target_linear_names(
                model,
                lora_namespan_exclude=training_args.lora_namespan_exclude,
                num_lora_modules=training_args.num_lora_modules,
            ),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA to the model...")
        model = get_peft_model(model, peft_config)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_id,
        padding_side="right",
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    model.config.tokenizer_model_max_length = tokenizer.model_max_length
    model.config.tokenizer_padding_side = tokenizer.padding_side

    if not training_args.lora_enable:
        configure_llm(model, training_args)

    data_module = make_supervised_data_module(processor=tokenizer, data_args=data_args)
    trainer_kwargs = dict(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        baseline_svd=baseline_svd,
        baseline_block=baseline_block,
        baseline_params=baseline_params,
        svd_reg_coeff=training_args.svd_reg_coeff,
        svd_reg_topk=training_args.svd_reg_topk,
        **data_module,
    )

    if sft_method == "kl":
        trainer_cls = QwenKLLoraTrainer if training_args.lora_enable else QwenKLSFTTrainer
        if ref_model is None:
            raise ValueError("KL SFT requires a reference model; set --ref_model_id or keep default.")
        trainer_kwargs.update(ref_model=ref_model, kl_coeff=training_args.kl_coeff)
    elif sft_method == "dft":
        trainer_cls = QwenDFTLoraTrainer if training_args.lora_enable else QwenDFTTrainer
    elif sft_method == "iw":
        trainer_cls = QwenIwSFTLoraTrainer if training_args.lora_enable else QwenIwSFTTrainer
        if ref_model is None:
            raise ValueError("IW SFT requires a reference model; set --ref_model_id or keep default.")
        trainer_kwargs.update(
            ref_model=ref_model,
            iw_temperature=training_args.iw_temperature,
            iw_max_scale=training_args.iw_max_scale,
        )
    else:
        trainer_cls = QwenLoraTrainer if training_args.lora_enable else QwenTrainer
    trainer = trainer_cls(**trainer_kwargs)

    benchmark_method = sft_method
    if benchmark_method == "base" and training_args.svd_reg_coeff:
        benchmark_method = "svd"
    elif benchmark_method == "base" and training_args.lora_enable:
        benchmark_method = "lora"
    trainer.add_callback(TrainBenchmarkCallback(benchmark_method))

    resume_path = _select_resume_checkpoint(
        training_args.output_dir, getattr(training_args, "resume_from_checkpoint", None)
    )
    if resume_path:
        resume_step = _infer_resume_step(resume_path)
        rank0_print(f"[DEBUG] resume_path={resume_path} resume_step={resume_step}")
        if resume_step:
            trainer.state.global_step = resume_step
            trainer._globalstep_last_logged = resume_step
            trainer._globalstep_last_eval = getattr(trainer, "_globalstep_last_eval", resume_step)
            trainer._globalstep_last_save = getattr(trainer, "_globalstep_last_save", resume_step)
            trainer._globalstep_last_reported = getattr(trainer, "_globalstep_last_reported", resume_step)
        trainer.train(resume_from_checkpoint=resume_path)
    else:
        trainer.train()

    benchmark_only = os.getenv("SFT_BENCHMARK_ONLY", "0").strip().lower() in {"1", "true", "yes", "y"}
    benchmark_only = benchmark_only or bool(getattr(training_args, "sft_benchmark_only", False))
    if benchmark_only:
        rank0_print("[TRAIN_BENCHMARK] SFT_BENCHMARK_ONLY enabled; skipping final checkpoint save.")
        return

    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(model.named_parameters(), training_args.lora_bias)
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters(), require_grad_only=False
        )
        if local_rank == 0 or local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(
                non_lora_state_dict,
                os.path.join(training_args.output_dir, "non_lora_state_dict.bin"),
            )
    else:
        safe_save_model_for_hf_trainer(trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
