from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

# If you get rid of AutoProcessor, the code dosen't work.
from transformers import AutoProcessor, TrainingArguments

@dataclass
class ModelArguments:
    model_id: Optional[str] = field(default="meta-llama/Llama-3.2-11B-Vision-Instruct")
    ref_model_id: Optional[str] = field(
        default=None,
        metadata={"help": "Optional teacher/reference model. Defaults to model_id when not set."},
    )

@dataclass
class TrainingArguments(TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.98)
    adam_epsilon: float = field(default=1e-7)

    freeze_vision_tower: bool = field(default=False)
    freeze_llm: bool = field(default=False)
    tune_img_projector: bool = field(default=True)
    disable_flash_attn2: bool = field(default=True)
    overlap_comm: bool = field(default=False)

    svd_reg_coeff: float = field(
        default=0.0,
        metadata={"help": "weights SVD regularization coefficient λ；0 means closed"}
    )
    svd_reg_topk: int = field(
        default=4096,
        metadata={"help": "top k singular vector"}
    )
    svd_reg_layers: Optional[List[str]] = field(
        default_factory=list,
        metadata={
            "help": "regularization in terms of layers，\
                         eg. ['q_proj', 'k_proj', 'mlp']"
        },
    )

    max_seq_length: int = field(
        default=131072, # This is the default max_length for phi3-vision-128k-instruct
        metadata={
            "help":
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )

    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    sft_method: str = field(
        default="base",
        metadata={"help": "SFT objective: base, kl, dft, iw"}
    )
    kl_coeff: float = field(
        default=0.0,
        metadata={"help": "KL weight when using the KL-regularized SFT objective."}
    )
    iw_temperature: float = field(
        default=1.0,
        metadata={"help": "Temperature for IW-SFT importance weights."}
    )
    iw_max_scale: float = field(
        default=16.0,
        metadata={"help": "Upper bound for IW-SFT importance weights; set <=0 to disable clipping."}
    )
    sft_benchmark_only: bool = field(
        default=False,
        metadata={"help": "Stop after training benchmark metrics are emitted; skip final model/checkpoint save."},
    )
    lora_enable: bool = False
    vision_lora: bool = False
    lora_rank: int = 32
    lora_alpha: int = lora_rank * 2
    lora_dropout: float = 0.0
    lora_weight_path: str = ""
    lora_bias: str = "none"
    projector_lr: Optional[float] = None
    vision_lr: Optional[float] = None
    lora_namespan_exclude: str = field(default=None, metadata={"help": "List of namespan to exclude for LoRA"})
    num_lora_modules: int = -1



@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    lazy_preprocess: bool = False
    image_folder: Optional[str] = field(default=None)
    max_num_frames: int = 10
    use_cot: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Optional toggle for inserting a chain-of-thought system prompt. "
                "Pass a truthy value to enable."
            )
        },
    )
