import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    is_peft_available,
    WEIGHTS_NAME,
    TRAINING_ARGS_NAME,
    SAFE_WEIGHTS_NAME,
    TRAINER_STATE_NAME,
    PREFIX_CHECKPOINT_DIR,
    logger,
)
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

import safetensors
from peft import PeftModel
from typing import Optional
import numpy as np
from transformers.processing_utils import ProcessorMixin
from transformers.modeling_utils import PreTrainedModel
from peft import PeftModel
from training.train_utils import get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3
from tqdm import tqdm
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
import torch, ast, operator as op, numpy as np, csv, re
from torch.nn import Embedding, Linear
def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, "no ignore status")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

def _ece(confs, correct, n_bins=15):
    confs = np.asarray(confs, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    bins = np.linspace(0, 1, n_bins + 1)
    N = confs.shape[0]; e = 0.0
    for i in range(n_bins):
        m = (confs > bins[i]) & (confs <= (bins[i+1] if i < n_bins-1 else bins[i+1]))
        if m.sum() == 0:
            continue
        e += (m.sum()/N) * abs(correct[m].mean() - confs[m].mean())
    return float(e)


def _shift_logits_labels_and_mask(logits, labels, ignore_index: int = -100):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    mask = shift_labels.ne(ignore_index)
    return shift_logits, shift_labels, mask


def _token_log_probs(logits, targets):
    safe_targets = targets.clamp(min=0).unsqueeze(-1)
    logp = F.log_softmax(logits.float(), dim=-1)
    gathered = logp.gather(-1, safe_targets).squeeze(-1)
    return gathered


def _masked_mean(values: torch.Tensor, mask: torch.Tensor):
    mask_f = mask.float()
    denom = mask_f.sum()
    return (values * mask_f).sum() / (denom + 1e-12)


class _ReferenceModelMixin:
    def __init__(self, *args, ref_model=None, **kwargs):
        self.ref_model = ref_model
        self._ref_model_device = None
        if self.ref_model is not None:
            self.ref_model.eval()
            for p in self.ref_model.parameters():
                p.requires_grad_(False)
        super().__init__(*args, **kwargs)

    def _ensure_ref_device(self, device: torch.device):
        if self.ref_model is None:
            return
        if self._ref_model_device != device:
            self.ref_model.to(device)
            self._ref_model_device = device


class _KLLossMixin(_ReferenceModelMixin):
    def __init__(self, *args, kl_coeff: float = 0.0, **kwargs):
        self.kl_coeff = float(kl_coeff)
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        outputs = model(**inputs)
        base_loss = outputs["loss"]

        if self.ref_model is None or self.kl_coeff == 0.0 or "labels" not in inputs:
            return (base_loss, outputs) if return_outputs else base_loss

        shift_logits, shift_labels, mask = _shift_logits_labels_and_mask(outputs["logits"], inputs["labels"])
        self._ensure_ref_device(shift_logits.device)
        with torch.no_grad():
            ref_outputs = self.ref_model(
                input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask")
            )
            ref_logits = ref_outputs["logits"] if isinstance(ref_outputs, dict) else ref_outputs[0]
            ref_shift = ref_logits[..., :-1, :]
            ref_logp = F.log_softmax(ref_shift.float(), dim=-1)

        student_logp = F.log_softmax(shift_logits.float(), dim=-1)
        kl = (student_logp.exp() * (student_logp - ref_logp)).sum(dim=-1)
        kl_mean = _masked_mean(kl, mask)
        total = base_loss + float(self.kl_coeff) * kl_mean

        if return_outputs:
            outputs["kl"] = kl_mean.detach()
            outputs["loss"] = total
            return total, outputs
        return total


class _DFTLossMixin:
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        outputs = model(**inputs)
        if "labels" not in inputs:
            return (outputs["loss"], outputs) if return_outputs else outputs["loss"]

        shift_logits, shift_labels, mask = _shift_logits_labels_and_mask(outputs["logits"], inputs["labels"])
        vocab = shift_logits.size(-1)
        ce = F.cross_entropy(
            shift_logits.view(-1, vocab),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        ).view_as(shift_labels)

        with torch.no_grad():
            probs = F.softmax(shift_logits.float(), dim=-1)
            target_probs = probs.gather(-1, shift_labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)

        weighted = ce * target_probs.detach()
        loss = _masked_mean(weighted, mask)
        if return_outputs:
            outputs["weighted_loss"] = loss
            outputs["loss"] = loss
            return loss, outputs
        return loss


class _IwLossMixin(_ReferenceModelMixin):
    def __init__(self, *args, iw_temperature: float = 1.0, iw_max_scale: float = 16.0, **kwargs):
        self.iw_temperature = float(iw_temperature)
        self.iw_max_scale = float(iw_max_scale)
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        outputs = model(**inputs)
        base_loss = outputs["loss"]

        if self.ref_model is None or "labels" not in inputs:
            return (base_loss, outputs) if return_outputs else base_loss

        shift_logits, shift_labels, mask = _shift_logits_labels_and_mask(outputs["logits"], inputs["labels"])
        token_logp = _token_log_probs(shift_logits, shift_labels)
        nll = -token_logp

        self._ensure_ref_device(shift_logits.device)
        with torch.no_grad():
            ref_outputs = self.ref_model(
                input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask")
            )
            ref_logits = ref_outputs["logits"] if isinstance(ref_outputs, dict) else ref_outputs[0]
            ref_shift = ref_logits[..., :-1, :]
            ref_logp = _token_log_probs(ref_shift, shift_labels)

        mask_f = mask.float()
        counts = mask_f.sum(dim=-1).clamp(min=1.0)
        diff = ((token_logp - ref_logp) * mask_f).sum(dim=-1) / counts
        scale = torch.exp(self.iw_temperature * diff)
        if self.iw_max_scale and self.iw_max_scale > 0:
            scale = torch.clamp(scale, max=float(self.iw_max_scale))

        nll = nll * scale.unsqueeze(-1)
        loss = _masked_mean(nll, mask)
        if return_outputs:
            outputs["iw_scale"] = scale.detach()
            outputs["loss"] = loss
            return loss, outputs
        return loss


class LoraSVDCriterionMixin:
    """LoRA-specific SVD regularization helpers."""

    def _resolve_baseline_key(self, module_name: str, weight_owner=None) -> Optional[str]:
        baseline_map = getattr(self, "baseline_svd", {})
        if not baseline_map:
            baseline_map = getattr(self, "baseline_params", {})
        if not baseline_map:
            baseline_map = getattr(self, "baseline_block", {})
        if not baseline_map:
            return None

        def _canonical(name: str) -> str:
            n = name
            if n.startswith("module."):
                n = n[len("module."):]
            while n.startswith("base_model."):
                n = n[len("base_model."):]
            while "model.model." in n:
                n = n.replace("model.model.", "model.")
            return n

        base = module_name
        canon = _canonical(module_name)
        # Also try a version with a single leading "model." stripped
        if canon.startswith("model."):
            canon_no_model = canon[len("model."):]
        else:
            canon_no_model = canon

        candidates = {
            f"{base}.weight",
            f"{canon}.weight",
            f"{canon_no_model}.weight",
            f"module.{canon}.weight",
            f"module.{canon_no_model}.weight",
            f"module.{base}.weight",
        }

        for cand in candidates:
            if cand in baseline_map:
                return cand
        return None

    @staticmethod
    def _lora_effective_delta(module):
        try:
            from peft.tuners.lora import LoraLayer
        except Exception:
            return None

        if not isinstance(module, LoraLayer):
            return None

        # Resolve which adapters are active; PEFT may store str, list, or "all".
        aa = getattr(module, "active_adapter", None)
        if aa in (None, "all"):
            adapters = list(getattr(module, "lora_A", {}).keys())
        elif isinstance(aa, str):
            adapters = [aa]
        else:
            # assume iterable
            adapters = list(aa)

        deltas = []
        for adapter in adapters:
            if hasattr(module, "get_delta_weight"):
                delta = module.get_delta_weight(adapter)
            else:
                if adapter not in module.lora_A or adapter not in module.lora_B:
                    continue
                A = module.lora_A[adapter].weight
                B = module.lora_B[adapter].weight
                if A is None or B is None:
                    continue
                scaling = module.scaling[adapter] if isinstance(module.scaling, dict) else module.scaling
                delta = (B @ A) * scaling
                if getattr(module, "fan_in_fan_out", False):
                    delta = delta.T
            deltas.append(delta)

        if not deltas:
            return None
        total_delta = deltas[0]
        for extra in deltas[1:]:
            total_delta = total_delta + extra
        return total_delta

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        outputs = model(**inputs)
        task_loss = outputs["loss"]

        if self.svd_reg_coeff == 0.0 or (not self.baseline_svd and not self.baseline_params):
            return (task_loss, outputs) if return_outputs else task_loss

        reg_terms = []
        matched = skipped_missing_baseline = skipped_no_delta = skipped_shape = 0
        if not hasattr(self, "_svd_missing_samples"):
            self._svd_missing_samples = []
        if not hasattr(self, "_svd_matched_samples"):
            self._svd_matched_samples = []

        for module_name, module in model.named_modules():
            weight_owner = module
            if not (hasattr(weight_owner, "weight") or (
                    hasattr(weight_owner, "base_layer") and hasattr(weight_owner.base_layer, "weight"))):
                continue
            p = getattr(weight_owner, "weight", None)
            if p is None and hasattr(weight_owner, "base_layer"):
                p = getattr(weight_owner.base_layer, "weight", None)
            if p is None or p.ndim != 2:
                skipped_shape += 1
                continue
            key = self._resolve_baseline_key(module_name, getattr(weight_owner, "base_layer", weight_owner))
            if key is None:
                skipped_missing_baseline += 1
                self._svd_missing_samples.append(module_name)
                continue

            delta = self._lora_effective_delta(module)
            if delta is None:
                skipped_no_delta += 1
                continue

            min_dim = min(p.shape)
            if self.svd_reg_topk >= min_dim:
                if key not in self.baseline_params:
                    skipped_missing_baseline += 1
                    self._svd_missing_samples.append(module_name)
                    continue
                W0_cpu = self.baseline_params[key]
                if W0_cpu.ndim != 2 or W0_cpu.shape != tuple(p.shape):
                    skipped_shape += 1
                    continue
                dev = p.device
                W0 = W0_cpu.to(dev, dtype=torch.float32, non_blocking=True)
                W_now = p.to(dtype=torch.float32) + delta.to(dtype=torch.float32, device=dev, non_blocking=True)
                reg_terms.append((W_now - W0).pow(2).sum())
                matched += 1
                self._svd_matched_samples.append(module_name)
                continue

            if key not in self.baseline_block:
                skipped_missing_baseline += 1
                self._svd_missing_samples.append(module_name)
                continue

            U0_cpu, V0_cpu = self.baseline_svd[key]
            Sref_cpu = self.baseline_block[key]

            dev = p.device
            U = U0_cpu.to(dev, dtype=torch.float32, non_blocking=True)
            V = V0_cpu.to(dev, dtype=torch.float32, non_blocking=True)
            Sref = Sref_cpu.to(dev, dtype=torch.float32, non_blocking=True)

            Delta = delta.to(dtype=torch.float32, device=dev, non_blocking=True)
            S_now = U.T @ ((p.to(dtype=torch.float32)) @ V) + U.T @ (Delta @ V)

            num = (S_now - Sref).pow(2).sum()
            # den = Sref.pow(2).sum() + 1e-12
            reg_terms.append(num)
            matched += 1
            self._svd_matched_samples.append(module_name)

        reg_loss = (torch.stack(reg_terms).sum() if reg_terms else torch.zeros((), device=task_loss.device))
        total = task_loss + float(self.svd_reg_coeff) * reg_loss

        if getattr(self.accelerator, "is_main_process", True):
            step = int(getattr(self.state, "global_step", 0))
            print(f"[DEBUG] step={step} loss={task_loss.item():.4f} reg={float(reg_loss):.3e} "
                  f"total={float(total):.4f} coeff={self.svd_reg_coeff} matched={matched} "
                  f"missed_baseline={skipped_missing_baseline} no_delta={skipped_no_delta} "
                  f"bad_shape={skipped_shape} svd_topk={getattr(self, 'svd_reg_topk', None)}")

        if return_outputs:
            outputs["reg_loss"] = reg_loss
            return total, outputs
        return total

class LLamaVTrainer(Trainer):

    # def __init__(self, *args, processor: Optional[ProcessorMixin] = None,
    #              baseline_svd=None,
    #              baseline_params=None,
    #              svd_reg_coeff=0.1,
    #              svd_reg_topk=4096, **kwargs):
    #     super(LLamaVTrainer, self).__init__(*args, **kwargs)
    #     self.processor = processor
    #     self.baseline_svd = baseline_svd or {}
    #     self.baseline_params = baseline_params or {}
    #     self.svd_reg_coeff = svd_reg_coeff
    #     self.svd_reg_topk = svd_reg_topk
    #     if not hasattr(self, "_prev_W"):
    #         self._prev_W = {}
    def __init__(self, *args, processor: Optional[ProcessorMixin] = None,
                 baseline_svd=None,
                 baseline_block=None,
                 baseline_params=None,
                 svd_reg_coeff=0.1,
                 svd_reg_topk=4096, **kwargs):
        super().__init__(*args, **kwargs)
        self.processor = processor
        self._is_ood_eval = False

        # pad id to allow prompt-trimming
        self._pad_id = None
        if self.processor and self.processor.tokenizer.pad_token_id is not None:
            self._pad_id = self.processor.tokenizer.pad_token_id
        if self._pad_id is None:
            self._pad_id = getattr(self.model.config, "pad_token_id", None)
        if self._pad_id is None:
            self._pad_id = getattr(self.model.generation_config, "pad_token_id", None)
        if self._pad_id is None:
            self._pad_id = self.model.config.eos_token_id

        self.baseline_svd = baseline_svd or {}
        self.baseline_params = baseline_params or {}
        self.baseline_block = baseline_block or {}
        self.svd_reg_coeff = float(svd_reg_coeff)
        self.svd_reg_topk = int(svd_reg_topk)

    def _reset_calib_buffers(self):
        self._eval_seq_conf = []   # list[np.ndarray], shape [B]
        self._eval_seq_nll  = []   # list[np.ndarray], shape [B]
        self._eval_correct  = []   # list[np.ndarray], shape [B] (filled by compute_metrics)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        # reset buffers for this eval
        self._lbl_cov = [];
        self._trim_pos = []
        self._lbl_comp_digits = [];
        self._lbl_comp_alpha = []
        self._lbl_has_formula_key = [];
        self._lbl_has_equals = []

        metrics = super().evaluate(eval_dataset=eval_dataset,
                                   ignore_keys=ignore_keys,
                                   metric_key_prefix=metric_key_prefix)

        # aggregate
        import numpy as _np
        def _m(x): return float(_np.mean(x)) if len(x) > 0 else float("nan")

        metrics[f"{metric_key_prefix}_label_cov_mean"] = _m(self._lbl_cov)
        metrics[f"{metric_key_prefix}_trim_pos_mean"] = _m(self._trim_pos)
        metrics[f"{metric_key_prefix}_lbl_digits_share"] = _m(self._lbl_comp_digits)
        metrics[f"{metric_key_prefix}_lbl_alpha_share"] = _m(self._lbl_comp_alpha)
        metrics[f"{metric_key_prefix}_lbl_has_formula"] = _m(self._lbl_has_formula_key)
        metrics[f"{metric_key_prefix}_lbl_has_equals"] = _m(self._lbl_has_equals)

        if self.accelerator.is_main_process:
            step = int(self.state.global_step) if hasattr(self, "state") and self.state.global_step is not None else -1
            print(f"[{metric_key_prefix}] step={step} "
                  f"label_cov_mean={metrics[f'{metric_key_prefix}_label_cov_mean']:.4f} "
                  f"trim_pos_mean={metrics[f'{metric_key_prefix}_trim_pos_mean']:.4f} "
                  f"digits={metrics[f'{metric_key_prefix}_lbl_digits_share']:.3f} "
                  f"alpha={metrics[f'{metric_key_prefix}_lbl_alpha_share']:.3f} "
                  f"has_formula={metrics[f'{metric_key_prefix}_lbl_has_formula']:.2f} "
                  f"has_eq={metrics[f'{metric_key_prefix}_lbl_has_equals']:.2f}")

        # (optionally) also push via trainer.log so W&B can pick them up if enabled
        self.log(metrics)
        return metrics

    # ----------------- Public helper to run OOD eval -----------------
    def evaluate_ood(self, eval_dataset, metric_prefix="ood", face_rule="rule13"):
        prev_cm = self.compute_metrics
        try:
            self.compute_metrics = self._metric_ood if face_rule == "rule13" else self._make_24pt_metric(
                face_rule=face_rule, csv_name="./eval_debug_24pt_ood.csv"
            )
            return self.evaluate(eval_dataset=eval_dataset, metric_key_prefix=metric_prefix)  # <—
        finally:
            self.compute_metrics = prev_cm

    # ----------------- In-class 24pt metric factory -----------------
    def _make_24pt_metric_both(self, csv_prefix="./eval_debug_24pt"):
        # local helpers (same parsing/eval you already use)
        import numpy as _np, csv, re, ast, operator as op
        _ALLOWED_BINOPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv}
        _ALLOWED_UNOPS = {ast.UAdd: lambda x: x, ast.USub: op.neg}

        FACE_MAPS = {
            "rule10": {"j": 10, "q": 10, "k": 10, "a": 1},
            "rule13": {"j": 11, "q": 12, "k": 13, "a": 1},
        }

        def _safe_eval(expr):
            try:
                node = ast.parse(expr, mode="eval")
            except Exception:
                return False, float("nan"), []
            used = []

            def ev(n):
                if isinstance(n, ast.Expression): return ev(n.body)
                if isinstance(n, ast.Num):        v = float(n.n); used.append(int(round(v))); return v
                if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
                    v = float(n.value);
                    used.append(int(round(v)));
                    return v
                if isinstance(n, ast.UnaryOp) and type(n.op) in _ALLOWED_UNOPS:
                    return _ALLOWED_UNOPS[type(n.op)](ev(n.operand))
                if isinstance(n, ast.BinOp) and type(n.op) in _ALLOWED_BINOPS:
                    a = ev(n.left);
                    b = ev(n.right)
                    if isinstance(n.op, ast.Div) and abs(b) < 1e-12: raise ZeroDivisionError
                    return _ALLOWED_BINOPS[type(n.op)](a, b)
                raise ValueError

            try:
                val = ev(node)
            except Exception:
                return False, float("nan"), []
            return True, float(val), used

        def _parse_cards_with_rule(text, face_map):
            def _norm_rank(s):
                s = s.strip().lower().strip("'\"")
                return int(face_map.get(s, s))

            m = re.search(r"(?i)cards?\s*[:：]\s*\[([^\]]+)\]", text)
            if not m: return None
            toks = re.findall(r"[A-Za-z]+|\d+", m.group(1))
            return [_norm_rank(t) for t in toks] if toks else None

        def _mset_eq(a, b):
            from collections import Counter
            return Counter(a) == Counter(b)

        def _extract_formula_candidates(text: str):
            cands = []
            # JSON-style value
            for s in re.findall(r'"formula"\s*:\s*[\'"]([^\'"\n\r}]+)[\'"]', text, flags=re.IGNORECASE):
                s = s.strip()
                if s: cands.append(s)
            # loose lines
            for line in text.splitlines():
                line = line.strip()
                if any(ch in line for ch in "+-*/()=") and len(line) >= 3:
                    if re.search(r"[A-Za-z]", line) and not re.match(r"^[\s(]*\d", line):
                        continue
                    cands.append(line)
            # de-dup
            out, seen = [], set()
            for s in cands:
                s = s.strip()
                if s and s not in seen:
                    seen.add(s);
                    out.append(s)
            return out

        def _eval_formula_or_equality(s):
            s = s.strip()
            if "=" in s:
                left, right = s.split("=", 1)
                left, right = left.strip(), right.strip()
                okL, vL, numsL = _safe_eval(left)
                okR, vR, numsR = _safe_eval(right)
                if not (okL and okR): return False, float("nan"), []
                if abs(vL - vR) > 1e-6: return False, float("nan"), []
                # Prefer counting numbers from the side that is NOT literally 24
                if abs(vR - 24.0) < 1e-6:  # constructed expr on left
                    return True, vL, numsL
                if abs(vL - 24.0) < 1e-6:  # constructed expr on right
                    return True, vR, numsR
                # equal but not 24; caller will reject on the 24-check
                return True, vL, numsL
            else:
                return _safe_eval(s)

        tok = self.processor.tokenizer if self.processor is not None else self.tokenizer

        def _metric(eval_pred):
            preds_obj = eval_pred.predictions
            if isinstance(preds_obj, (list, tuple)) and len(preds_obj) == 1:
                preds_obj = preds_obj[0]
            if isinstance(preds_obj, torch.Tensor):
                preds_np = preds_obj.detach().cpu().numpy()
            else:
                preds_np = _np.asarray(preds_obj)
            if preds_np.ndim == 1: preds_np = preds_np[None, :]

            # decode only the generated tail using the prompt lengths we stashed
            inputs = getattr(eval_pred, "inputs", None)
            if inputs is None or "__gen_prompt_len" not in inputs:
                # fallback: decode whole seq (should rarely happen)
                tails = tok.batch_decode(preds_np, skip_special_tokens=True)
            else:
                lens = inputs["__gen_prompt_len"]
                lens = lens.detach().cpu().numpy() if hasattr(lens, "device") else _np.asarray(lens)
                tails = []
                for i in range(preds_np.shape[0]):
                    start = int(lens[i]);
                    seq_ids = preds_np[i]
                    tail_ids = seq_ids[start:] if start < seq_ids.shape[0] else seq_ids[-0:]
                    tails.append(tok.decode(tail_ids.tolist(), skip_special_tokens=True))

            # decode prompts (for cards)
            prompt_texts = None
            if inputs is not None and "input_ids" in inputs:
                inp_ids = inputs["input_ids"]
                prompt_texts = tok.batch_decode(inp_ids, skip_special_tokens=True) if hasattr(inp_ids, "device") \
                    else tok.batch_decode(inp_ids.tolist(), skip_special_tokens=True)

            ok_id = []
            ok_ood = []
            rows = []
            for i, tail in enumerate(tails):
                prompt_text = prompt_texts[i] if prompt_texts is not None else ""
                cands = _extract_formula_candidates(tail)
                if self.accelerator.is_main_process and i < 8:
                    # print(f"[dbg] tail[{i}] len={len(tail)} head={repr(tail[:256])}")
                    print(f"[dbg] tail[{i}] len={len(tail)} head={repr(tail)}")
                    print(f"[dbg] cands[{i}]: {cands[:3]}")

                def score_for(face_map):
                    cards = _parse_cards_with_rule(prompt_text, face_map)
                    if not cands:  return 0, "no-formula-in-gen", "", [], ""
                    last_reason = "no-valid-formula"
                    for cand in reversed(cands):
                        safe, v, nums = _eval_formula_or_equality(cand)
                        if not safe:            last_reason = "unsafe-eval"; continue
                        if cards is None:       last_reason = "no-cards-in-prompt"; continue
                        if not _mset_eq([int(x) for x in nums], [int(x) for x in cards]):
                            last_reason = f"wrong-multiset used={nums} cards={cards}";
                            continue
                        if abs(v - 24.0) > 1e-6:    last_reason = f"value!=24({v})"; continue
                        return 1, "ok", cand, nums, f"{v:.6f}"
                    return 0, last_reason, "", [], ""

                id_hit, id_reason, id_c, id_nums, id_val = score_for(FACE_MAPS["rule10"])
                ood_hit, ood_reason, ood_c, ood_nums, ood_val = score_for(FACE_MAPS["rule13"])

                ok_id.append(id_hit);
                ok_ood.append(ood_hit)

                if self.accelerator.is_main_process and i < 20:
                    print(f"[both] idx={i} ID={'OK' if id_hit else id_reason} | "
                          f"OOD={'OK' if ood_hit else ood_reason}")

                rows.append([i, id_hit, ood_hit, id_reason, ood_reason, id_c or ood_c,
                             id_val or ood_val, str(id_nums or ood_nums), tail[:200].replace("\n", "\\n")])

            # optional CSV per eval call (named by global step)
            if self.accelerator.is_main_process:
                step = int(self.state.global_step) if getattr(self.state, "global_step", None) is not None else 0
                csv_path = f"{csv_prefix}_both_step{step}.csv"
                try:
                    with open(csv_path, "w", newline="") as f:
                        w = csv.writer(f)
                        w.writerow(["idx", "id_ok", "ood_ok", "id_reason", "ood_reason",
                                    "chosen_formula", "value", "used_nums", "gen_head"])
                        w.writerows(rows)
                    print(f"[both] wrote {len(rows)} rows -> {csv_path}")
                except Exception as e:
                    print("[both] csv write failed:", e)

            return {
                "eval_id_accuracy": float(_np.mean(ok_id)) if len(ok_id) > 0 else 0.0,
                "eval_ood_accuracy": float(_np.mean(ok_ood)) if len(ok_ood) > 0 else 0.0,
            }

        return _metric

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        """
        Block-drift regularization:
          L = task_loss + λ * sum_layers || U0^T W V0 - U0^T W0 V0 ||_F^2 / (||U0^T W0 V0||_F^2 + eps)

        • Penalizes changes INSIDE the baseline top-k singular subspace
        • Does NOT penalize updates in the orthogonal complement (tail)
        • Zero at initialization by construction
        """
        outputs = model(**inputs)
        task_loss = outputs["loss"]

        # short-circuit if no regularization configured
        if getattr(self, "svd_reg_coeff", 0.0) == 0.0 or (
                not getattr(self, "baseline_svd", {}) and not getattr(self, "baseline_params", {})):
            return (task_loss, outputs) if return_outputs else task_loss

        def _norm_key(n: str) -> str:
            for pref in ("module.", "base_model.model."):
                if n.startswith(pref): n = n[len(pref):]
            return n

        reg_terms = []
        matched = 0
        for name, p in model.named_parameters():
            # only 2D weight matrices
            if p.ndim != 2:
                continue

            # find matching baseline key
            key = name
            if key not in self.baseline_svd and key not in self.baseline_params:
                nk = _norm_key(name)
                if nk in self.baseline_svd or nk in self.baseline_params:
                    key = nk
                elif ("module." + nk) in self.baseline_svd or ("module." + nk) in self.baseline_params:
                    key = "module." + nk
                else:
                    continue

            min_dim = min(p.shape)
            if self.svd_reg_topk >= min_dim:
                if not hasattr(self, "baseline_params") or key not in self.baseline_params:
                    continue
                W0_base = self.baseline_params[key]
                if W0_base.ndim != 2 or W0_base.shape != tuple(p.shape):
                    continue

                def _full_drift_term(W_tensor: torch.Tensor) -> torch.Tensor:
                    if W_tensor.device != W0_base.device:
                        raise RuntimeError(f"Full-rank SVD baseline for {key} is on {W0_base.device} but weight is on {W_tensor.device}")
                    W = W_tensor.to(dtype=W0_base.dtype)
                    return (W - W0_base).pow(2).sum()

                # ZeRO-3 safe: gather before reading p
                if hasattr(p, "ds_id"):
                    from deepspeed import zero
                    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
                    with zero.GatheredParameters([p], modifier_rank=None):
                        if p.ds_status != ZeroParamStatus.NOT_AVAILABLE:
                            reg_terms.append(_full_drift_term(p))
                            matched += 1
                else:
                    reg_terms.append(_full_drift_term(p))
                    matched += 1
                continue

            if key not in self.baseline_svd:
                continue

            # we need (U0, V0) and a reference block S_ref
            U0_base, V0_base = self.baseline_svd[key]
            # prefer precomputed baseline_block if available; otherwise build from baseline_params
            if hasattr(self, "baseline_block") and key in self.baseline_block:
                Sref_base = self.baseline_block[key]
            else:
                if not hasattr(self, "baseline_params") or key not in self.baseline_params:
                    # cannot form S_ref — skip this layer
                    continue
                # lazily build S_ref on first use, cache it to avoid recomputation
                W0_cpu = self.baseline_params[key]
                # ensure matrix shape matches
                if W0_cpu.ndim != 2 or W0_cpu.shape != (U0_base.shape[0], V0_base.shape[0]):
                    continue
                with torch.no_grad():
                    dev = U0_base.device
                    U = U0_base
                    V = V0_base
                    W0 = W0_cpu.to(dev, dtype=U.dtype, non_blocking=True)  # (m,n)
                    Sref = (U.T @ (W0 @ V)).detach()
                    if not hasattr(self, "baseline_block"):
                        self.baseline_block = {}
                    self.baseline_block[key] = Sref
                    Sref_base = Sref
                    # free temps asap
                    del W0, Sref

            def _block_drift_term(W_tensor: torch.Tensor) -> torch.Tensor:
                dev = U0_base.device
                if W_tensor.device != dev:
                    raise RuntimeError(f"SVD baseline for {key} is on {dev} but weight is on {W_tensor.device}")
                W = W_tensor.to(dtype=U0_base.dtype)
                S_now = (U0_base.T @ (W @ V0_base))

                num = (S_now - Sref_base).pow(2).sum()
                # den = Sref_base.pow(2).sum() + 1e-12
                # return num / den
                return num

            # ZeRO-3 safe: gather before reading p
            if hasattr(p, "ds_id"):
                from deepspeed import zero
                from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
                with zero.GatheredParameters([p], modifier_rank=None):
                    if p.ds_status != ZeroParamStatus.NOT_AVAILABLE:
                        reg_terms.append(_block_drift_term(p))
                        matched += 1
            else:
                reg_terms.append(_block_drift_term(p))
                matched += 1

        reg_loss = (torch.stack(reg_terms).sum()
                    if reg_terms else torch.zeros((), device=task_loss.device))
        total = task_loss + float(self.svd_reg_coeff) * reg_loss

        # clear, scientific-notation debug every ~50 steps
        if self.accelerator.is_main_process and (int(getattr(self.state, "global_step", 0)) % 1 == 0):
            print(f"[DEBUG] loss={task_loss.item():.4f}  reg={float(reg_loss):.3e}  total_loss={total}  "
                  f"coeff={self.svd_reg_coeff}  matched={matched}  "
                  f"svd_reg_topk={getattr(self, 'svd_reg_topk', None)}")

        if return_outputs:
            outputs["reg_loss"] = reg_loss
            return total, outputs
        return total

    def create_optimizer(self):
        """
        Setup the optimizer.
        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            lr_mapper = {}
            if self.args.projector_lr is not None:
                lr_mapper["multi_modal_projector"] = self.args.projector_lr
            if self.args.vision_lr is not None:
                lr_mapper["vision_model"] = self.args.vision_lr
            if len(lr_mapper) > 0:
                special_lr_parameters = [name for name, _ in opt_model.named_parameters() if any(module_keyword in name for module_keyword in lr_mapper)]
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in special_lr_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in special_lr_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                ]
                for module_keyword, lr in lr_mapper.items():
                    module_parameters = [name for name, _ in opt_model.named_parameters() if module_keyword in name]
                    optimizer_grouped_parameters.extend(
                        [
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in module_parameters and p.requires_grad)],
                                "weight_decay": self.args.weight_decay,
                                "lr": lr,
                            },
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in module_parameters and p.requires_grad)],
                                "weight_decay": 0.0,
                                "lr": lr,
                            },
                        ]
                    )
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None):
        if self.args.lora_enable:
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            if self.hp_search_backend is None and trial is None:
                self.store_flos()

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            self.save_model(output_dir, _internal_call=True)

            non_lora_weights = get_peft_state_non_lora_maybe_zero_3(self.model.named_parameters(), require_grad_only=False)
            torch.save(non_lora_weights, os.path.join(output_dir, "non_lora_state_dict.bin"))

            if not self.args.save_only_model:
                # Save optimizer and scheduler
                self._save_optimizer_and_scheduler(output_dir)
                # Save RNG state
                self._save_rng_state(output_dir)

            # Determine the new best metric / best model checkpoint
            if metrics is not None and self.args.metric_for_best_model is not None:
                metric_to_check = self.args.metric_for_best_model
                if not metric_to_check.startswith("eval_"):
                    metric_to_check = f"eval_{metric_to_check}"
                metric_value = metrics[metric_to_check]

                operator = np.greater if self.args.greater_is_better else np.less
                if (
                    self.state.best_metric is None
                    or self.state.best_model_checkpoint is None
                    or operator(metric_value, self.state.best_metric)
                ):
                    self.state.best_metric = metric_value
                    self.state.best_model_checkpoint = output_dir

            # Save the Trainer state
            if self.args.should_save:
                # Update the `TrainerControl` state to where we are currently
                self.state.stateful_callbacks["TrainerControl"] = self.control.state()
                self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

            if self.args.push_to_hub:
                self._push_from_checkpoint(output_dir)

            # Maybe delete some older checkpoints.
            if self.args.should_save:
                # Solely rely on numerical checkpoint id for rotation.
                # mtime is not reliable especially on some fuse fs in cloud environments.
                self._rotate_checkpoints(use_mtime=False, output_dir=run_dir)

        else:
            super(LLamaVTrainer, self)._save_checkpoint(model, trial)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
            # If we are executing this function, we are the process zero, so we don't check for that.
            output_dir = output_dir if output_dir is not None else self.args.output_dir
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Saving model checkpoint to {output_dir}")

            supported_classes = (PreTrainedModel,) if not is_peft_available() else (PreTrainedModel, PeftModel)
            # Save a trained model and configuration using `save_pretrained()`.
            # They can then be reloaded using `from_pretrained()`
            if not isinstance(self.model, supported_classes):
                if state_dict is None:
                    state_dict = self.model.state_dict()

                if isinstance(self.accelerator.unwrap_model(self.model), supported_classes):
                    self.accelerator.unwrap_model(self.model).save_pretrained(
                        output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
                    )
                else:
                    logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                    if self.args.save_safetensors:
                        safetensors.torch.save_file(
                            state_dict, os.path.join(output_dir, SAFE_WEIGHTS_NAME), metadata={"format": "pt"}
                        )
                    else:
                        torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
            else:
                state_dict = {k:v for k, v in state_dict.items() if "wte" not in k}
                self.model.save_pretrained(
                    output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
                )

            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(output_dir)

            if self.processor is not None:
                self.processor.save_pretrained(output_dir)

            # Good practice: save your training arguments together with the trained model
            torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

class QwenTrainer(Trainer):

    def __init__(self, *args, processor: Optional[ProcessorMixin] = None,
                 baseline_svd=None,
                 baseline_block=None,
                 baseline_params=None,
                 svd_reg_coeff=0.1,
                 svd_reg_topk=4096, **kwargs):
        super().__init__(*args, **kwargs)
        self.processor = processor
        self._is_ood_eval = False

        # pad id to allow prompt-trimming
        self._pad_id = None
        if self.processor and self.processor.tokenizer.pad_token_id is not None:
            self._pad_id = self.processor.tokenizer.pad_token_id
        if self._pad_id is None:
            self._pad_id = getattr(self.model.config, "pad_token_id", None)
        if self._pad_id is None:
            self._pad_id = getattr(self.model.generation_config, "pad_token_id", None)
        if self._pad_id is None:
            self._pad_id = self.model.config.eos_token_id

        self.baseline_svd = baseline_svd or {}
        self.baseline_params = baseline_params or {}
        self.baseline_block = baseline_block or {}
        self.svd_reg_coeff = float(svd_reg_coeff)
        self.svd_reg_topk = int(svd_reg_topk)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        """
        Block-drift regularization:
          L = task_loss + λ * sum_layers || U0^T W V0 - U0^T W0 V0 ||_F^2 / (||U0^T W0 V0||_F^2 + eps)

        • Penalizes changes INSIDE the baseline top-k singular subspace
        • Does NOT penalize updates in the orthogonal complement (tail)
        • Zero at initialization by construction
        """
        outputs = model(**inputs)
        task_loss = outputs["loss"]

        # short-circuit if no regularization configured
        if getattr(self, "svd_reg_coeff", 0.0) == 0.0 or (
                not getattr(self, "baseline_svd", {}) and not getattr(self, "baseline_params", {})):
            return (task_loss, outputs) if return_outputs else task_loss

        def _norm_key(n: str) -> str:
            for pref in ("module.", "base_model.model."):
                if n.startswith(pref): n = n[len(pref):]
            return n

        reg_terms = []
        matched = 0
        for name, p in model.named_parameters():
            # only 2D weight matrices
            if p.ndim != 2:
                continue

            # find matching baseline key
            key = name
            if key not in self.baseline_svd and key not in self.baseline_params:
                nk = _norm_key(name)
                if nk in self.baseline_svd or nk in self.baseline_params:
                    key = nk
                elif ("module." + nk) in self.baseline_svd or ("module." + nk) in self.baseline_params:
                    key = "module." + nk
                else:
                    continue

            min_dim = min(p.shape)
            if self.svd_reg_topk >= min_dim:
                if not hasattr(self, "baseline_params") or key not in self.baseline_params:
                    continue
                W0_base = self.baseline_params[key]
                if W0_base.ndim != 2 or W0_base.shape != tuple(p.shape):
                    continue

                def _full_drift_term(W_tensor: torch.Tensor) -> torch.Tensor:
                    if W_tensor.device != W0_base.device:
                        raise RuntimeError(f"Full-rank SVD baseline for {key} is on {W0_base.device} but weight is on {W_tensor.device}")
                    W = W_tensor.to(dtype=W0_base.dtype)
                    return (W - W0_base).pow(2).sum()

                # ZeRO-3 safe: gather before reading p
                if hasattr(p, "ds_id"):
                    from deepspeed import zero
                    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
                    with zero.GatheredParameters([p], modifier_rank=None):
                        if p.ds_status != ZeroParamStatus.NOT_AVAILABLE:
                            reg_terms.append(_full_drift_term(p))
                            matched += 1
                else:
                    reg_terms.append(_full_drift_term(p))
                    matched += 1
                continue

            if key not in self.baseline_svd:
                continue

            # we need (U0, V0) and a reference block S_ref
            U0_base, V0_base = self.baseline_svd[key]
            # prefer precomputed baseline_block if available; otherwise build from baseline_params
            if hasattr(self, "baseline_block") and key in self.baseline_block:
                Sref_base = self.baseline_block[key]
            else:
                if not hasattr(self, "baseline_params") or key not in self.baseline_params:
                    # cannot form S_ref — skip this layer
                    continue
                # lazily build S_ref on first use, cache it to avoid recomputation
                W0_cpu = self.baseline_params[key]
                # ensure matrix shape matches
                if W0_cpu.ndim != 2 or W0_cpu.shape != (U0_base.shape[0], V0_base.shape[0]):
                    continue
                with torch.no_grad():
                    dev = U0_base.device
                    U = U0_base
                    V = V0_base
                    W0 = W0_cpu.to(dev, dtype=U.dtype, non_blocking=True)  # (m,n)
                    Sref = (U.T @ (W0 @ V)).detach()
                    if not hasattr(self, "baseline_block"):
                        self.baseline_block = {}
                    self.baseline_block[key] = Sref
                    Sref_base = Sref
                    # free temps asap
                    del W0, Sref

            def _block_drift_term(W_tensor: torch.Tensor) -> torch.Tensor:
                dev = U0_base.device
                if W_tensor.device != dev:
                    raise RuntimeError(f"SVD baseline for {key} is on {dev} but weight is on {W_tensor.device}")
                W = W_tensor.to(dtype=U0_base.dtype)
                S_now = (U0_base.T @ (W @ V0_base))

                num = (S_now - Sref_base).pow(2).sum()
                # den = Sref_base.pow(2).sum() + 1e-12
                # return num / den
                return num

            # ZeRO-3 safe: gather before reading p
            if hasattr(p, "ds_id"):
                from deepspeed import zero
                from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
                with zero.GatheredParameters([p], modifier_rank=None):
                    if p.ds_status != ZeroParamStatus.NOT_AVAILABLE:
                        reg_terms.append(_block_drift_term(p))
                        matched += 1
            else:
                reg_terms.append(_block_drift_term(p))
                matched += 1

        reg_loss = (torch.stack(reg_terms).sum()
                    if reg_terms else torch.zeros((), device=task_loss.device))
        total = task_loss + float(self.svd_reg_coeff) * reg_loss

        # clear, scientific-notation debug every ~50 steps
        if self.accelerator.is_main_process and (int(getattr(self.state, "global_step", 0)) % 1 == 0):
            print(f"[DEBUG] loss={task_loss.item():.4f}  reg={float(reg_loss):.3e}  total_loss={total}  "
                  f"coeff={self.svd_reg_coeff}  matched={matched}  "
                  f"svd_reg_topk={getattr(self, 'svd_reg_topk', None)}")

        if return_outputs:
            outputs["reg_loss"] = reg_loss
            return total, outputs
        return total

    def create_optimizer_bak(self):
        """
        Setup the optimizer.
        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            lr_mapper = {}
            if self.args.projector_lr is not None:
                lr_mapper["multi_modal_projector"] = self.args.projector_lr
            if self.args.vision_lr is not None:
                lr_mapper["vision_model"] = self.args.vision_lr
            if len(lr_mapper) > 0:
                special_lr_parameters = [name for name, _ in opt_model.named_parameters() if
                                         any(module_keyword in name for module_keyword in lr_mapper)]
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if
                                   (n in decay_parameters and n not in special_lr_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (
                                    n not in decay_parameters and n not in special_lr_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                ]
                for module_keyword, lr in lr_mapper.items():
                    module_parameters = [name for name, _ in opt_model.named_parameters() if module_keyword in name]
                    optimizer_grouped_parameters.extend(
                        [
                            {
                                "params": [p for n, p in opt_model.named_parameters() if
                                           (n in decay_parameters and n in module_parameters and p.requires_grad)],
                                "weight_decay": self.args.weight_decay,
                                "lr": lr,
                            },
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (
                                            n not in decay_parameters and n in module_parameters and p.requires_grad)],
                                "weight_decay": 0.0,
                                "lr": lr,
                            },
                        ]
                    )
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if
                                   (n in decay_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if
                                   (n not in decay_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped / 2 ** 20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped / 2 ** 20}M params")

        return self.optimizer

    def _save_checkpoint_bak(self, model, trial, metrics=None):
        if self.args.lora_enable:
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            if self.hp_search_backend is None and trial is None:
                self.store_flos()

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            self.save_model(output_dir, _internal_call=True)

            non_lora_weights = get_peft_state_non_lora_maybe_zero_3(self.model.named_parameters(),
                                                                    require_grad_only=False)
            torch.save(non_lora_weights, os.path.join(output_dir, "non_lora_state_dict.bin"))

            if not self.args.save_only_model:
                # Save optimizer and scheduler
                self._save_optimizer_and_scheduler(output_dir)
                # Save RNG state
                self._save_rng_state(output_dir)

            # Determine the new best metric / best model checkpoint
            if metrics is not None and self.args.metric_for_best_model is not None:
                metric_to_check = self.args.metric_for_best_model
                if not metric_to_check.startswith("eval_"):
                    metric_to_check = f"eval_{metric_to_check}"
                metric_value = metrics[metric_to_check]

                operator = np.greater if self.args.greater_is_better else np.less
                if (
                        self.state.best_metric is None
                        or self.state.best_model_checkpoint is None
                        or operator(metric_value, self.state.best_metric)
                ):
                    self.state.best_metric = metric_value
                    self.state.best_model_checkpoint = output_dir

            # Save the Trainer state
            if self.args.should_save:
                # Update the `TrainerControl` state to where we are currently
                self.state.stateful_callbacks["TrainerControl"] = self.control.state()
                self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

            if self.args.push_to_hub:
                self._push_from_checkpoint(output_dir)

            # Maybe delete some older checkpoints.
            if self.args.should_save:
                # Solely rely on numerical checkpoint id for rotation.
                # mtime is not reliable especially on some fuse fs in cloud environments.
                self._rotate_checkpoints(use_mtime=False, output_dir=run_dir)

        else:
            super(QwenTrainer, self)._save_checkpoint(model, trial)

    def _save_bak(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        supported_classes = (PreTrainedModel,) if not is_peft_available() else (PreTrainedModel, PeftModel)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, supported_classes):
            if state_dict is None:
                state_dict = self.model.state_dict()

            if isinstance(self.accelerator.unwrap_model(self.model), supported_classes):
                self.accelerator.unwrap_model(self.model).save_pretrained(
                    output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
                )
            else:
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if self.args.save_safetensors:
                    safetensors.torch.save_file(
                        state_dict, os.path.join(output_dir, SAFE_WEIGHTS_NAME), metadata={"format": "pt"}
                    )
                else:
                    torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            state_dict = {k: v for k, v in state_dict.items() if "tok_embedding" not in k}
            self.model.save_pretrained(
                output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
            )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        if self.processor is not None:
            self.processor.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))


class LLamaVLoraTrainer(LoraSVDCriterionMixin, LLamaVTrainer):
    """LoRA variant of LLamaVTrainer with delta-based SVD regularization."""
    pass


class QwenLoraTrainer(LoraSVDCriterionMixin, QwenTrainer):
    """LoRA variant of QwenTrainer with delta-based SVD regularization."""
    pass


class LLamaKLSFTTrainer(_KLLossMixin, LLamaVTrainer):
    """LLama trainer with KL regularization against a frozen reference model."""
    pass


class LLamaKLLoraTrainer(_KLLossMixin, LLamaVLoraTrainer):
    """LoRA variant of KL-regularized LLama trainer."""
    pass


class LLamaDFTTrainer(_DFTLossMixin, LLamaVTrainer):
    """LLama trainer with probability-weighted token loss (DFT-style)."""
    pass


class LLamaDFTLoraTrainer(_DFTLossMixin, LLamaVLoraTrainer):
    """LoRA variant of probability-weighted token loss."""
    pass


class LLamaIwSFTTrainer(_IwLossMixin, LLamaVTrainer):
    """LLama trainer with importance-weighted SFT objective."""
    pass


class LLamaIwSFTLoraTrainer(_IwLossMixin, LLamaVLoraTrainer):
    """LoRA variant of importance-weighted SFT objective."""
    pass


class QwenKLSFTTrainer(_KLLossMixin, QwenTrainer):
    """Qwen trainer with KL regularization against a frozen reference model."""
    pass


class QwenKLLoraTrainer(_KLLossMixin, QwenLoraTrainer):
    """LoRA variant of KL-regularized Qwen trainer."""
    pass


class QwenDFTTrainer(_DFTLossMixin, QwenTrainer):
    """Qwen trainer with probability-weighted token loss (DFT-style)."""
    pass


class QwenDFTLoraTrainer(_DFTLossMixin, QwenLoraTrainer):
    """LoRA variant of probability-weighted token loss for Qwen."""
    pass


class QwenIwSFTTrainer(_IwLossMixin, QwenTrainer):
    """Qwen trainer with importance-weighted SFT objective."""
    pass


class QwenIwSFTLoraTrainer(_IwLossMixin, QwenLoraTrainer):
    """LoRA variant of importance-weighted SFT for Qwen."""
    pass
