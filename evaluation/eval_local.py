import argparse
import json
import os
import random
import re
import time
import multiprocessing as mp
from numbers import Integral
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluation.eval.openmathinst_utils import extract_answer, process_results
from evaluation.eval.objudge import OBJudge
from evaluation.metrics.ifeval_checker import (
    evaluate_ifeval_instructions,
    text_answer_matches,
)
from evaluation.prompt_utils import enforce_boxed_system_prompt

TEXT_DATASETS = {"truthfulqa"}
from evaluation.data.io_utils import load_eval_dataframe


def generate_batch(model, tokenizer, prompts, max_new_tokens=2048, temperature=0.7, top_p=0.95):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    input_lengths = inputs["attention_mask"].sum(dim=1).tolist()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            # eos_token_id=None,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
        )
    sequences = outputs.sequences
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    total_lengths = (sequences != pad_id).sum(dim=1).tolist()
    new_lengths = [tot - inp for tot, inp in zip(total_lengths, input_lengths)]
    texts = tokenizer.batch_decode(sequences, skip_special_tokens=True)
    return texts, input_lengths, total_lengths, new_lengths


def generate_batch_vllm(llm, tokenizer, prompts, max_new_tokens=2048, temperature=0.7, top_p=0.95, seed=None):
    # try:
    from vllm import SamplingParams
    # except Exception as exc:
    #     raise RuntimeError("vLLM backend requested but vllm is not installed") from exc

    params_kwargs = {
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_new_tokens,
        "n": 1,
    }
    if seed is not None:
        params_kwargs["seed"] = int(seed)
    sampling_params = SamplingParams(**params_kwargs)

    outputs = llm.generate(prompts, sampling_params)
    texts = []
    input_lengths = []
    total_lengths = []
    new_lengths = []
    for prompt, out in zip(prompts, outputs):
        completion = out.outputs[0].text if out.outputs else ""
        full = f"{prompt}{completion}"
        texts.append(full)

        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        input_len = len(prompt_ids)
        if out.outputs and getattr(out.outputs[0], "token_ids", None) is not None:
            new_len = len(out.outputs[0].token_ids)
        else:
            new_len = len(tokenizer(completion, add_special_tokens=False).input_ids)
        input_lengths.append(input_len)
        new_lengths.append(new_len)
        total_lengths.append(input_len + new_len)
    return texts, input_lengths, total_lengths, new_lengths


def messages_to_prompt(tokenizer, messages):
    def _to_py(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        return obj

    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(_to_py(messages), tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    # fallback
    s = []
    for m in _to_py(messages):
        s.append(f"{m.get('role','user')}: {m.get('content','')}")
    s.append("assistant:")
    return "\n".join(s)


def setup_distributed(local_rank_arg: int = -1):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(local_rank_arg if local_rank_arg >= 0 else 0)))

    if world_size <= 1 or not dist.is_available():
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")
        return 0, 1, device, local_rank

    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        timeout_env = os.environ.get("TORCH_DIST_TIMEOUT", "").strip()
        try:
            timeout_seconds = int(timeout_env) if timeout_env else 10800
        except ValueError:
            timeout_seconds = 10800
        dist.init_process_group(backend=backend, init_method="env://", timeout=timedelta(seconds=timeout_seconds))

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", local_rank))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    return rank, world_size, device, local_rank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base_model", type=str, default=None)
    ap.add_argument("--test_file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", choices=["hf", "vllm"], default="vllm", help="Inference backend")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_new_tokens", type=int, default=2048)
    ap.add_argument("--data_id", type=str, default="OlympiadBench")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--k", type=int, default=1, help="Number of samples per prompt for pass@k (default: 1)")
    ap.add_argument("--seed", type=int, default=None, help="Random seed for sampling (default: unset)")
    ap.add_argument("--log_tag", type=str, default=None, help="Optional tag to append to log filename")
    ap.add_argument("--vllm_tensor_parallel_size", type=int, default=None, help="Tensor parallel size for vLLM")
    ap.add_argument("--vllm_pipeline_parallel_size", type=int, default=None, help="Pipeline parallel size for vLLM")
    ap.add_argument("--vllm_gpu_memory_utilization", type=float, default=None, help="vLLM GPU memory utilization")
    ap.add_argument("--vllm_dtype", type=str, default=None, help="vLLM dtype (e.g., float16, bfloat16)")
    ap.add_argument("--vllm_max_model_len", type=int, default=None, help="vLLM max model length override")
    ap.add_argument("--vllm_max_num_seqs", type=int, default=None, help="vLLM max number of sequences")
    ap.add_argument("--vllm_max_num_batched_tokens", type=int, default=None, help="vLLM max batched tokens")
    ap.add_argument("--local_rank", type=int, default=-1, help="torch.distributed launcher passes this value")
    args = ap.parse_args()

    use_vllm = args.backend == "vllm"
    if use_vllm:
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
    if use_vllm:
        env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
        env_local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if env_world_size > 1 and env_local_rank != 0:
            return
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        rank, world_size, device, local_rank = setup_distributed(args.local_rank)
    is_main = rank == 0
    num_samples = max(1, args.k)

    if args.seed is not None:
        seed = int(args.seed)
        rank_seed = seed + rank
        random.seed(rank_seed)
        np.random.seed(rank_seed)
        torch.manual_seed(rank_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(rank_seed)
        if is_main:
            print(f"[INFO] seed={seed} rank_seed={rank_seed}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%m%d_%H%M%S", time.localtime())
    tag_suffix = ""
    if args.log_tag:
        safe_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(args.log_tag).strip())
        if safe_tag:
            tag_suffix = f"_{safe_tag}"
    log_path = out_dir / f"eval_{ts}_k_{num_samples}{tag_suffix}.log.jsonl"

    tokenizer_source = args.base_model if args.base_model else args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if use_vllm:
        if args.base_model:
            raise ValueError("vLLM backend does not support base_model/PEFT in this script.")
        from vllm import LLM
        # try:
        #     from vllm import LLM
        # except Exception as exc:
        #     raise RuntimeError("vLLM backend requested but vllm is not installed") from exc
        llm_kwargs = {}
        if args.vllm_tensor_parallel_size is not None:
            llm_kwargs["tensor_parallel_size"] = int(args.vllm_tensor_parallel_size)
        if args.vllm_pipeline_parallel_size is not None:
            llm_kwargs["pipeline_parallel_size"] = int(args.vllm_pipeline_parallel_size)
        if args.vllm_gpu_memory_utilization is not None:
            llm_kwargs["gpu_memory_utilization"] = float(args.vllm_gpu_memory_utilization)
        if args.vllm_dtype:
            llm_kwargs["dtype"] = args.vllm_dtype
        if args.vllm_max_model_len is not None:
            llm_kwargs["max_model_len"] = int(args.vllm_max_model_len)
        if args.vllm_max_num_seqs is not None:
            llm_kwargs["max_num_seqs"] = int(args.vllm_max_num_seqs)
        if args.vllm_max_num_batched_tokens is not None:
            llm_kwargs["max_num_batched_tokens"] = int(args.vllm_max_num_batched_tokens)
        model = None
        llm = LLM(model=args.model, **llm_kwargs)
    else:
        if args.base_model:
            from peft import PeftModel

            base_model = AutoModelForCausalLM.from_pretrained(args.base_model)
            model = PeftModel.from_pretrained(base_model, args.model)
        else:
            model = AutoModelForCausalLM.from_pretrained(args.model)
        model = model.eval().to(device)
        model.generation_config.max_new_tokens = args.max_new_tokens
        model.generation_config.max_length = args.max_new_tokens + 4096

    df = load_eval_dataframe(args.test_file)
    if "prompt" in df.columns:
        df["prompt"] = df["prompt"].apply(enforce_boxed_system_prompt)
    prompts = [messages_to_prompt(tokenizer, m) for m in df["prompt"]]
    if is_main:
        print("length======================", len(prompts))
        print(f"[INFO] world_size={world_size} local_rank={local_rank} batch_size={args.batch_size}")
        print(f"[INFO] model_path={args.model} base_model={args.base_model if args.base_model else 'None'}")
        print(f"[INFO] log_path={log_path}")
        print(f"[INFO] pass@k samples per prompt: {num_samples}")
    outputs_all = [[None for _ in range(num_samples)] for _ in prompts]
    input_token_lens_all = [[0 for _ in range(num_samples)] for _ in prompts]
    total_token_lens_all = [[0 for _ in range(num_samples)] for _ in prompts]
    new_token_lens_all = [[0 for _ in range(num_samples)] for _ in prompts]
    batch_size = max(1, args.batch_size)

    indices = list(range(len(prompts)))
    shard_indices = indices[rank :: world_size] if world_size > 1 else indices
    shard_results = []

    def _jsonify(obj):
        if isinstance(obj, dict):
            return {k: _jsonify(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_jsonify(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        return obj

    if is_main:
        print("max_new_tokens==================", args.max_new_tokens)

    for attempt_idx in range(num_samples):
        if is_main and num_samples > 1:
            print(f"[INFO] Generating sample {attempt_idx + 1}/{num_samples}")
        if is_main:
            desc = "Generating" if num_samples == 1 else f"Generating k{attempt_idx + 1}"
            iterator = tqdm(range(0, len(shard_indices), batch_size), desc=desc, unit="batch")
        else:
            iterator = range(0, len(shard_indices), batch_size)
        for offset_start in iterator:
            batch_indices = shard_indices[offset_start : offset_start + batch_size]
            if not batch_indices:
                continue
            batch_prompts = [prompts[idx] for idx in batch_indices]
            if use_vllm:
                seed = None
                if args.seed is not None:
                    seed = args.seed + rank + (attempt_idx * 100000) + offset_start
                texts, input_token_lens, total_token_lens, new_token_lens = generate_batch_vllm(
                    llm,
                    tokenizer,
                    batch_prompts,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    seed=seed,
                )
            else:
                texts, input_token_lens, total_token_lens, new_token_lens = generate_batch(
                    model,
                    tokenizer,
                    batch_prompts,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
            if is_main:
                print("length================", len(texts))
            for local_offset, (full, in_len, tot_len, new_len) in enumerate(
                zip(texts, input_token_lens, total_token_lens, new_token_lens)
            ):
                sample_idx = batch_indices[local_offset]
                shard_results.append(
                    {
                        "index": sample_idx,
                        "attempt": attempt_idx,
                        "output": full,
                        "input_len": in_len,
                        "total_len": tot_len,
                        "new_len": new_len,
                    }
                )
                tail_preview = full.replace("\n", " ")
                if is_main:
                    print(
                        f"[DEBUG] benchmark={args.data_id} model={args.model} gen_sample={sample_idx}, attempt={attempt_idx} tokens(in={in_len}, new={new_len}, total={tot_len}) tail=\"{tail_preview}\"",
                        flush=True,
                    )

    if world_size > 1 and dist.is_initialized():
        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, shard_results)
        if is_main:
            merged = []
            for part in gathered:
                if part:
                    merged.extend(part)
        else:
            merged = None
    else:
        merged = shard_results

    if world_size > 1 and dist.is_initialized():
        dist.barrier()

    if not is_main:
        if dist.is_initialized():
            dist.destroy_process_group()
        return

    if merged is None:
        merged = []

    for item in merged:
        idx = item["index"]
        attempt_idx = item.get("attempt", 0)
        outputs_all[idx][attempt_idx] = item["output"]
        input_token_lens_all[idx][attempt_idx] = item["input_len"]
        total_token_lens_all[idx][attempt_idx] = item["total_len"]
        new_token_lens_all[idx][attempt_idx] = item["new_len"]

    missing = [
        (i, a)
        for i, out_list in enumerate(outputs_all)
        for a, out in enumerate(out_list)
        if out is None
    ]
    if missing:
        raise RuntimeError(f"Missing generations for indices: {missing}")

    df["output"] = [vals[0] for vals in outputs_all]
    if num_samples > 1:
        df["outputs"] = outputs_all

    # Scoring with per-sample logging
    data_id = args.data_id
    scorer = OBJudge()
    res = []
    pass_at_k_scores = []
    avg_at_k_scores = []
    reward_model_col = list(df["reward_model"]) if "reward_model" in df.columns else [{} for _ in range(len(df))]

    def score_single_output(out_text, rm_meta, prompt_val, sample_idx, attempt_idx):
        gt = rm_meta.get("ground_truth", None)
        # Ensure ground truth is stringified for downstream math/string routines
        def _stringify_gt(val):
            import numpy as np

            if isinstance(val, (int, float, np.integer, np.floating, np.generic)):
                return str(val)
            if isinstance(val, (list, tuple)):
                return [str(v) if isinstance(v, (int, float, np.integer, np.floating, np.generic)) else v for v in val]
            return val

        gt = _stringify_gt(gt)
        score_val = 0.0
        is_correct = False
        score = None
        predicted_answer = None

        instructions = rm_meta.get("instructions")
        answer_type = (rm_meta.get("answer_type") or "").lower()
        data_is_text = answer_type == "text" or data_id.lower() in TEXT_DATASETS

        if instructions:
            is_correct = evaluate_ifeval_instructions(out_text, instructions, prompt_val)
            score_val = 1.0 if is_correct else 0.0
            score = is_correct
        elif data_is_text and gt:
            aliases = rm_meta.get("aliases") or []
            is_correct = text_answer_matches(out_text, gt, aliases)
            score_val = 1.0 if is_correct else 0.0
            score = score_val
        elif gt is not None:
            answer_text = out_text.split("</think>")[-1].strip()
            predicted_answer = (
                extract_answer(out_text, extract_from_boxed=True)
                or extract_answer(out_text, extract_from_boxed=False, extract_regex=r"The answer is: (.+)$")
                or extract_answer(answer_text, extract_from_boxed=True)
                or extract_answer(answer_text, extract_from_boxed=False, extract_regex=r"The answer is: (.+)$")
            )
            tail_preview = out_text[-400:].replace("\n", " ")
            print(
                f"[DEBUG] benchmark={data_id} model={args.model} sample={sample_idx}, attempt={attempt_idx} gt={gt} predicted={predicted_answer} tail=\"{tail_preview}\"",
                flush=True,
            )
            if data_id == "OlympiadBench":
                gt0 = gt[0] if isinstance(gt, (list, tuple)) and len(gt) > 0 else gt
                score = (
                    process_results(out_text, gt0, response_extract_from_boxed=True)
                    or process_results(
                        out_text,
                        gt0,
                        response_extract_from_boxed=False,
                        response_extract_regex=r"The answer is: (.+)$",
                    )
                    or scorer.judge(gt0, out_text.split("</think>")[-1].strip(), 1e-8)
                )
            else:
                score = (
                    process_results(out_text, gt, response_extract_from_boxed=True)
                    or process_results(
                        out_text,
                        gt,
                        response_extract_from_boxed=False,
                        response_extract_regex=r"The answer is: (.+)$",
                    )
                )
            is_correct = bool(score)
            if isinstance(score, (int, float)):
                score_val = float(score)
            else:
                score_val = 1.0 if is_correct else 0.0
        return is_correct, score_val, predicted_answer, score

    print(f"[INFO] Writing sample-level logs to {log_path}")
    with open(log_path, "w", encoding="utf-8") as log_f:
        # Emit a header/meta line so downstream consumers see which model was used.
        meta_record = {
            "type": "meta",
            "model": args.model,
            "base_model": args.base_model,
            "data_id": args.data_id,
            "test_file": args.test_file,
            "timestamp": ts,
            "pass_k": num_samples,
            "seed": args.seed,
            "log_tag": args.log_tag,
        }
        log_f.write(json.dumps(_jsonify(meta_record), ensure_ascii=False) + "\n")
        log_f.flush()
        for idx, rm in enumerate(
            tqdm(reward_model_col, desc="Scoring", total=len(df), unit="sample"),
            start=0,
        ):
            row = df.iloc[idx]
            outputs_for_row = row["outputs"] if "outputs" in df.columns else [row["output"]]
            if not isinstance(outputs_for_row, (list, tuple)):
                outputs_for_row = [outputs_for_row]

            attempt_flags = []
            attempt_scores = []
            attempt_pred_answers = []
            attempt_raw_scores = []
            for attempt_idx, out_text in enumerate(outputs_for_row):
                is_correct, score_val, predicted_answer, score = score_single_output(
                    out_text, rm, row.get("prompt"), idx, attempt_idx
                )
                attempt_flags.append(bool(is_correct))
                attempt_scores.append(score_val)
                attempt_pred_answers.append(predicted_answer)
                attempt_raw_scores.append(score)

            first_correct = attempt_flags[0] if attempt_flags else False
            first_score_val = attempt_scores[0] if attempt_scores else 0.0
            first_predicted_answer = attempt_pred_answers[0] if attempt_pred_answers else None
            first_raw_score = attempt_raw_scores[0] if attempt_raw_scores else None
            pass_k_flag = any(attempt_flags)
            sample_avg_score = sum(attempt_scores) / max(1, len(attempt_scores))

            res.append(1.0 if first_correct else 0.0)
            pass_at_k_scores.append(1.0 if pass_k_flag else 0.0)
            avg_at_k_scores.append(float(sample_avg_score))

            sample_id = row.get("sample_id", row.get("id", idx))
            if isinstance(sample_id, (pd.Series, pd.DataFrame)):
                sample_id = idx
            if isinstance(sample_id, Integral):
                sample_id_serializable = int(sample_id)
            elif sample_id is None:
                sample_id_serializable = idx
            else:
                sample_id_serializable = str(sample_id)

            log_record = {
                "sample_index": int(idx),
                "sample_id": sample_id_serializable,
                "prompt_messages": _jsonify(row["prompt"]),
                "prompt_text": prompts[idx],
                "model_output": outputs_for_row[0],
                "predicted_answer": first_predicted_answer,
                "ground_truth": _jsonify(rm.get("ground_truth")),
                "reward_model_meta": _jsonify(rm),
                "raw_score": _jsonify(first_raw_score),
                "score": _jsonify(first_score_val),
                "correct": bool(first_correct),
                "pass_at_1": bool(first_correct),
            }
            if num_samples > 1:
                log_record.update(
                    {
                        "all_model_outputs": outputs_for_row,
                        "attempt_correct": attempt_flags,
                        "pass_at_k": bool(pass_k_flag),
                        "attempt_scores": attempt_scores,
                        "avg_attempt_score": sample_avg_score,
                    }
                )
            log_f.write(json.dumps(_jsonify(log_record), ensure_ascii=False) + "\n")
            log_f.flush()

        # Aggregate metrics (kept in-log for easy parsing)
        acc = sum(res) / max(1, len(res))
        pass_k_avg = (
            sum(pass_at_k_scores) / max(1, len(pass_at_k_scores)) if num_samples > 1 else acc
        )
        avg_k_avg = sum(avg_at_k_scores) / max(1, len(avg_at_k_scores))
        summary_record = {
            "type": "summary",
            "k": num_samples,
            "samples": len(df),
            "acc": acc,
            "pass_at_k": pass_k_avg,
            "avg_at_k": avg_k_avg,
        }
        log_f.write(json.dumps(_jsonify(summary_record), ensure_ascii=False) + "\n")
        log_f.flush()

    df["res"] = res
    if num_samples > 1:
        df["pass_at_k"] = pass_at_k_scores
    acc = sum(res) / max(1, len(res))
    print(f"acc: {acc}")
    if num_samples > 1:
        pass_k_avg = sum(pass_at_k_scores) / max(1, len(pass_at_k_scores))
        print(f"pass@{num_samples}: {pass_k_avg}")
    avg_k_avg = sum(avg_at_k_scores) / max(1, len(avg_at_k_scores))
    print(f"avg@{num_samples}: {avg_k_avg}")
    # out_path = out_dir / f"eval_{ts}.parquet"
    # df.to_parquet(out_path, compression=None)
    # print("Saved:", out_path)
    print(f"Detailed log: {log_path}")

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
