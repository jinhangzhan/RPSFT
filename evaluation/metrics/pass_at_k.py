#!/usr/bin/env python3
"""
Compute pass@k for Determinant (det) evaluation logs.

This script reads a JSONL produced by evaluators (see evaluation/launcher.py),
groups attempts by `sample_id`, and computes pass@k metrics. It is standalone
and does not modify or depend on the existing evaluation code.

Usage examples:
  python evaluation/metrics/pass_at_k.py --input logs/llama_det_language/in-distribution.jsonl --k 1,3,5
  python evaluation/metrics/pass_at_k.py --input logs/det_l_indist_verify_5/det_l_indist-180.jsonl --k 5 --estimator unbiased

Notes:
 - The script treats lines with a numeric `veri_step` as one attempt.
 - A correct attempt is detected by reward == REWARD_FN["CORRECT_SOLUTION"] (default 5).
 - Two variants are reported:
     * any:    fraction of problems with any correct attempt among the first k attempts
     * unbiased (Chen et al. 2021): 1 - C(n-c, k)/C(n, k) if n>=k; if n<k, uses 1 if c>0 else 0
 - If a problem terminates early upon a correct attempt, n is simply the number
   of attempts logged for that problem.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from typing import Dict, List, Tuple


def parse_args():
    p = argparse.ArgumentParser(description="Compute pass@k from det JSONL logs")
    p.add_argument("--input", required=True, help="Path to evaluation JSONL file")
    p.add_argument("--k", required=True, help="Comma-separated k list, e.g., 1,3,5")
    p.add_argument("--reward_correct", type=int, default=5,
                   help="Reward value considered CORRECT (default: 5)")
    p.add_argument("--estimator", choices=["any", "unbiased", "both"], default="both",
                   help="Which estimator(s) to report")
    p.add_argument("--max_samples", type=int, default=None,
                   help="Optional cap on number of problems to include (for quick sanity checks)")
    return p.parse_args()


def nCk(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    try:
        return math.comb(n, k)
    except AttributeError:
        # Py < 3.8 fallback
        if k == 0:
            return 1
        k = min(k, n - k)
        num = 1
        den = 1
        for i in range(1, k + 1):
            num *= (n - (k - i))
            den *= i
        return num // den


def load_attempts(path: str, reward_correct: int) -> Dict[int, List[Tuple[int, bool]]]:
    """Return dict: sample_id -> list of (veri_step, is_correct) attempts sorted by veri_step."""
    grouped: Dict[int, List[Tuple[int, bool]]] = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            # We only care about per-attempt lines
            if "sample_id" not in obj or "veri_step" not in obj:
                continue
            sid = obj["sample_id"]
            vstep = obj.get("veri_step")
            reward = obj.get("reward")
            if not isinstance(vstep, int):
                continue
            is_correct = (reward == reward_correct)
            grouped[sid].append((vstep, is_correct))
    # sort attempts in each group by veri_step
    for sid in grouped:
        grouped[sid].sort(key=lambda x: x[0])
    return grouped


def compute_any_pass_at_k(attempts: Dict[int, List[Tuple[int, bool]]], k: int) -> float:
    total = len(attempts)
    if total == 0:
        return 0.0
    num_pass = 0
    for sid, att_list in attempts.items():
        first_k = att_list[:k]
        if any(is_corr for _, is_corr in first_k):
            num_pass += 1
    return num_pass / total


def compute_unbiased_pass_at_k(attempts: Dict[int, List[Tuple[int, bool]]], k: int) -> float:
    total = len(attempts)
    if total == 0:
        return 0.0
    s = 0.0
    for sid, att_list in attempts.items():
        n = len(att_list)
        c = sum(1 for _, ok in att_list if ok)
        if c <= 0:
            contrib = 0.0
        elif n >= k:
            contrib = 1.0 - (nCk(n - c, k) / max(1, nCk(n, k)))
        else:
            # If fewer than k attempts exist, treat as pass if any correct exists
            contrib = 1.0
        s += contrib
    return s / total


def main():
    args = parse_args()
    k_list = [int(x) for x in args.k.split(",") if x.strip()]
    k_list = sorted(set(k_list))

    grouped = load_attempts(args.input, args.reward_correct)

    # Optionally cap number of problems for quick checks
    if args.max_samples is not None:
        sids = sorted(grouped.keys())[: args.max_samples]
        grouped = {sid: grouped[sid] for sid in sids}

    result = {
        "file": args.input,
        "num_problems": len(grouped),
        "avg_attempts": sum(len(v) for v in grouped.values()) / max(1, len(grouped)),
        "k_list": k_list,
        "estimator": args.estimator,
        "metrics": {},
    }

    if args.estimator in ("any", "both"):
        any_res = {}
        for k in k_list:
            any_res[str(k)] = compute_any_pass_at_k(grouped, k)
        result["metrics"]["any"] = any_res

    if args.estimator in ("unbiased", "both"):
        unb_res = {}
        for k in k_list:
            unb_res[str(k)] = compute_unbiased_pass_at_k(grouped, k)
        result["metrics"]["unbiased"] = unb_res

    # Pretty print JSON result
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

