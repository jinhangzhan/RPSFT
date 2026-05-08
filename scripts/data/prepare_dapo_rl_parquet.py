#!/usr/bin/env python3
"""Prepare DAPO/GRPO parquet data for VERL RLFT.

The RL entrypoint expects a parquet file with VERL-style columns:
  data_source, prompt, ability, reward_model, extra_info
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd


DAPO_INSTRUCTION = (
    "A conversation between User and Assistant. The user asks a math question, "
    "and the Assistant solves it step by step. The Assistant first thinks about "
    "the complete reasoning process in the mind enclosed within <think> </think> "
    "tags. Then the Assistant provides a clear answer with the final result "
    "enclosed in \\boxed{} notation."
)


def load_rows(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if args.input:
        path = Path(args.input)
        if path.suffix == ".json":
            return json.loads(path.read_text())
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if path.suffix == ".parquet":
            return pd.read_parquet(path).to_dict(orient="records")
        raise ValueError(f"Unsupported input suffix: {path.suffix}")

    from datasets import load_dataset

    return list(load_dataset(args.dataset, split=args.split))


def first_present(row: Dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def normalize_prompt(prompt: Any) -> List[Dict[str, str]]:
    if isinstance(prompt, list):
        normalized = []
        for message in prompt:
            if isinstance(message, dict):
                role = str(message.get("role", message.get("from", "user")))
                content = message.get("content", message.get("value", ""))
                normalized.append({"role": role, "content": str(content)})
            else:
                normalized.append({"role": "user", "content": str(message)})
        return normalized
    return [{"role": "user", "content": f"{DAPO_INSTRUCTION}\n\nUser: {prompt}\nAssistant:"}]


def normalize_row(row: Dict[str, Any], idx: int, data_source: str) -> Dict[str, Any] | None:
    if {"prompt", "reward_model"}.issubset(row.keys()):
        reward_model = row["reward_model"]
        if not isinstance(reward_model, dict):
            reward_model = {"style": "rule", "ground_truth": reward_model}
        return {
            "data_source": row.get("data_source", data_source),
            "prompt": normalize_prompt(row["prompt"]),
            "ability": row.get("ability", "math"),
            "reward_model": reward_model,
            "extra_info": row.get("extra_info", {"split": "train", "index": idx}),
        }

    question = first_present(row, ["question", "problem", "prompt", "input"])
    answer = first_present(row, ["answer", "solution", "target", "ground_truth"])
    if question is None or answer is None:
        return None
    return {
        "data_source": data_source,
        "prompt": normalize_prompt(question),
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": str(answer)},
        "extra_info": {"split": "train", "index": idx},
    }


def write_jsonl(rows: List[Dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="wh-zhu/dapo")
    parser.add_argument("--split", default="train")
    parser.add_argument("--input", default=None, help="Optional local JSON/JSONL/parquet input")
    parser.add_argument("--output", required=True, help="Output parquet path")
    parser.add_argument("--jsonl-output", default=None)
    parser.add_argument("--data-source", default="math_dapo")
    parser.add_argument("--compression", default="none", choices=["none", "snappy", "gzip", "brotli", "zstd", "lz4"])
    args = parser.parse_args()

    rows = load_rows(args)
    processed = []
    skipped = 0
    for idx, row in enumerate(rows):
        item = normalize_row(row, idx, args.data_source)
        if item is None:
            skipped += 1
            continue
        processed.append(item)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    compression = None if args.compression == "none" else args.compression
    pd.DataFrame(processed).to_parquet(out_path, compression=compression, index=False)

    if args.jsonl_output:
        write_jsonl(processed, Path(args.jsonl_output))

    print(f"rows={len(rows)} written={len(processed)} skipped={skipped} output={out_path}")


if __name__ == "__main__":
    main()
