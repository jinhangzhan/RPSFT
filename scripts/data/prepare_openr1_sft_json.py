#!/usr/bin/env python3
"""Prepare OpenR1-style supervised data for the SFT trainers.

The SFT code expects a JSON list with:
  {"conversations": [{"from": "human", "value": ...}, {"from": "gpt", "value": ...}]}

The default source matches the dataset cited in the paper, but this script also
accepts local JSON/JSONL/parquet files with common question/answer columns.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd


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


def message_content(message: Any) -> str:
    if isinstance(message, dict):
        content = message.get("content", message.get("value", ""))
        if isinstance(content, list):
            return "\n".join(str(part.get("text", part)) if isinstance(part, dict) else str(part) for part in content)
        return str(content)
    return str(message)


def first_present(row: Dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def extract_question(row: Dict[str, Any]) -> str:
    prompt = first_present(row, ["prompt", "question", "problem", "input"])
    if isinstance(prompt, list):
        user_messages = [
            msg for msg in prompt
            if not isinstance(msg, dict) or msg.get("role", msg.get("from")) in ("user", "human")
        ]
        return message_content(user_messages[-1] if user_messages else prompt[-1])
    return message_content(prompt)


def extract_answer(row: Dict[str, Any]) -> str:
    target = first_present(row, ["target", "answer", "solution", "demonstration", "output"])
    if isinstance(target, list):
        return message_content(target[0] if target else "")
    return message_content(target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="wh-zhu/train_openr1_4k")
    parser.add_argument("--split", default="train")
    parser.add_argument("--input", default=None, help="Optional local JSON/JSONL/parquet input")
    parser.add_argument("--output", required=True, help="Output train_openr1.sft.json path")
    args = parser.parse_args()

    rows = load_rows(args)
    output = []
    skipped = 0
    for row in rows:
        question = extract_question(row).strip()
        answer = extract_answer(row).strip()
        if not question or not answer:
            skipped += 1
            continue
        output.append(
            {
                "conversations": [
                    {"from": "human", "value": question},
                    {"from": "gpt", "value": answer},
                ]
            }
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"rows={len(rows)} written={len(output)} skipped={skipped} output={out_path}")


if __name__ == "__main__":
    main()
